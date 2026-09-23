# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma model, ModelState, and Sampler for vLLM.

Single Gemma4 backbone run in two modes (like YOCO):
- encoder mode: causal attention, writes KV cache
- decoder mode: bidirectional attention, reads encoder KV, doesn't write

Same weights, same layers. The only decoder-unique component is a
self-conditioning MLP.

Multimodal support: the model always includes a vision tower (shared with Gemma4).
Images are encoded through the vision tower and projected into the LM embedding space
via Gemma4MultimodalEmbedder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4Model
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty
from vllm.v1.worker.gpu.states import RequestState

from .diffusion_gemma_reason_counters import ReasoningCounterTrace
from .diffusion_gemma_vision_rmsnorm import (
    replace_diffusion_gemma_vision_rms_norms,
)
from .diffusion_gemma_vision_rope import (
    replace_diffusion_gemma_vision_attentions,
)
from .interfaces import (
    SupportsMultiModal,
    SupportsQuant,
)

logger = init_logger(__name__)


class DiffusionGemmaSelfConditioning(nn.Module):
    """Gated MLP that processes soft embeddings from the previous denoising step.

    Structurally identical to Gemma4MLP but with self_conditioning_size
    and post_norm without learned scale.
    """

    def __init__(
        self, hidden_size: int, self_conditioning_size: int, eps: float = 1e-6
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.post_norm = RMSNorm(hidden_size, eps=eps, has_weight=False)
        self.gate_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.down_proj = nn.Linear(self_conditioning_size, hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        soft_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        sc_signal = self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
        return self.post_norm(inputs_embeds + sc_signal)


# ---------------------------------------------------------------------------
# Multimodal processing info (overrides Gemma4 config type check)
# ---------------------------------------------------------------------------


class DiffusionGemmaProcessingInfo(Gemma4ProcessingInfo):
    """Processing info for DiffusionGemma.

    Overrides ``get_hf_config`` to accept ``DiffusionGemmaConfig``
    (which inherits from ``PretrainedConfig``, not ``Gemma4Config``).
    Supports image and video modalities.
    """

    def get_hf_config(self):
        # DiffusionGemmaConfig doesn't inherit from Gemma4Config, so we
        # accept any PretrainedConfig here.
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # DiffusionGemma supports image and video inputs.
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _softcap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    # fp32 before tanh for numerical stability (matches HF DiffusionGemma).
    # Compiling fuses the cast/div/tanh/mul into one elementwise kernel over
    # the [num_tokens, vocab] logits instead of four separate passes.
    logits = logits.float()
    return torch.tanh(logits / cap) * cap


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=DiffusionGemmaProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class DiffusionGemmaForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsQuant,
):
    """DiffusionGemma for vLLM.

    Single Gemma4 backbone that switches between encoder and decoder mode.
    The encoder path uses standard Gemma4 layers (causal attention, KV write).
    The decoder path uses the same weights with bidirectional attention and
    KV read-only, plus self-conditioning.

    Always includes a vision tower (same as Gemma4) for image understanding.

    In practice, the model's forward() dispatches based on the `mode` kwarg
    set by DiffusionGemmaModelState.prepare_inputs().
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.decoder.": "model.",
            "model.encoder.language_model.": "model.",
            "model.encoder.vision_tower.": "vision_tower.",
            "model.encoder.embed_vision.": "embed_vision.",
        },
        orig_to_new_substr={
            ".experts.": ".moe.experts.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        return DiffusionGemmaModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model_dtype = vllm_config.model_config.dtype

        # DiffusionGemma's full-attention layers have NO v_proj — V is
        # computed from k_proj's output (`value_states = key_states` before
        # k_norm in `DiffusionGemmaDecoderTextAttention.forward`). This is
        # the "k_eq_v" variant in our Gemma4 backbone. The checkpoint has no
        # v_proj weights for full-attention layers; without this flag they
        # would silently load with random V projections.
        text_config.attention_k_eq_v = True

        # ---- Vision tower ----
        vision_config = getattr(config, "vision_config", None)
        self.embed_vision: Gemma4MultimodalEmbedder | None
        if vision_config is not None:
            quant_config = vllm_config.quant_config
            tower_quant: QuantizationConfig | None
            if quant_config and quant_config.get_name() in [
                "bitsandbytes",
                "torchao",
                "compressed-tensors",
            ]:
                tower_quant = quant_config
            else:
                quantizable = (
                    vision_config.hidden_size % 64 == 0
                    and vision_config.intermediate_size % 64 == 0
                )
                tower_quant = quant_config if quantizable else None

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.vision_tower = AutoModel.from_config(config=vision_config)
                self.embed_vision = Gemma4MultimodalEmbedder(
                    vision_config,
                    text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_vision"),
                )
                recursive_replace_linear(
                    self.vision_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
                if os.getenv("VLLM_DIFFUSION_FUSE_VISION_RMSNORM") == "1":
                    replaced = replace_diffusion_gemma_vision_rms_norms(
                        self.vision_tower
                    )
                    logger.info(
                        "Enabled exact fused Gemma4 vision RMSNorm for %d modules",
                        replaced,
                    )
                if os.getenv("VLLM_DIFFUSION_FUSE_VISION_ROPE") == "1":
                    replaced = replace_diffusion_gemma_vision_attentions(
                        self.vision_tower
                    )
                    logger.info(
                        "Enabled exact fused Gemma4 vision RoPE for %d layers",
                        replaced,
                    )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # ---- Language backbone (Gemma4Model) ----
        # Use maybe_prefix to ensure correct weight name prefixes for
        # quantization. The quantization config uses hf_to_vllm_mapper to
        # match checkpoint weight names to model parameter names.
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if os.getenv("VLLM_DIFFUSION_FUSE_TEXT_NORM") == "1":
            if (
                os.getenv("VLLM_BATCH_INVARIANT") != "1"
                or self.model_dtype != torch.bfloat16
                or not current_platform.is_cuda()
                or not current_platform.is_device_capability(90)
                or text_config.hidden_size != 2816
                or vllm_config.quant_config is not None
            ):
                raise ValueError("Fused text norms require the H200 BF16 profile")
            for layer in self.model.modules():
                if not isinstance(layer, Gemma4DecoderLayer):
                    continue
                attention = layer.self_attn
                attention.use_fused_batch_invariant_norm = (
                    not attention.is_kv_shared_layer
                    and attention.num_heads == 16
                    and (attention.head_dim, attention.num_kv_heads)
                    in ((256, 8), (512, 2))
                )
                layer.use_fused_batch_invariant_norm = layer.enable_moe_block
            logger.info("Enabled fused batch-invariant DiffusionGemma text norms")

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
        )

        if text_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # HF DiffusionGemma applies the final-logit softcap in fp32, before
        # any other processing. Do it manually in `compute_logits` so the
        # LogitsProcessor only handles the lm_head GEMM.
        self.final_logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=None,
        )

        sc_size = (
            getattr(config, "self_conditioning_size", None)
            or text_config.intermediate_size
        )
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=text_config.hidden_size,
            self_conditioning_size=sc_size,
            eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

    def compute_self_conditioning(
        self,
        inputs_embeds: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        embed_weight = self.model.embed_tokens.weight
        soft_embeds = torch.matmul(
            probs.to(embed_weight.dtype), embed_weight
        ) * self.model.normalizer.to(inputs_embeds.dtype)
        return self.self_conditioning(inputs_embeds, soft_embeds)

    # ------------------------------------------------------------------ #
    # Multimodal: reuse Gemma4's image parsing, processing & embedding
    # ------------------------------------------------------------------ #
    # The vision tower, pooler, embed_vision, and their processing logic
    # are architecturally identical to Gemma4.  Delegate to avoid
    # maintaining a duplicate copy.

    # The shared image encoder supports LoRA; DiffusionGemma does not.
    _enable_mm_lora = False
    mm_prefix_clamp_sliding_window = True

    _parse_and_validate_image_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_image_input
    )
    _parse_and_validate_video_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_video_input
    )
    _parse_and_validate_multimodal_inputs = (
        Gemma4ForConditionalGeneration._parse_and_validate_multimodal_inputs
    )
    _encoder_chunk = staticmethod(Gemma4ForConditionalGeneration._encoder_chunk)
    _process_image_input = Gemma4ForConditionalGeneration._process_image_input
    _process_video_input = Gemma4ForConditionalGeneration._process_video_input
    embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["embed_vision"],
            tower_model=["vision_tower"],
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.final_logit_softcapping is not None:
            logits = _softcap_logits(logits, self.final_logit_softcapping)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from checkpoint.

        Checkpoint layout (HF DiffusionGemma):
          model.encoder.vision_tower.*            → vision tower
          model.encoder.embed_vision.*            → vision embedder
          model.encoder.language_model.layers.*   → backbone
          model.decoder.layers.*                  → backbone (tied)
          model.decoder.embed_tokens.*            → embeddings
          model.decoder.self_conditioning.*       → self-conditioning MLP
          lm_head.*                               → LM head (tied)

        We load encoder weights into our single ``Gemma4Model`` backbone,
        skip duplicate decoder backbone weights, handle vision tower and
        self-conditioning separately.
        """
        sc_params = dict(
            (n, p)
            for n, p in self.named_parameters()
            if n.startswith("self_conditioning.")
        )

        # Collect vision tower + embedder parameters AND buffers for manual
        # loading.  The HF vision tower registers std_bias / std_scale as
        # buffers (not parameters) when config.standardize is True, so we
        # must include named_buffers() to avoid "not found in model" warnings.
        vision_params: dict[str, torch.Tensor] = {}
        for n, p in self.named_parameters():
            if n.startswith(("vision_tower.", "embed_vision.")):
                vision_params[n] = p
        for n, b in self.named_buffers():
            if n.startswith(("vision_tower.", "embed_vision.")):
                vision_params[n] = b

        def _remap_weights():
            # Use full weight names (including suffixes like .weight_scale,
            # .weight_packed) for dedup instead of just the base layer name. Critical
            # for quantized checkpoints where each weight has multiple tensors;
            # tracking only base names skips scales as duplicates.
            seen_weights: set[str] = set()
            for name, weight in weights:
                # Self-conditioning lives under model.decoder.self_conditioning.*
                # in the checkpoint but at self_conditioning.* in our model.
                if "self_conditioning" in name:
                    sc_name = name.split("self_conditioning.", 1)[1]
                    sc_name = "self_conditioning." + sc_name
                    if sc_name in sc_params:
                        sc_params[sc_name].data.copy_(weight)
                    continue

                # Vision tower: model.encoder.vision_tower.* → vision_tower.*
                # In HF, the vision tower is a sibling of language_model
                # under the encoder module.
                if name.startswith("model.encoder.vision_tower."):
                    vt_name = name[len("model.encoder.") :]
                    if vt_name in vision_params:
                        vision_params[vt_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Vision tower weight %s (mapped to %s) not found in model",
                            name,
                            vt_name,
                        )
                    continue

                # Vision embedder: model.encoder.embed_vision.* → embed_vision.*
                if name.startswith("model.encoder.embed_vision."):
                    ev_name = name[len("model.encoder.") :]
                    if ev_name in vision_params:
                        vision_params[ev_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Embed vision weight %s (mapped to %s) not found in model",
                            name,
                            ev_name,
                        )
                    continue

                # Skip vestigial embed_vision.embedding weights.
                if "embed_vision.embedding." in name:
                    continue

                # Encoder backbone → model.*
                if name.startswith("model.encoder.language_model."):
                    name = name.replace("model.encoder.language_model.", "model.")
                # Decoder backbone → model.* (skip exact duplicates)
                elif name.startswith("model.decoder."):
                    name = name.replace("model.decoder.", "model.")

                # Skip only if we've seen the exact same weight name (including scales)
                if name in seen_weights:
                    continue
                seen_weights.add(name)
                yield name, weight

        # Delegate to Gemma4ForCausalLM.load_weights for the backbone,
        # which handles stacked params, MoE, k_eq_v, etc.
        # Temporarily set self.config to text_config since Gemma4's
        # load_weights expects it (e.g. tie_word_embeddings, layer_types).
        from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

        saved_config = self.config
        self.config = self.model.config
        try:
            Gemma4ForCausalLM.load_weights(self, _remap_weights())
        finally:
            self.config = saved_config

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _compiled_sample_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,  # [num_decode] int64 → slot indices
    decode_idx: torch.Tensor,  # [num_decode] int64 → position in num_reqs
    all_slots: torch.Tensor,  # [num_reqs] int64 → all slot indices
    valid_canvas_len: torch.Tensor,  # [num_decode] int64 → real canvas length (<=CL)
    # State tensors (modified in-place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    argmax_canvas: torch.Tensor,  # [max_num_reqs, CL]
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs]
    confident_tensor: torch.Tensor,  # [max_num_reqs]
    sc_embeds: torch.Tensor,  # [max_num_reqs, CL, hidden]
    embed_weight: torch.Tensor,  # [vocab, hidden]
    normalizer: torch.Tensor,
    history: torch.Tensor,  # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    # Output tensors (modified in-place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    # Sampler config
    entropy_bound: float,
    # Tensor-parallel vocab sharding for the self-conditioning matmul.
    # ``embed_weight`` is vocab-sharded ([vocab/tp, hidden]) while ``probs``
    # spans the full vocab; [sc_vocab_start, sc_vocab_end) is this rank's slice.
    sc_vocab_start: int,
    sc_vocab_end: int,
    tp_size: int,
    tp_group_name: str,
) -> torch.Tensor:
    """Compiled decode step: temperature → Gumbel sample → probs/confidence →
    accept/renoise → convergence, all as vectorized PyTorch ops.

    Returns the temperature-scaled logits ``[num_decode, CL, vocab]`` so the
    caller can compute logprobs outside the compiled region."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    # ---- Phase 2: Temperature scaling + Gumbel-max sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    # Gumbel-max trick: argmax(logits/T + Gumbel) ~ sample from softmax(logits/T)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    # Zero noise when temp==0 (greedy)
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    # ---- Phase 3: Probs, self-conditioning, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    token_entropy = -(probs * log_probs).sum(dim=-1)  # [num_decode, CL]
    # A canvas truncated near max_model_len is zero-padded up to CL by the
    # caller; those padded rows are uniform (max entropy, argmax 0), so they
    # never trigger early convergence and are stable, and only the real
    # ``valid_canvas_len`` tokens are committed (num_sampled below).
    mean_entropy = token_entropy.mean(dim=-1)  # [num_decode]
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    # ---- Phase 4: Entropy-bound acceptance mask ----
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Phase 5: Post-sample ----
    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    # Step update: +1 for denoise, reset to 0 for commit
    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    # Random tokens for renoise / canvas reinit
    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )

    # Compute denoise canvas (accept/renoise)
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)

    # Canvas: commit → random reinit, denoise → accept/renoise result
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    # History: write argmax_tokens for denoise requests at circular position
    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    # Argmax canvas: update for denoise, preserve for commit
    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    # History length: increment for denoise, reset for commit
    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    # Sampled output: commit → emit argmax_canvas, denoise → 0 (pre-zeroed)
    sampled[decode_idx] = argmax_canvas[decode_slots].to(
        sampled.dtype
    ) * is_commit.unsqueeze(1).to(sampled.dtype)
    # Commit only the real canvas length (== CL except for a canvas truncated
    # near max_model_len); the padded tail positions are never emitted.
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    # Commit done → denoise next (False); denoise converged → commit next (True)
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # SC soft embedding: store ``probs @ embed_weight`` (the value the next step's
    # self-conditioning MLP consumes) only for slots that will denoise next — i.e.
    # this step denoised AND it isn't about to commit (is_encoder_phase now False).
    # Masking here (rather than in the consumer) lets _apply_self_conditioning read
    # sc_embeds directly. Storing the [.., hidden] soft embed instead of the full
    # [.., vocab] probs avoids a giant persistent buffer.
    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    # Self-conditioning soft embed = probs @ embed_tokens.weight. Under tensor
    # parallelism the embedding is vocab-sharded ([vocab/tp, hidden]) while
    # probs spans the full vocab, so each rank multiplies its local vocab slice
    # [sc_vocab_start, sc_vocab_end) and the partials are summed across ranks.
    local_probs = probs[..., sc_vocab_start:sc_vocab_end].to(embed_weight.dtype)
    soft_embeds = torch.matmul(
        local_probs, embed_weight[: sc_vocab_end - sc_vocab_start]
    )
    if tp_size > 1:
        soft_embeds = torch.ops.vllm.all_reduce(soft_embeds, group_name=tp_group_name)
    soft_embeds = soft_embeds * normalizer
    sc_embeds[decode_slots] = (soft_embeds * sc_keep).to(sc_embeds.dtype)

    # Overwrite canvas with argmax for newly converged denoise requests
    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return scaled


@dataclass
class _ReadReasoning:
    positions: torch.Tensor
    positions_cpu: tuple[int, ...]
    forbidden_token_ids: torch.Tensor
    allowed_token_ids: torch.Tensor
    steps: int
    generator: torch.Generator
    step: int = 0
    convergence_stop: bool = False
    terminal: bool = False
    previous_argmax: torch.Tensor | None = None
    answer_noise: str = "candidates"
    token_noise: str = "allowed"
    token_proposals: str = "allowed"
    schedule_steps: int | None = None
    entropy_bound: float | None = None
    round_feedback_logits: bool = False
    preserve_feedback: bool = False
    answer_feedback: bool = False
    unmasked_feedback: bool = False
    known_positions_mask: torch.Tensor | None = None
    delayed_suffix_positions: torch.Tensor | None = None
    delayed_suffix_positions_cpu: tuple[int, ...] = ()


@dataclass
class _ReadAnswerMean:
    steps: int
    count: int = 0
    log_probability_sum: torch.Tensor | None = None


class DiffusionGemmaRequestStates:
    """Pre-allocated GPU tensors for DiffusionGemma per-request state.

    Follows the indexed-slot pattern used by ``RequestState``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
        hidden_size: int,
        stability_threshold: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = stability_threshold
        self.device = device
        self._fast_read_admission = (
            os.environ.get("VLLM_DIFFUSION_READ_FAST_ADMISSION") == "1"
        )

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length]
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Step counter (counts up from 0 to max_denoising_steps)
        self.step = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        # Accepted canvas history for stability check
        self.accepted_canvas_history = torch.zeros(
            max_num_reqs,
            stability_threshold,
            canvas_length,
            dtype=torch.int64,
            device=device,
        )
        self.accepted_canvas_history_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # Latest argmax(processed_logits) per slot — what we COMMIT.
        # NOT `current_canvas` (which is the post-renoise stochastic input for
        # the next denoise step). We keep this separate from `canvas` because
        # canvas gets renoised in-place during denoise, while argmax_canvas is
        # the deterministic best-guess we ultimately emit.
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )

        # Per-slot prompt length (set by add_request).
        self.prompt_len = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

        # Per-slot confidence flag, set by the sampler each step.
        self.confident = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Seed canvas and answer positions for constrained reads.
        self.read_requests: dict[
            int, tuple[torch.Tensor, torch.Tensor, tuple[int, ...], bool, bool]
        ] = {}
        self.read_progress: dict[int, tuple[int, bool]] = {}
        self.read_reasoning: dict[int, _ReadReasoning] = {}
        self.reasoning_counter_trace: ReasoningCounterTrace | None = None
        self.read_answer_means: dict[int, _ReadAnswerMean] = {}
        self.read_self_conditioning: set[int] = set()
        self.read_canvas_self_conditioning: set[int] = set()

        # Per-slot self-conditioning soft embedding (probs @ embed_weight) from
        # the previous denoise step. Storing the [.., hidden] soft embed instead
        # of the full [.., vocab] distribution shrinks this buffer by
        # vocab/hidden (~170x) and moves the matmul to denoise time; the result
        # is identical (SC consumes probs @ embed_weight anyway).
        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices: torch.Tensor) -> None:
        """Initialize canvas with random tokens for the given slots.

        `slot_indices` must already be on device to avoid a cpu->gpu sync.
        """
        n = slot_indices.shape[0]
        self.canvas[slot_indices] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )

    def add_request(self, slot_idx: int, *, read_only: bool = False) -> None:
        skip_read_init = self._fast_read_admission and read_only
        if self.reasoning_counter_trace is not None:
            self.reasoning_counter_trace.abort(slot_idx, "slot_reused")
        self.read_requests.pop(slot_idx, None)
        self.read_progress.pop(slot_idx, None)
        self.read_reasoning.pop(slot_idx, None)
        self.read_answer_means.pop(slot_idx, None)
        self.read_self_conditioning.discard(slot_idx)
        self.read_canvas_self_conditioning.discard(slot_idx)
        self.is_encoder_phase[slot_idx].fill_(True)
        if not skip_read_init:
            self.init_canvas(async_tensor_h2d([slot_idx], device=self.device))
        self.step[slot_idx].fill_(0)
        self.accepted_canvas_history_len[slot_idx].fill_(0)
        if not skip_read_init:
            self.self_conditioning_embeds[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        if self.reasoning_counter_trace is not None:
            self.reasoning_counter_trace.abort(slot_idx)
        read_only = self.read_requests.pop(slot_idx, None) is not None
        self.read_progress.pop(slot_idx, None)
        self.read_reasoning.pop(slot_idx, None)
        self.read_answer_means.pop(slot_idx, None)
        self.read_self_conditioning.discard(slot_idx)
        self.read_canvas_self_conditioning.discard(slot_idx)
        self.is_encoder_phase[slot_idx].fill_(False)
        self.accepted_canvas_history_len[slot_idx].fill_(0)
        if not (self._fast_read_admission and read_only):
            self.self_conditioning_embeds[slot_idx] = 0


class DiffusionGemmaModelState(ModelState):
    """ModelState for DiffusionGemma.

    Single Gemma4 backbone in two modes:
    - encoder mode (num_draft_tokens == 0): causal attention, writes KV
    - decoder mode (num_draft_tokens > 0): bidirectional attention, reads KV
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        self._mm_prefix_mode = os.environ.get("VLLM_DIFFUSION_MM_PREFIX_MODE", "off")
        if self._mm_prefix_mode not in ("off", "hf_both", "sliding_only"):
            raise ValueError("Unknown VLLM_DIFFUSION_MM_PREFIX_MODE")
        if self._mm_prefix_mode != "off":
            if not self.model_config.is_mm_prefix_lm:
                raise ValueError(
                    "Experimental diffusion MM prefix requires config flag"
                )
            if not vllm_config.scheduler_config.disable_chunked_mm_input:
                raise ValueError(
                    "Experimental diffusion MM prefix requires atomic images"
                )
            if self.model_config.hf_text_config.use_bidirectional_attention != "vision":
                raise ValueError(
                    "Experimental diffusion MM prefix requires vision config"
                )
            mm_config = self.model_config.multimodal_config
            assert mm_config is not None
            image_limit = mm_config.get_limit_per_prompt("image")
            video_limit = mm_config.get_limit_per_prompt("video")
            if video_limit:
                raise ValueError(
                    "Experimental diffusion MM prefix currently requires "
                    "the video limit to be zero"
                )
            max_ranges = image_limit
            if max_ranges < 1:
                raise ValueError(
                    "Experimental diffusion MM prefix requires enabled media"
                )
            shape = (self.max_num_reqs, max_ranges, 2)
            self._mm_prefix_range_cpu = torch.zeros(
                shape, dtype=torch.int32, pin_memory=True
            )
            self._mm_prefix_range_np = self._mm_prefix_range_cpu.numpy()
            self._mm_prefix_range_buf = torch.zeros(
                shape, dtype=torch.int32, device=device
            )
            self._mm_prefix_range_copy_event = torch.cuda.Event(blocking=True)
            self._mm_prefix_range_copy_pending = False
        else:
            self._mm_prefix_range_cpu = None
            self._mm_prefix_range_np = None
            self._mm_prefix_range_buf = None
            self._mm_prefix_range_copy_event = None
            self._mm_prefix_range_copy_pending = False
        self._full_attn_layer_idxs = {
            i
            for i, layer_type in enumerate(self.model_config.hf_text_config.layer_types)
            if layer_type == "full_attention"
        }
        self._mm_prefix_trace_file = os.environ.get(
            "VLLM_DIFFUSION_MM_PREFIX_TRACE_FILE"
        )

        # Per-step MM data produced by prepare_inputs_embeds and consumed by
        # prepare_inputs.  Stored as raw (mm_embeds, is_mm_embed) so that
        # prepare_inputs can call embed_input_ids directly into the
        # persistent _inputs_embeds_buf, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds: tuple[list[torch.Tensor], torch.Tensor] | None = None

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 32

        text_config = self.model_config.hf_text_config
        self.gen_config = self.model_config.try_get_generation_config()
        max_denoising_steps = (
            diffusion_config.max_denoising_steps if diffusion_config else None
        ) or self.gen_config.get("max_denoising_steps", 48)
        self.diffusion_states = DiffusionGemmaRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            vocab_size=self.model_config.get_vocab_size(),
            max_denoising_steps=max_denoising_steps,
            device=device,
            hidden_size=text_config.hidden_size,
            # In Transformers, `stability_threshold=1` (the default) means the current
            # step must match the previous step. In vLLM, the history buffer includes
            # the current step, so we add 1 to match the same behavior.
            stability_threshold=self.gen_config["stability_threshold"] + 1,
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self._fuse_read_prefill = (
            os.environ.get("VLLM_DIFFUSION_FUSE_READ_PREFILL") == "1"
        )
        if self._fuse_read_prefill:
            self._split_query_staging = UvaBackedTensor(
                2 * self.max_num_reqs + 1, dtype=torch.int32
            )
            self._split_query = torch.zeros(
                2 * self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self._split_logits_staging = UvaBackedTensor(
                self.max_num_reqs, dtype=torch.int32
            )
            self._split_read_staging = UvaBackedTensor(
                self.max_num_reqs, dtype=torch.bool
            )
            self._split_seq_lens = torch.zeros(
                2 * self.max_num_reqs, dtype=torch.int32, device=device
            )
            self._split_causal = torch.zeros(
                2 * self.max_num_reqs, dtype=torch.bool, device=device
            )
            self._split_block_tables: dict[int, torch.Tensor] = {}
            self._split_mm_prefix_range_buf = (
                torch.zeros(
                    (2 * self.max_num_reqs, *self._mm_prefix_range_buf.shape[1:]),
                    dtype=self._mm_prefix_range_buf.dtype,
                    device=device,
                )
                if self._mm_prefix_range_buf is not None
                else None
            )

        # Persistent inputs_embeds buffer — required so FULL CUDA graph
        # capture and runtime point at the SAME memory address.
        # `prepare_dummy_inputs` (capture path) and `prepare_inputs` (runtime
        # path) both must hand the captured graph a tensor at this address.
        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens,
            text_config.hidden_size,
            dtype=self.model_config.dtype,
            device=device,
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config
        sampler_cfg = gen.get("sampler_config") or {}
        if "EntropyBound" not in sampler_cfg.get("_cls_name", ""):
            raise ValueError("DiffusionGemma requires an EntropyBound sampler_config")
        entropy_bound = sampler_cfg.get("entropy_bound")
        if entropy_bound is None or entropy_bound <= 0:
            raise ValueError(
                f"entropy_bound must be a positive float (got {entropy_bound})"
            )
        # The self-conditioning matmul (probs @ embed_tokens.weight) runs over a
        # vocab-parallel embedding shard. Hand the sampler this rank's vocab
        # slice and TP group so it can all-reduce the partial products.
        embed_tokens = self.model.model.embed_tokens
        shard = embed_tokens.shard_indices
        tp_group = get_tp_group()
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=gen["t_min"],
            t_max=gen["t_max"],
            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            embed_weight=embed_tokens.weight,
            normalizer=self.model.model.normalizer,
            sc_vocab_start=shard.org_vocab_start_index,
            sc_vocab_end=shard.org_vocab_end_index,
            tp_size=tp_group.world_size,
            tp_group_name=tp_group.unique_name,
        ), None

    def apply_staged_writes(self) -> None:
        pass

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        params = new_req_data.sampling_params
        read_only = params is not None and bool(
            (params.extra_args or {}).get("diffusion_read_only")
        )
        self.diffusion_states.add_request(req_index, read_only=read_only)
        if not new_req_data.req_id.startswith("_warmup_"):
            prompt_len = len(new_req_data.prompt_token_ids)
            self.diffusion_states.prompt_len[req_index].fill_(prompt_len)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        if not self.supports_mm_inputs:
            return None

        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)

        if not mm_embeds:
            # No MM tokens in this batch (e.g. all-decode step).
            # prepare_inputs will use embed_input_ids (text-only) directly.
            self._pending_mm_embeds = None
            return None

        # Stash raw MM ingredients for prepare_inputs to merge directly
        # into the persistent buffer, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds = (mm_embeds, is_mm_embed)
        return None

    def _apply_self_conditioning(
        self,
        decode_slots_np: np.ndarray,
        decode_idx_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        inputs_embeds: torch.Tensor,
        sc_embeds: torch.Tensor,
        num_logits_np: np.ndarray | None = None,
    ) -> None:
        # One self-conditioning MLP call per decode request, over that request's
        # query span [start, end) = its canvas. The span is the full canvas (CL)
        # or, for the final canvas truncated near max_model_len, fewer than CL
        # positions. sc_embeds already holds probs @ embed_weight from the prior
        # denoise step, masked to zero by the sampler for slots not denoising
        # this step; only the MLP runs here. CPU metadata -> no GPU syncs.
        for slot, idx in zip(decode_slots_np.tolist(), decode_idx_np.tolist()):
            start = int(query_start_loc_np[idx])
            end = int(query_start_loc_np[idx + 1])
            if num_logits_np is not None:
                start = end - int(num_logits_np[idx])
            canvas = slice(start, end)
            if slot in self.diffusion_states.read_requests:
                reasoning = self.diffusion_states.read_reasoning.get(slot)
                if (
                    reasoning is not None
                    and reasoning.delayed_suffix_positions is not None
                    and reasoning.step < reasoning.steps
                    and not reasoning.terminal
                ):
                    count = sum(
                        position < end - start
                        for position in reasoning.delayed_suffix_positions_cpu
                    )
                    if count:
                        positions = reasoning.delayed_suffix_positions[:count]
                        inputs_embeds[start + positions] = self.model.embed_input_ids(
                            torch.zeros_like(positions)
                        )
                if (
                    reasoning is not None
                    and reasoning.known_positions_mask is not None
                    and self.diffusion_states.read_progress[slot][1]
                ):
                    soft = (
                        sc_embeds[slot, : end - start].to(inputs_embeds.dtype)
                        if slot in self.diffusion_states.read_self_conditioning
                        else torch.zeros_like(inputs_embeds[canvas])
                    )
                    soft = torch.where(
                        reasoning.known_positions_mask[: end - start, None],
                        inputs_embeds[canvas],
                        soft,
                    )
                    inputs_embeds[canvas] = self.model.self_conditioning(
                        inputs_embeds[canvas], soft
                    )
                    continue
            if (
                slot in self.diffusion_states.read_requests
                and slot not in self.diffusion_states.read_self_conditioning
            ):
                # The first read pass has no previous self-conditioning signal.
                inputs_embeds[canvas] = self.model.self_conditioning.post_norm(
                    inputs_embeds[canvas]
                )
                continue
            soft = sc_embeds[slot, : end - start]
            inputs_embeds[canvas] = self.model.self_conditioning(
                inputs_embeds[canvas], soft.to(inputs_embeds.dtype)
            )

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        states = self.diffusion_states
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs

        # Write into the PERSISTENT inputs_embeds buffer so FULL CUDA graph
        # replay sees the latest values at the captured address.
        num_tokens_padded = input_batch.num_tokens_after_padding
        inputs_embeds = self._inputs_embeds_buf[:num_tokens_padded]

        # Populate embeddings: merge MM features when available,
        # otherwise embed input_ids as text-only.
        input_ids = input_batch.input_ids[:num_tokens]
        if self._pending_mm_embeds is not None:
            mm_embeds, is_mm_embed = self._pending_mm_embeds
            self._pending_mm_embeds = None
            inputs_embeds[:num_tokens].copy_(
                self.model.embed_input_ids(
                    input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
            )
        else:
            inputs_embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))

        # Apply self-conditioning ONLY for denoising decode requests.
        if input_batch.num_draft_tokens > 0 and self._req_id_to_index:
            slots_np = input_batch.idx_mapping_np[:num_reqs]
            num_logits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
            is_decode_indices_np = np.where(num_logits_np > 0)[0]
            self._apply_self_conditioning(
                slots_np[is_decode_indices_np],
                is_decode_indices_np,
                input_batch.query_start_loc_np,
                inputs_embeds,
                states.self_conditioning_embeds,
                num_logits_np=num_logits_np,
            )

        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # CUDA graph capture path — return a slice of the SAME persistent
        # inputs_embeds buffer that `prepare_inputs` writes to at runtime,
        # so the captured graph and runtime point to identical addresses.
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    def postprocess_state(
        self, idx_mapping, num_sampled, num_computed_tokens=None
    ) -> None:
        return None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Per-request causal mode: encoder (commit) = causal,
        # denoise = bidirectional. Pass GPU tensor so the attention
        # backend can handle mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        # Invariant: the sampler flips is_encoder_phase to False only after a
        # request's FINAL prompt chunk, so a prompt spanning multiple chunks
        # (longer than the token budget) stays causal for every chunk.
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: torch.Tensor = self._causal_buf[:num_reqs]

        req_doc_ranges = None
        if (
            self._mm_prefix_mode != "off"
            and self.supports_mm_inputs
            and self.encoder_cache is not None
        ):
            ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=None,
            )
            req_doc_ranges = {
                i: spans if input_batch.is_prefilling_np[i] else []
                for i, spans in ranges.items()
            }

        metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=causal,
            mm_req_doc_ranges=None,
        )
        if req_doc_ranges is not None:
            assert self._mm_prefix_range_buf is not None
            assert self._mm_prefix_range_cpu is not None
            assert self._mm_prefix_range_np is not None
            if self._mm_prefix_range_copy_pending:
                assert self._mm_prefix_range_copy_event is not None
                self._mm_prefix_range_copy_event.synchronize()
            range_tensor = self._mm_prefix_range_buf[:num_reqs]
            range_cpu = self._mm_prefix_range_cpu[:num_reqs]
            range_cpu.zero_()
            max_ranges = range_tensor.shape[1]
            for req_idx, spans in req_doc_ranges.items():
                if len(spans) > max_ranges:
                    raise ValueError(
                        f"Request has {len(spans)} multimodal ranges, "
                        f"but configured capacity is {max_ranges}"
                    )
                if spans:
                    self._mm_prefix_range_np[req_idx, : len(spans)] = spans
            range_tensor.copy_(range_cpu, non_blocking=True)
            assert self._mm_prefix_range_copy_event is not None
            self._mm_prefix_range_copy_event.record()
            self._mm_prefix_range_copy_pending = True
            for layer_metadata in metadata.values():
                if not hasattr(layer_metadata, "mm_prefix_range_tensor"):
                    raise ValueError(
                        "Experimental diffusion MM prefix requires an attention "
                        "backend with prefix support"
                    )
                layer_metadata.mm_prefix_range = req_doc_ranges
                layer_metadata.mm_prefix_range_tensor = range_tensor
        if req_doc_ranges is not None and self._mm_prefix_mode == "sliding_only":
            for name, layer_metadata in metadata.items():
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
                if layer_idx in self._full_attn_layer_idxs:
                    layer_metadata = copy(layer_metadata)
                    for field in (
                        "mm_prefix_range",
                        "mm_prefix_range_tensor",
                        "mm_prefix_query_range_tensor",
                    ):
                        if hasattr(layer_metadata, field):
                            setattr(layer_metadata, field, None)
                    metadata[name] = layer_metadata
        if (
            req_doc_ranges
            and any(req_doc_ranges.values())
            and self._mm_prefix_trace_file
        ):
            record = {
                "mode": self._mm_prefix_mode,
                "req_ids": input_batch.req_ids,
                "is_prefilling": input_batch.is_prefilling_np.tolist(),
                "causal": causal.tolist(),
                "ranges": req_doc_ranges,
                "layers_with_mm_ranges": [
                    name
                    for name, item in metadata.items()
                    if getattr(item, "mm_prefix_range_tensor", None) is not None
                ],
            }
            with open(self._mm_prefix_trace_file, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        if self._fuse_read_prefill:
            self._split_read_attention(metadata, input_batch, causal, num_reqs)
        return metadata

    def _split_read_attention(self, metadata, input_batch, causal, num_reqs):
        """Keep prompt and canvas attention tile origins within one backbone call."""
        actual_num_reqs = input_batch.num_reqs
        q = self._split_query_staging
        nlogits = self._split_logits_staging
        read = self._split_read_staging
        q.np.fill(int(input_batch.query_start_loc_np[actual_num_reqs]))
        nlogits.np.fill(0)
        read.np.fill(False)
        if input_batch.num_draft_tokens:
            nlogits.np[:actual_num_reqs] = np.diff(
                input_batch.cu_num_logits_np[: actual_num_reqs + 1]
            )
        for idx, slot in enumerate(
            input_batch.idx_mapping_np[:actual_num_reqs].tolist()
        ):
            start, end = input_batch.query_start_loc_np[idx : idx + 2]
            q.np[2 * idx : 2 * idx + 3] = (start, end - nlogits.np[idx], end)
            read.np[idx] = slot in self.diffusion_states.read_requests
        self._split_query.copy_(q.copy_to_uva(), non_blocking=True)
        split_seq_lens = self._split_seq_lens[: 2 * num_reqs]
        split_seq_lens.view(num_reqs, 2).copy_(
            input_batch.seq_lens[:num_reqs].unsqueeze(1)
        )
        self._split_seq_lens[: 2 * num_reqs : 2].sub_(nlogits.copy_to_uva()[:num_reqs])
        self._split_causal[: 2 * num_reqs : 2].copy_(causal)
        self._split_causal[1 : 2 * num_reqs : 2].copy_(
            causal & ~read.copy_to_uva()[:num_reqs]
        )
        split_query = self._split_query[: 2 * num_reqs + 1]
        split_causal = self._split_causal[: 2 * num_reqs]
        max_query_len = max(1, int(np.diff(q.np[: 2 * num_reqs + 1]).max()))
        seen = set()
        for layer_metadata in metadata.values():
            if id(layer_metadata) in seen:
                continue
            seen.add(id(layer_metadata))
            if not hasattr(layer_metadata, "mm_prefix_range_tensor"):
                raise ValueError("Fused reads require Triton attention metadata")
            table = layer_metadata.block_table
            key = table.data_ptr()
            if key not in self._split_block_tables:
                self._split_block_tables[key] = torch.zeros(
                    (2 * self.max_num_reqs, *table.shape[1:]),
                    dtype=table.dtype,
                    device=table.device,
                )
            split_table = self._split_block_tables[key][: 2 * num_reqs]
            split_table.view(num_reqs, 2, *table.shape[1:]).copy_(
                table[:num_reqs].unsqueeze(1)
            )
            layer_metadata.block_table = split_table
            layer_metadata.query_start_loc = split_query
            layer_metadata.seq_lens = split_seq_lens
            layer_metadata.causal = split_causal
            layer_metadata.max_query_len = max_query_len
            ranges = layer_metadata.mm_prefix_range_tensor
            if ranges is not None:
                if (
                    getattr(layer_metadata, "mm_prefix_query_range_tensor", None)
                    is not None
                ):
                    raise ValueError(
                        "Fused diffusion MM prefix currently requires Triton attention"
                    )
                assert self._split_mm_prefix_range_buf is not None
                split_ranges = self._split_mm_prefix_range_buf[: 2 * num_reqs]
                split_ranges[::2].copy_(ranges[:num_reqs])
                split_ranges[1::2].copy_(ranges[:num_reqs])
                layer_metadata.mm_prefix_range_tensor = split_ranges
                if layer_metadata.mm_prefix_range is not None:
                    layer_metadata.mm_prefix_range = {
                        split_idx: spans
                        for req_idx, spans in layer_metadata.mm_prefix_range.items()
                        for split_idx in (2 * req_idx, 2 * req_idx + 1)
                    }
        if (
            self._mm_prefix_range_copy_pending
            and self._mm_prefix_range_copy_event is not None
        ):
            self._mm_prefix_range_copy_event.record()

    num_new_sampled_tokens_per_step: int = 0


# Penalty stub for the diffusion path: the runner reads
# penalties_state.output_bin_counts, and post_update treats None as
# "no penalty bookkeeping".
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class DiffusionSampler:
    """Batched accept/renoise sampler for DiffusionGemma.

    Follows the same structure as ``vllm.v1.worker.gpu.sample.sampler.Sampler``:
    decomposed into named methods, all GPU state in pre-allocated buffers,
    no GPU→CPU syncs on the hot path.
    """

    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: DiffusionGemmaRequestStates,
        *,
        confidence_threshold: float,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
        sc_vocab_start: int = 0,
        sc_vocab_end: int | None = None,
        tp_size: int = 1,
        tp_group_name: str = "",
    ):
        self._sampler = sampler
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.compute_nans = sampler.compute_nans
        self.logits_mode = sampler.logprobs_mode in ("raw_logits", "processed_logits")
        # Self-conditioning soft embed = probs @ embed_weight * normalizer,
        # computed in the sampler (see _compiled_sample_step). ``embed_weight``
        # is the vocab-parallel shard; [sc_vocab_start, sc_vocab_end) is this
        # rank's slice of the full vocab and tp_* drive the cross-rank
        # all-reduce.
        self.embed_weight = embed_weight
        self.normalizer = normalizer
        self.sc_vocab_start = sc_vocab_start
        self.sc_vocab_end = sc_vocab_end if sc_vocab_end is not None else vocab_size
        self.tp_size = tp_size
        self.tp_group_name = tp_group_name
        self.canvas_length = (
            diffusion_config.canvas_length if diffusion_config is not None else 32
        )
        self.t_min = t_min
        self.t_max = t_max
        self.confidence_threshold = confidence_threshold
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.entropy_bound = entropy_bound

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs,
            self.canvas_length,
            dtype=torch.int32,
            device=device,
        )
        self._num_sampled = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # Per-slot stash for logprobs computed on the converging denoise step.
        # Populated after the post-sample kernel detects convergence; consumed
        # on the subsequent commit step when num_sampled=CANVAS_LEN.
        self._pending_logprobs: dict[int, LogprobsTensors] = {}
        self._read_logits_compacted = False
        self._read_projected_spans: dict[int, tuple[int, int, bool]] = {}
        if directory := os.environ.get("VLLM_TYPESAFE_REASON_COUNTER_TRACE_DIR"):
            diffusion_states.reasoning_counter_trace = ReasoningCounterTrace(
                directory,
                os.environ.get("VLLM_TYPESAFE_REASON_COUNTER_RUN_ID", ""),
                __file__,
                get_tp_group().rank_in_group if tp_size > 1 else 0,
            )

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "DiffusionGemma does not support repetition/frequency/presence "
                "penalties; ignoring them for this request."
            )
        # Purge any stale logprobs stashed under this slot by a prior request
        # that was aborted between its converging denoise and commit steps.
        self._pending_logprobs.pop(req_idx, None)
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        states = self.diffusion_states
        if states.reasoning_counter_trace is not None:
            states.reasoning_counter_trace.abort(req_idx, "slot_reused")
        states.read_requests.pop(req_idx, None)
        states.read_progress.pop(req_idx, None)
        states.read_reasoning.pop(req_idx, None)
        states.read_answer_means.pop(req_idx, None)
        states.read_self_conditioning.discard(req_idx)
        states.read_canvas_self_conditioning.discard(req_idx)
        extra = sampling_params.extra_args or {}
        if extra.get("diffusion_read_only"):
            states.read_progress[req_idx] = (
                extra.get("diffusion_read_steps", 1),
                bool(extra.get("diffusion_read_self_conditioning", True)),
            )
            if extra.get("diffusion_answer_pass_aggregation") == "mean":
                states.read_answer_means[req_idx] = _ReadAnswerMean(
                    steps=extra.get("diffusion_read_steps", 1)
                )
            if (
                extra.get("diffusion_read_self_conditioning_scope", "answer")
                == "canvas"
            ):
                states.read_canvas_self_conditioning.add(req_idx)
            positions = tuple(extra["diffusion_answer_positions"])
            answer_only_logprobs = bool(
                extra.get("diffusion_answer_only_logprobs", False)
            )
            sparse_answer_logprobs = bool(
                answer_only_logprobs and not sampling_params.detokenize
            )
            states.read_requests[req_idx] = (
                async_tensor_h2d(
                    extra["diffusion_seed_canvas"],
                    dtype=torch.int64,
                    device=states.device,
                ),
                async_tensor_h2d(
                    positions,
                    dtype=torch.int64,
                    device=states.device,
                ),
                positions,
                sparse_answer_logprobs,
                answer_only_logprobs,
            )
            if extra.get("diffusion_reasoning_positions"):
                reasoning_seed = extra.get("diffusion_reasoning_seed")
                if reasoning_seed is None:
                    seed_digest = hashlib.sha256(
                        json.dumps(extra["diffusion_seed_canvas"]).encode()
                    ).digest()
                    reasoning_seed = int.from_bytes(seed_digest[:8], "little")
                generator = torch.Generator(device=states.device)
                generator.manual_seed(reasoning_seed)
                forbidden_token_ids = async_tensor_h2d(
                    extra["diffusion_reasoning_forbidden_token_ids"],
                    dtype=torch.int64,
                    device=states.device,
                )
                allowed = torch.ones(
                    self.vocab_size, dtype=torch.bool, device=states.device
                )
                allowed[forbidden_token_ids] = False
                known_positions_mask = None
                if (
                    known := extra.get("diffusion_reasoning_known_positions")
                ) is not None:
                    known_positions_mask = torch.zeros(
                        len(extra["diffusion_seed_canvas"]),
                        dtype=torch.bool,
                        device=states.device,
                    )
                    known_positions_mask[
                        async_tensor_h2d(known, dtype=torch.int64, device=states.device)
                    ] = True
                states.read_reasoning[req_idx] = _ReadReasoning(
                    positions=async_tensor_h2d(
                        extra["diffusion_reasoning_positions"],
                        dtype=torch.int64,
                        device=states.device,
                    ),
                    positions_cpu=tuple(extra["diffusion_reasoning_positions"]),
                    forbidden_token_ids=forbidden_token_ids,
                    allowed_token_ids=allowed.nonzero().flatten(),
                    steps=extra.get("diffusion_reasoning_steps", 8),
                    convergence_stop=extra.get(
                        "diffusion_reasoning_convergence_stop", False
                    ),
                    generator=generator,
                    answer_noise=extra.get(
                        "diffusion_reasoning_answer_noise", "candidates"
                    ),
                    token_noise=extra.get("diffusion_reasoning_token_noise", "allowed"),
                    token_proposals=extra.get(
                        "diffusion_reasoning_token_proposals", "allowed"
                    ),
                    schedule_steps=extra.get("diffusion_reasoning_schedule_steps"),
                    entropy_bound=extra.get("diffusion_reasoning_entropy_bound"),
                    round_feedback_logits=extra.get(
                        "diffusion_reasoning_round_feedback_logits", False
                    ),
                    preserve_feedback=extra.get(
                        "diffusion_reasoning_preserve_feedback", False
                    ),
                    answer_feedback=extra.get(
                        "diffusion_reasoning_answer_feedback", False
                    ),
                    unmasked_feedback=extra.get(
                        "diffusion_reasoning_unmasked_feedback", False
                    ),
                    known_positions_mask=known_positions_mask,
                    delayed_suffix_positions=(
                        async_tensor_h2d(
                            delayed,
                            dtype=torch.int64,
                            device=states.device,
                        )
                        if (
                            delayed := extra.get(
                                "diffusion_reasoning_delayed_suffix_positions"
                            )
                        )
                        is not None
                        else None
                    ),
                    delayed_suffix_positions_cpu=tuple(delayed or ()),
                )
                if states.reasoning_counter_trace is not None:
                    states.reasoning_counter_trace.start(
                        req_idx,
                        self.req_states.index_to_req_id[req_idx],
                        states.read_reasoning[req_idx].steps,
                        states.read_progress[req_idx][0],
                        states.read_reasoning[req_idx].convergence_stop,
                    )
            if os.environ.get("VLLM_DIFFUSION_FUSE_READ_PREFILL") == "1":
                seed = states.read_requests[req_idx][0]
                self.req_states.draft_tokens[req_idx, : len(seed)].copy_(seed)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def get_logprobs_dims(
        self, idx_mapping_np: np.ndarray, include_token_ids: bool = True
    ) -> tuple[int, int] | None:
        return self._sampler.get_logprobs_dims(idx_mapping_np, include_token_ids)

    def prepare_sample_hidden_states(
        self, hidden_states: torch.Tensor, input_batch: InputBatch
    ) -> torch.Tensor:
        """Project only the vocabulary rows needed by each read's current phase."""
        self._read_logits_compacted = False
        self._read_projected_spans = {}
        if input_batch.num_draft_tokens == 0:
            return hidden_states[:0]

        num_logits = np.diff(input_batch.cu_num_logits_np[: input_batch.num_reqs + 1])
        decode_indices = np.flatnonzero(num_logits)
        slots = input_batch.idx_mapping_np[decode_indices]
        if not all(
            int(slot) in self.diffusion_states.read_requests
            and self.diffusion_states.read_requests[int(slot)][3]
            and self.diffusion_states.read_progress[int(slot)][0] == 1
            and int(slot) not in self.diffusion_states.read_answer_means
            and not self._is_reasoning(int(slot))
            for slot in slots
        ):
            return self._prepare_reasoning_rows(
                hidden_states, input_batch, num_logits, decode_indices, slots
            )

        answer_rows: list[int] = []
        for request_index, slot in zip(decode_indices.tolist(), slots.tolist()):
            start = int(input_batch.cu_num_logits_np[request_index])
            valid_len = int(num_logits[request_index])
            positions = self.diffusion_states.read_requests[slot][2]
            answer_rows.extend(start + pos for pos in positions if pos < valid_len)
        if not answer_rows:
            return hidden_states

        self._read_logits_compacted = True
        indices = async_tensor_h2d(
            answer_rows, dtype=torch.int64, device=hidden_states.device
        )
        return hidden_states[indices]

    def _prepare_reasoning_rows(
        self, hidden_states, input_batch, num_logits, decode_indices, slots
    ) -> torch.Tensor:
        states = self.diffusion_states
        positions_by_slot: dict[int, tuple[int, ...]] = {}
        for slot in slots.tolist():
            reasoning = states.read_reasoning.get(slot)
            if reasoning is None:
                continue
            remaining, use_sc = states.read_progress[slot]
            reasoning_active = self._is_reasoning(slot)
            if (
                use_sc
                and slot in states.read_canvas_self_conditioning
                and (
                    reasoning_active
                    or (remaining > 1 and slot not in states.read_answer_means)
                )
            ):
                continue
            if (
                not reasoning_active
                and remaining == 1
                and not states.read_requests[slot][4]
            ):
                continue
            positions = states.read_requests[slot][2]
            if reasoning_active:
                positions = reasoning.positions_cpu + positions
            positions_by_slot[slot] = positions
        if not positions_by_slot:
            return hidden_states

        rows: list[int] = []
        for request_index, slot in zip(decode_indices.tolist(), slots.tolist()):
            start = int(input_batch.cu_num_logits_np[request_index])
            valid_len = int(num_logits[request_index])
            selected_positions = positions_by_slot.get(slot)
            offset = len(rows)
            rows.extend(
                start + position
                for position in (
                    range(valid_len)
                    if selected_positions is None
                    else selected_positions
                )
                if position < valid_len
            )
            self._read_projected_spans[slot] = (
                offset,
                len(rows),
                selected_positions is not None,
            )
        indices = async_tensor_h2d(rows, dtype=torch.int64, device=hidden_states.device)
        return hidden_states[indices]

    @property
    def penalties_state(self):
        # Diffusion applies no penalties. The runner reads
        # penalties_state.output_bin_counts, so expose a stub holding None;
        # post_update treats None bin counts as "no penalty bookkeeping".
        return _NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Transition requests whose prompt completes this step to denoising.

        Initializes their canvas, seeds draft tokens, and flips
        is_encoder_phase to False. Mid-chunk requests (prompt longer than the
        token budget) are left untouched so is_encoder_phase stays True and
        prepare_attn keeps causal attention for their remaining chunks.
        """
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        # Move the slot indices across once, up front: indexing a device
        # tensor with a numpy array copies them over synchronously each time.
        ps_gpu = async_tensor_h2d(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        ordinary_slots = (
            [slot for slot in ps.tolist() if slot not in states.read_requests]
            if states._fast_read_admission
            else ps
        )
        if len(ordinary_slots) == len(ps):
            states.init_canvas(ps_gpu)
        elif len(ordinary_slots):
            states.init_canvas(async_tensor_h2d(ordinary_slots, device=states.device))
        for slot in ps.tolist():
            if slot in states.read_requests:
                seed = states.read_requests[slot][0]
                states.canvas[slot, : seed.shape[0]] = seed
        self.req_states.draft_tokens[ps_gpu, : self.canvas_length] = states.canvas[
            ps_gpu
        ]
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _is_reasoning(self, slot: int) -> bool:
        reasoning = self.diffusion_states.read_reasoning.get(slot)
        return (
            reasoning is not None
            and not reasoning.terminal
            and reasoning.step < reasoning.steps
        )

    @staticmethod
    def _reasoning_convergence(reasoning, scaled, vocab_proposals):
        constrained = scaled
        if vocab_proposals:
            constrained = scaled.clone()
            constrained[:, reasoning.forbidden_token_ids] = -float("inf")
        prediction = constrained.argmax(dim=-1)
        probabilities = constrained.softmax(dim=-1)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1).mean()
        previous = reasoning.previous_argmax
        stable = (
            (prediction == previous).all()
            if previous is not None
            else torch.zeros((), dtype=torch.bool, device=scaled.device)
        )
        stop = (
            reasoning.step + 1 < reasoning.steps
            and previous is not None
            and bool((stable & torch.isfinite(entropy) & (entropy < 0.005)).item())
        )
        reasoning.previous_argmax = prediction
        return stop, prediction, (stable, entropy)

    def _read_canvas_base(self, slot: int, canvas_len: int) -> torch.Tensor:
        states = self.diffusion_states
        seed = states.read_requests[slot][0]
        reasoning = states.read_reasoning.get(slot)
        source = (
            states.canvas[slot]
            if reasoning is not None and reasoning.step > 0
            else seed
        )
        return source[:canvas_len].clone()

    def _read_soft_embeddings(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.float().softmax(dim=-1)
        local_probabilities = probabilities[
            :, self.sc_vocab_start : self.sc_vocab_end
        ].to(self.embed_weight.dtype)
        soft = (
            local_probabilities
            @ self.embed_weight[: self.sc_vocab_end - self.sc_vocab_start]
        )
        if self.tp_size > 1:
            soft = torch.ops.vllm.all_reduce(soft, group_name=self.tp_group_name)
        return soft * self.normalizer

    def _read_canvas_feedback(self, slot: int, logits: torch.Tensor) -> None:
        # Bound each fp32 vocabulary tensor to 64 MiB for long canvases.
        rows_per_tile = max(1, min(64, (64 << 20) // (logits.shape[-1] * 4)))
        for start in range(0, len(logits), rows_per_tile):
            end = min(start + rows_per_tile, len(logits))
            soft = self._read_soft_embeddings(logits[start:end])
            self.diffusion_states.self_conditioning_embeds[slot, start:end].copy_(soft)

    def _read_candidates(self, slot: int) -> torch.Tensor:
        n = int(self.logprob_token_ids_state.num_token_ids.np[slot])
        return self.logprob_token_ids_state.token_ids.gpu[slot, :n].long()

    def _stage_read_canvas(self, slot: int, canvas: torch.Tensor) -> None:
        """Retain predictions while renewing reason-mode answer inputs."""
        states = self.diffusion_states
        states.canvas[slot, : len(canvas)] = canvas
        draft = self.req_states.draft_tokens[slot, : len(canvas)]
        draft.copy_(canvas)
        if reasoning := states.read_reasoning.get(slot):
            positions = states.read_requests[slot][1]
            if reasoning.answer_noise == "vocab":
                draft[positions] = torch.randint(
                    self.vocab_size,
                    positions.shape,
                    device=canvas.device,
                    generator=reasoning.generator,
                )
                return
            candidates = self._read_candidates(slot)
            noise = torch.randint(
                len(candidates),
                positions.shape,
                device=canvas.device,
                generator=reasoning.generator,
            )
            draft[positions] = candidates[noise]

    def _advance_reasoning(
        self, slot: int, logits: torch.Tensor, *, projected: bool = False
    ) -> None:
        """Denoise reasoning and reread answers without emitting either."""
        states = self.diffusion_states
        reasoning = states.read_reasoning[slot]
        assert self._is_reasoning(slot)
        canvas_len = len(states.read_requests[slot][0])
        positions = reasoning.positions
        answer_positions = states.read_requests[slot][1]
        expected_rows = (
            len(positions) + len(answer_positions) if projected else canvas_len
        )
        assert len(logits) == expected_rows, (
            "Reasoning requires logits for every free canvas position"
        )
        canvas = self._read_canvas_base(slot, canvas_len)
        reason_logits = logits[: len(positions)] if projected else logits[positions]
        answer_logits = (
            logits[len(positions) :] if projected else logits[answer_positions]
        )
        remaining = reasoning.steps - reasoning.step
        schedule_steps = (
            reasoning.steps
            if reasoning.schedule_steps is None
            else reasoning.schedule_steps
        )
        temperature = self.t_min + (self.t_max - self.t_min) * (
            (schedule_steps - reasoning.step) / schedule_steps
        )
        scaled = reason_logits.float() / max(temperature, 1e-10)
        vocab_proposals = remaining > 1 and reasoning.token_proposals == "vocab"
        if not vocab_proposals:
            scaled[:, reasoning.forbidden_token_ids] = -float("inf")
        selected = scaled.argmax(dim=-1)
        prediction = selected
        early_stop = False
        statistics = None
        if reasoning.convergence_stop:
            early_stop, prediction, statistics = self._reasoning_convergence(
                reasoning, scaled, vocab_proposals
            )
        if remaining > 1:
            uniform = torch.rand(
                scaled.shape,
                dtype=scaled.dtype,
                device=scaled.device,
                generator=reasoning.generator,
            ).clamp_(min=1e-20)
            gumbel = -torch.log(-torch.log(uniform))
            sampled = (scaled + gumbel * (temperature > 0)).argmax(dim=-1)
            log_probs = scaled.log_softmax(dim=-1)
            probabilities = log_probs.exp()
            entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
            sorted_entropy, indices = entropy.sort()
            accepted = torch.zeros_like(indices, dtype=torch.bool)
            entropy_bound = (
                self.entropy_bound
                if reasoning.entropy_bound is None
                else reasoning.entropy_bound
            )
            accepted[indices] = (
                sorted_entropy.cumsum(0) - sorted_entropy.cummax(0).values
            ) <= entropy_bound
            noise = torch.randint(
                self.vocab_size
                if reasoning.token_noise == "vocab"
                else len(reasoning.allowed_token_ids),
                selected.shape,
                device=scaled.device,
                generator=reasoning.generator,
            )
            noise_tokens = (
                noise
                if reasoning.token_noise == "vocab"
                else reasoning.allowed_token_ids[noise]
            )
            selected = torch.where(accepted, sampled, noise_tokens)
        if early_stop:
            selected = prediction
        terminal = remaining == 1 or early_stop
        canvas[positions] = selected
        candidates = self._read_candidates(slot)
        canvas[answer_positions] = candidates[
            answer_logits[:, candidates].argmax(dim=-1)
        ]
        self._stage_read_canvas(slot, canvas)
        reasoning.step += 1
        reasoning.terminal = terminal
        if states.reasoning_counter_trace is not None:
            states.reasoning_counter_trace.reasoning(
                slot,
                reasoning.step,
                terminal,
                "stable_confident" if early_stop else "cap",
                temperature,
                statistics,
            )
        if terminal:
            reasoning.previous_argmax = None

        use_self_conditioning = states.read_progress[slot][1]
        preserve_feedback = reasoning.preserve_feedback and use_self_conditioning
        if terminal and not preserve_feedback:
            states.self_conditioning_embeds[slot].zero_()
            states.read_self_conditioning.discard(slot)
        elif use_self_conditioning:
            feedback = states.self_conditioning_embeds[slot]
            feedback.zero_()
            if not terminal and slot in states.read_canvas_self_conditioning:
                assert not projected
                self._read_canvas_feedback(slot, logits)
            rows_per_tile = max(1, min(64, (64 << 20) // (logits.shape[-1] * 4)))
            for start in range(0, len(positions), rows_per_tile):
                end = min(start + rows_per_tile, len(positions))
                feedback_logits = (
                    reason_logits[start:end].float() / max(temperature, 1e-10)
                    if reasoning.unmasked_feedback
                    else scaled[start:end]
                )
                if vocab_proposals and not reasoning.unmasked_feedback:
                    feedback_logits = feedback_logits.clone()
                    feedback_logits[:, reasoning.forbidden_token_ids] = -float("inf")
                if reasoning.round_feedback_logits:
                    feedback_logits = feedback_logits.to(self.embed_weight.dtype)
                soft = self._read_soft_embeddings(feedback_logits)
                feedback[positions[start:end]] = soft.to(feedback.dtype)
            if (
                not terminal
                and reasoning.answer_feedback
                and slot not in states.read_canvas_self_conditioning
            ):
                soft = self._read_soft_embeddings(answer_logits)
                feedback[answer_positions] = soft.to(feedback.dtype)
            states.read_self_conditioning.add(slot)
        self._pending_logprobs.pop(slot, None)

    def _advance_read(
        self,
        slot: int,
        canvas: torch.Tensor,
        logits: torch.Tensor,
        *,
        projected: bool = False,
    ) -> None:
        """Stage the next full canvas and optional soft embeddings."""
        states = self.diffusion_states
        remaining, use_self_conditioning = states.read_progress[slot]
        assert remaining > 1
        assert len(canvas) == len(states.read_requests[slot][0]), (
            "Constrained refinement requires the complete read canvas"
        )
        positions = states.read_requests[slot][1]
        self._stage_read_canvas(slot, canvas)
        states.read_progress[slot] = (remaining - 1, use_self_conditioning)
        if slot in states.read_answer_means:
            states.self_conditioning_embeds[slot].zero_()
            states.read_self_conditioning.discard(slot)
        elif use_self_conditioning:
            if slot in states.read_canvas_self_conditioning:
                assert not projected
                states.self_conditioning_embeds[slot].zero_()
                self._read_canvas_feedback(slot, logits[: len(canvas)])
            else:
                soft = self._read_soft_embeddings(
                    logits if projected else logits[positions]
                )
                states.self_conditioning_embeds[slot].zero_()
                states.self_conditioning_embeds[slot, positions] = soft.to(
                    states.self_conditioning_embeds.dtype
                )
            states.read_self_conditioning.add(slot)
        self._pending_logprobs.pop(slot, None)

    def _read_canvas(
        self,
        logits: torch.Tensor,
        slot: int,
        num_logprobs: int,
        max_token_ids: int,
        canvas_len: int,
        *,
        projected: bool = False,
    ) -> torch.Tensor:
        """Read answer slots without sampling or changing the fixed scaffold."""
        seed, positions, _, sparse, answer_only = self.diffusion_states.read_requests[
            slot
        ]
        canvas = self._read_canvas_base(slot, canvas_len)
        if canvas_len < seed.shape[0]:
            positions = positions[positions < canvas_len]
        candidates = self._read_candidates(slot)
        answer_logits = logits if projected else logits[positions]
        if slot in self.diffusion_states.read_answer_means:
            return self._mean_read_canvas(
                slot,
                canvas,
                positions,
                candidates,
                answer_logits,
                num_logprobs,
                max_token_ids,
                sparse,
            )
        selected = candidates[answer_logits[:, candidates].argmax(dim=-1)]
        canvas[positions] = selected
        if self.diffusion_states.read_progress[slot][0] > 1:
            return canvas
        if answer_only:
            answer_logprobs = self._compute_logprobs(
                answer_logits,
                selected,
                slot,
                num_logprobs,
                max_token_ids,
                logits_mode=False,
            )
            self._pending_logprobs[slot] = (
                answer_logprobs
                if sparse
                else self._expand_read_logprobs(canvas, positions, answer_logprobs)
            )
        else:
            assert not projected
            self._pending_logprobs[slot] = self._compute_logprobs(
                logits,
                canvas,
                slot,
                num_logprobs,
                max_token_ids,
                logits_mode=False,
            )
        return canvas

    def _mean_read_canvas(
        self,
        slot: int,
        canvas: torch.Tensor,
        positions: torch.Tensor,
        candidates: torch.Tensor,
        answer_logits: torch.Tensor,
        num_logprobs: int,
        max_token_ids: int,
        sparse: bool,
    ) -> torch.Tensor:
        """Transport log(mean constrained probabilities), never raw model scores."""
        states = self.diffusion_states
        assert len(canvas) == len(states.read_requests[slot][0])
        assert len(positions) == len(states.read_requests[slot][2])
        aggregate = states.read_answer_means[slot]
        log_probabilities = answer_logits[:, candidates].float().log_softmax(-1)
        log_sum = (
            log_probabilities
            if aggregate.log_probability_sum is None
            else torch.logaddexp(aggregate.log_probability_sum, log_probabilities)
        )
        aggregate.log_probability_sum = log_sum
        aggregate.count += 1
        assert aggregate.count <= aggregate.steps
        if states.read_progress[slot][0] > 1:
            canvas[positions] = candidates[log_probabilities.argmax(-1)]
            return canvas
        assert aggregate.count == aggregate.steps
        pooled = log_sum - math.log(aggregate.steps)
        selected_indices = pooled.argmax(-1, keepdim=True)
        selected = candidates[selected_indices.squeeze(-1)]
        canvas[positions] = selected
        width = 1 + max(num_logprobs, max_token_ids)
        token_ids = selected.new_zeros((len(positions), width))
        scores = pooled.new_full((len(positions), width), -float("inf"))
        token_ids[:, 0] = selected
        token_ids[:, 1 : 1 + len(candidates)] = candidates
        scores[:, 0] = pooled.gather(1, selected_indices).squeeze(-1)
        scores[:, 1 : 1 + len(candidates)] = pooled
        ranks = (pooled >= scores[:, :1]).sum(-1).to(torch.int64)
        answer_scores = LogprobsTensors(token_ids, scores, ranks)
        self._pending_logprobs[slot] = (
            answer_scores
            if sparse
            else self._expand_read_logprobs(canvas, positions, answer_scores)
        )
        states.read_answer_means.pop(slot)
        return canvas

    def _compute_compact_read_logprobs(
        self,
        logits: torch.Tensor,
        answer_slots: list[int],
        num_logprobs: int,
        max_token_ids: int,
    ) -> tuple[torch.Tensor, LogprobsTensors]:
        """Select and score all read answers in one batched launch."""
        assert answer_slots and max_token_ids > 0
        slots = async_tensor_h2d(answer_slots, dtype=torch.int64, device=logits.device)
        candidates = self.logprob_token_ids_state.token_ids.gpu[
            slots, :max_token_ids
        ].long()
        candidate_counts = self.logprob_token_ids_state.num_token_ids.gpu[slots]
        valid = (
            torch.arange(max_token_ids, device=logits.device)[None]
            < (candidate_counts[:, None])
        )
        candidate_logits = logits.gather(1, candidates).masked_fill_(
            ~valid, -float("inf")
        )
        selected = candidates.gather(
            1, candidate_logits.argmax(dim=-1, keepdim=True)
        ).squeeze(1)
        logprobs = compute_topk_scores(
            logits,
            num_logprobs,
            selected,
            logprob_token_ids_state=self.logprob_token_ids_state,
            expanded_idx_mapping=slots.to(torch.int32),
            max_per_req_token_ids=max_token_ids,
            logits_mode=False,
        )
        return selected, logprobs

    @staticmethod
    def _expand_read_logprobs(
        canvas: torch.Tensor,
        positions: torch.Tensor,
        answer_logprobs: LogprobsTensors,
    ) -> LogprobsTensors:
        """Expand answer scores with neutral placeholders for fixed tokens."""
        width = answer_logprobs.logprobs.shape[1]
        token_ids = canvas[:, None].expand(-1, width).clone()
        logprobs = torch.zeros(
            (canvas.shape[0], width), dtype=torch.float32, device=canvas.device
        )
        ranks = torch.ones(canvas.shape[0], dtype=torch.int64, device=canvas.device)
        token_ids[positions] = answer_logprobs.logprob_token_ids
        logprobs[positions] = answer_logprobs.logprobs
        ranks[positions] = answer_logprobs.selected_token_ranks
        return LogprobsTensors(token_ids, logprobs, ranks)

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        slot: int,
        num_logprobs: int,
        max_token_ids: int,
        *,
        logits_mode: bool,
    ) -> LogprobsTensors:
        return compute_topk_scores(
            logits,
            num_logprobs,
            sampled,
            logprob_token_ids_state=self.logprob_token_ids_state,
            expanded_idx_mapping=(
                torch.full(
                    (logits.shape[0],), slot, dtype=torch.int32, device=logits.device
                )
                if max_token_ids
                else None
            ),
            max_per_req_token_ids=max_token_ids,
            logits_mode=logits_mode,
        )

    def _collect_logprobs(
        self, slots_np: np.ndarray, emitting_slots: set[int]
    ) -> LogprobsTensors | None:
        parts: list[LogprobsTensors] = []
        cu_gen: list[int] = []
        offset = 0
        for slot in slots_np.tolist():
            cu_gen.append(offset)
            if slot in emitting_slots and slot in self._pending_logprobs:
                part = self._pending_logprobs.pop(slot)
                parts.append(part)
                offset += part.logprobs.shape[0]
        if not parts:
            return None
        # Requests may have converged in batches with different logprob widths.
        width = max(part.logprobs.shape[1] for part in parts)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [
                    F.pad(p.logprob_token_ids, (0, width - p.logprobs.shape[1]))
                    for p in parts
                ]
            ),
            logprobs=torch.cat(
                [
                    F.pad(
                        p.logprobs,
                        (0, width - p.logprobs.shape[1]),
                        value=-float("inf"),
                    )
                    for p in parts
                ]
            ),
            selected_token_ranks=torch.cat([p.selected_token_ranks for p in parts]),
            cu_num_generated_tokens=cu_gen,
        )

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        device: torch.device,
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> SamplerOutput:
        """Compute num_rejected and build SamplerOutput."""
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill, init
        # canvas for any new prefills, and stage decode slot indices to GPU. ---
        states = self.diffusion_states
        read_logits_compacted = self._read_logits_compacted
        self._read_logits_compacted = False
        projected_spans = self._read_projected_spans
        self._read_projected_spans = {}
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()
        logprobs_dims = self.get_logprobs_dims(slots_np)
        num_logprobs, max_token_ids = logprobs_dims or (0, 0)
        emitting_slots: set[int] = set()
        read_logprobs_tensors: LogprobsTensors | None = None

        if states.read_requests:
            ordinary_indices = []
            ordinary_logits = []
            read_logits_offset = 0
            compact_selected = None
            compact_logprobs = None
            if read_logits_compacted:
                answer_slots = [
                    slot
                    for idx, slot in zip(
                        decode_indices_np.tolist(), decode_slots_np.tolist()
                    )
                    for position in states.read_requests[slot][2]
                    if position < per_req_nlogits_np[idx]
                ]
                compact_selected, compact_logprobs = (
                    self._compute_compact_read_logprobs(
                        logits, answer_slots, num_logprobs, max_token_ids
                    )
                )
            for idx, slot in zip(decode_indices_np.tolist(), decode_slots_np.tolist()):
                start, end = input_batch.cu_num_logits_np[idx : idx + 2]
                canvas_len = int(end - start)
                projected = False
                if slot in projected_spans:
                    start, end, projected = projected_spans[slot]
                if slot not in states.read_requests:
                    ordinary_indices.append(idx)
                    ordinary_logits.append(logits[start:end])
                    continue
                if read_logits_compacted:
                    positions = states.read_requests[slot][2]
                    num_answers = sum(pos < canvas_len for pos in positions)
                    answer_slice = slice(
                        read_logits_offset, read_logits_offset + num_answers
                    )
                    assert compact_selected is not None
                    assert compact_logprobs is not None
                    seed, positions_gpu, _, _, _ = states.read_requests[slot]
                    canvas = self._read_canvas_base(slot, canvas_len)
                    if canvas_len < seed.shape[0]:
                        positions_gpu = positions_gpu[positions_gpu < canvas_len]
                    canvas[positions_gpu] = compact_selected[answer_slice]
                    read_logits_offset += num_answers
                else:
                    read_logits = logits[start:end]
                    if self._is_reasoning(slot):
                        self._advance_reasoning(slot, read_logits, projected=projected)
                        continue
                    canvas = self._read_canvas(
                        read_logits,
                        slot,
                        num_logprobs,
                        max_token_ids,
                        canvas_len,
                        projected=projected,
                    )
                    if states.reasoning_counter_trace is not None:
                        states.reasoning_counter_trace.answer(slot)
                    if states.read_progress[slot][0] > 1:
                        self._advance_read(
                            slot, canvas, read_logits, projected=projected
                        )
                        continue
                if read_logits_compacted and states.reasoning_counter_trace is not None:
                    states.reasoning_counter_trace.answer(slot)
                sampled[idx, : len(canvas)] = canvas.to(sampled.dtype)
                num_sampled[idx].fill_(len(canvas))
                emitting_slots.add(slot)
            if read_logits_compacted:
                assert read_logits_offset == logits.shape[0]
                assert compact_logprobs is not None
                cu_num_generated_tokens = []
                offset = 0
                for idx, slot in enumerate(slots_np.tolist()):
                    cu_num_generated_tokens.append(offset)
                    if slot in emitting_slots:
                        positions = states.read_requests[slot][2]
                        canvas_len = int(per_req_nlogits_np[idx])
                        offset += sum(pos < canvas_len for pos in positions)
                read_logprobs_tensors = compact_logprobs._replace(
                    cu_num_generated_tokens=cu_num_generated_tokens
                )
            if len(ordinary_indices) != len(decode_indices_np):
                # Read rows never enter filtering, Gumbel sampling, or the
                # self-conditioning matmul in the ordinary denoise step.
                logits = torch.cat(ordinary_logits) if ordinary_logits else logits[:0]
                decode_indices_np = np.array(ordinary_indices, dtype=np.int64)
                decode_slots_np = slots_np[decode_indices_np]

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        # Real canvas length per decode request. Equals CL except when a canvas
        # was truncated near max_model_len, in which case the scheduler gave us
        # fewer than CL logits for that request.
        valid_canvas_len_np = per_req_nlogits_np[decode_indices_np]
        valid_canvas_len = async_tensor_h2d(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        # Per-request top_k/top_p, mirroring the AR sampler. Masked tokens
        # become -inf and survive the temperature scaling in the compiled
        # step, so Gumbel sampling, probs, and entropy all see the filtered
        # distribution. The committed argmax (always the top-1 token) is
        # unaffected; only the canvas exploration is constrained. Applied
        # before canvas padding so phantom positions stay uniform.
        if num_decode > 0:
            top_k, top_p = self.sampling_states.get_top_k_top_p(
                decode_slots.repeat_interleave(
                    valid_canvas_len, output_size=int(valid_canvas_len_np.sum())
                ),
                decode_slots_np,
            )
            if top_k is not None or top_p is not None:
                logits = apply_top_k_top_p(logits.float(), top_k, top_p)

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Phantom (padded) positions are zeroed → uniform logits → high
        # entropy (no premature convergence) and argmax 0 (stable); they are
        # never committed (num_sampled == real length). masked_fill (not
        # multiply) so -inf entries from top_k/top_p filtering above don't
        # turn phantom rows into NaN.
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len  # row offset per req
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)].masked_fill_(~valid.reshape(-1, 1), 0)

        # Read requests stage noisy drafts separately from their prediction canvas.
        all_slots = decode_slots

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit→False, converge→True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        # Sample over the [num_decode * CL, vocab] logits. The fp32 pipeline in
        # _compiled_sample_step keeps several live [group * CL, vocab] copies, so
        # size each tile to a fraction of free memory to bound the transient at
        # high concurrency. Tiling is bit-identical to a single pass.
        group = max(num_decode, 1)
        if num_decode > 0:
            free, _ = current_platform.mem_get_info()
            # ~10 transient fp32 copies of [group * CL, vocab] inside the step
            # (eager peaks at ~8; pad for allocator overhead and small tensors).
            bytes_per_req = CL * self.vocab_size * 4 * 10
            budget = int(free * 0.5) // max(bytes_per_req, 1)
            group = max(1, min(num_decode, budget))

        for start_req in range(0, num_decode, group):
            end_req = min(start_req + group, num_decode)
            tile = slice(start_req, end_req)
            tile_slots = decode_slots[tile]

            scaled = _compiled_sample_step(
                logits[start_req * CL : end_req * CL],
                tile_slots,
                decode_idx[tile],
                all_slots,
                valid_canvas_len[tile],
                # State
                states.canvas,
                states.argmax_canvas,
                states.step,
                states.is_encoder_phase,
                states.confident,
                states.self_conditioning_embeds,
                self.embed_weight,
                self.normalizer,
                states.accepted_canvas_history,
                states.accepted_canvas_history_len,
                # Output
                sampled,
                num_sampled,
                self.req_states.draft_tokens,
                # Config
                max_denoising_steps=float(states.max_denoising_steps),
                t_min=self.t_min,
                t_max=self.t_max,
                confidence_threshold=self.confidence_threshold,
                vocab_size=self.vocab_size,
                CL=CL,
                ST=states.stability_threshold,
                entropy_bound=self.entropy_bound,
                sc_vocab_start=self.sc_vocab_start,
                sc_vocab_end=self.sc_vocab_end,
                tp_size=self.tp_size,
                tp_group_name=self.tp_group_name,
            )

            # Logprobs for denoise steps that just converged (is_encoder_phase
            # flipped False→True), stashed per tile so `scaled` is freed each tile.
            if logprobs_dims is not None:
                converged_mask = states.is_encoder_phase[tile_slots]
                just_converged = converged_mask & ~is_committing[tile]
                if just_converged.any():
                    flat_logits = scaled.reshape(-1, scaled.shape[-1])
                    argmax_tokens = scaled.argmax(dim=-1)
                    for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                        li = local_idx.item()
                        slot = int(decode_slots_np[start_req + li])
                        # Stash only the real canvas positions (== CL unless this
                        # canvas was truncated near max_model_len); padded tail
                        # positions are never emitted.
                        k_i = int(valid_canvas_len_np[start_req + li])
                        pos = li * CL
                        self._pending_logprobs[slot] = self._compute_logprobs(
                            flat_logits[pos : pos + k_i],
                            argmax_tokens[local_idx][:k_i],
                            slot,
                            num_logprobs,
                            max_token_ids,
                            logits_mode=self.logits_mode,
                        )

        # Commit steps: is_committing was True at entry. Reassemble previously
        # stashed logprobs and attach to SamplerOutput.
        if (
            logprobs_dims is not None
            and self._pending_logprobs
            and len(decode_slots_np) > 0
        ):
            emitting_slots.update(decode_slots_np[is_committing.cpu().numpy()].tolist())
        logprobs_tensors = read_logprobs_tensors
        if logprobs_tensors is None and emitting_slots and self._pending_logprobs:
            logprobs_tensors = self._collect_logprobs(slots_np, emitting_slots)

        output = self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            device,
            logprobs_tensors=logprobs_tensors,
        )
        if states.reasoning_counter_trace is not None:
            for slot in emitting_slots:
                states.reasoning_counter_trace.finish(slot)
        return output
