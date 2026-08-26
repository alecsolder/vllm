# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma model, ModelState, and Sampler for vLLM.

Single Gemma4 backbone run in two modes (like YOCO):
- encoder mode: causal attention, writes KV cache
- decoder mode: bidirectional attention, reads encoder KV, doesn't write

Same weights, same layers. The only decoder-unique component is a
self-conditioning MLP.

Multimodal support: the model always includes a vision tower (shared with
Gemma4). Images are encoded through the vision tower and projected into
the LM embedding space via Gemma4MultimodalEmbedder.

See ``DEMO_DIFFUSION_CONSTRAINTS.md`` for the constrained-decoding design.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import suppress
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
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)

# Reuse Gemma4 components directly
from vllm.model_executor.models.gemma4 import (
    Gemma4Model,
)
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
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, async_copy_to_gpu
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty

from .interfaces import (
    SupportsMultiModal,
    SupportsPP,
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
        config = self.get_hf_config()
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            # TODO(diffusion): backward-compat for the pre-RC0.1
            # architecture name. Remove once old checkpoints are gone.
            return {"image": 0, "video": 0}

        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)


@torch.compile(dynamic=True)
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
    SupportsPP,
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

        # DiffusionGemma feeds raw (non-prenormed) input to the MoE router,
        # matching the HF decoder which does router(residual) not
        # router(pre_norm(residual)).
        text_config.router_uses_prenormed_input = False

        # DiffusionGemma's full-attention layers have NO v_proj — V is
        # computed from k_proj's output (`value_states = key_states` before
        # k_norm in `DiffusionGemmaDecoderTextAttention.forward`). This is
        # the "k_eq_v" variant in our Gemma4 backbone. The checkpoint has no
        # v_proj weights for full-attention layers; without this flag they
        # would silently load with random V projections.
        text_config.attention_k_eq_v = True

        # ---- Vision tower ----
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            quant_config = vllm_config.quant_config
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
        else:
            # TODO(diffusion): backward-compat for the pre-RC0.1
            # architecture name. Remove once old checkpoints are gone.
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

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
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


DEFAULT_STABILITY_THRESHOLD = 2
DEFAULT_CONFIDENCE_THRESHOLD = 0.005


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True)
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
    # The four below are compile-specialized; all default to the OFF value so
    # the unconstrained path prunes every constrained sampling branch. On the
    # constrained path (VLLM_DIFFUSION_CONSTRAINED_DECODE), the caller enables
    # them together.
    #
    # Converge over the through-<channel|>(101) prefix only (default False =>
    # whole-canvas convergence).
    converge_on_prefix: bool = False,
    # NaN-safe entropy (masked p=0 tokens contribute 0 instead of 0*(-inf)=NaN).
    nansafe_entropy: bool = False,
    # GLOBAL pre-mask convergence entropy. When True, the accept/convergence
    # entropy is recomputed from ``premask_logits`` for ALL positions (the
    # model's true confidence) instead of the grammar-masked logits: the grammar
    # −inf mask otherwise makes the masked logits read fake-100%-confident
    # (entropy ≈0) at every constrained position, so the eb_mask over-accepts +
    # mean_entropy under-shoots and the block commits before its value slots
    # refine (stray-comma value corruption). Token SELECTION still comes from the
    # masked logits so the emitted token stays grammar-valid.
    premask_conv_entropy: bool = False,
    # Pre-grammar-mask logits clone (real tensor on the constrained path, else
    # None) — the entropy source for ``premask_conv_entropy``.
    premask_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compiled decode step: temperature → Gumbel sample → probs/confidence →
    accept/renoise → convergence, all as vectorized PyTorch ops.

    Returns the temperature-scaled logits ``[num_decode, CL, vocab]`` so the
    caller can compute logprobs outside the compiled region."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    # Clear outputs so prefill / non-decode slots report 0 (decode slots are
    # overwritten below).
    sampled.zero_()
    num_sampled.zero_()

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

    # Converge-on-prefix: a per-slot mask of canvas positions UP TO AND INCLUDING
    # the first <channel|>(101) (argmax_tokens already reflects the force-reasoning
    # mask, so 101-detection is clean). No 101 in a slot -> p=CL-1 -> all-True ->
    # whole-canvas (unchanged). Only built when converge_on_prefix is on, so the
    # disabled path adds no graph nodes. Fixed shape (CL) -> no graph break.
    if converge_on_prefix:
        is_101 = argmax_tokens == _CHANNEL_END_ID  # [num_decode, CL]
        has_101 = is_101.any(dim=-1)
        first_101 = is_101.int().argmax(dim=-1)
        p_prefix = torch.where(has_101, first_101, torch.full_like(first_101, CL - 1))
        prefix_mask = torch.arange(CL, device=device)[None, :] <= p_prefix[:, None]

    # ---- Phase 3: Probs, self-conditioning, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    if nansafe_entropy:
        # Masked-out tokens have p=0; entr(0)=0 (lim p->0 of -p*log p), so a -inf
        # logit no longer poisons the sum with 0*(-inf)=NaN. Equals the original
        # entropy when no token is masked.
        token_entropy = torch.special.entr(probs).sum(dim=-1)  # [num_decode, CL]
    else:
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [num_decode, CL]
    if premask_conv_entropy:
        # GLOBAL pre-mask entropy (the constrained-path default). The grammar
        # −inf mask makes the masked logits read fake-100%-confident (entropy ≈0)
        # at EVERY constrained position — that fake certainty floods the eb_mask
        # accept AND deflates mean_entropy, so the block converges/commits before
        # its value slots refine (stray-comma value corruption). Recompute the
        # accept/convergence entropy from premask_logits (the model's OWN
        # confidence) for ALL positions, at the SAME temperature. Token SELECTION
        # is untouched (new_tokens/argmax_tokens come from the masked ``scaled``)
        # so the emitted token stays grammar-valid.
        scaled_pre_g = premask_logits.reshape(num_decode, CL, -1).float() / temp[
            :, None, None
        ].clamp(min=1e-10)
        lp_pre_g = scaled_pre_g.log_softmax(dim=-1)
        p_pre_g = lp_pre_g.exp()
        if nansafe_entropy:
            token_entropy = torch.special.entr(p_pre_g).sum(dim=-1)
        else:
            token_entropy = -(p_pre_g * lp_pre_g).sum(dim=-1)
    # A canvas truncated near max_model_len is zero-padded up to CL by the
    # caller; those padded rows are uniform (max entropy, argmax 0), so they
    # never trigger early convergence and are stable, and only the real
    # ``valid_canvas_len`` tokens are committed (num_sampled below).
    if converge_on_prefix:
        # Confidence over the through-101 prefix only (mean entropy of those pos).
        mean_entropy = (token_entropy * prefix_mask).sum(dim=-1) / prefix_mask.sum(
            dim=-1
        ).clamp(min=1)
    else:
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
        if converge_on_prefix:
            # Stability over the through-101 prefix only (ignore tail churn).
            mismatch = (
                mismatch
                + ((ref != history[decode_slots, h]) & prefix_mask).sum(dim=-1).int()
            )
        else:
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
    soft_embeds = torch.matmul(probs.to(embed_weight.dtype), embed_weight) * normalizer
    sc_embeds[decode_slots] = soft_embeds * sc_keep

    # Overwrite canvas with argmax for newly converged denoise requests
    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return scaled


# ---------------------------------------------------------------------------
# Grammar-constrained decoding for a parallel diffusion canvas.
#
# The implementation maintains a grammar-valid committed prefix, masks the
# current frontier, and validates a converged canvas before allowing it to
# commit. It is opt-in and favors correctness over throughput.
# ---------------------------------------------------------------------------

_CHANNEL_END_ID = 101  # ``<channel|>`` — end-of-reasoning boundary token.
_TURN_END_ID = 106  # ``<turn|>`` — turn/NL terminator.
_TOOL_CALL_OPEN_ID = 48  # ``<|tool_call>`` — tool-call open token.
_TOOL_CALL_CLOSE_ID = 49  # ``<tool_call|>`` — per-call terminator (repair span end).
# ``<|"|>``: Gemma4 string quote, legal only at a string boundary.
_QUOTE_ID = 52
_CONSTRAINT_STOP_IDS = (1, 106, 50)  # <eos> / <turn|> / <|tool_response>
# Re-noise fallback bound: max times the post-step veto re-noises the first
# grammar-bad token of ONE in-flight call before falling back to the terminal
# (commit-through-last-complete-call) branch. Bounds the re-noise loop so a
# token the model deterministically re-emits can't livelock. Per-call (reset at
# each block-per-call boundary). Always on (constrained path only).
_CONSTRAINT_RENOISE_CAP = 8


def _constrained_decode_enabled() -> bool:
    """Whether the DiffusionGemma constrained-decoding path is active.

    Single master flag (``VLLM_DIFFUSION_CONSTRAINED_DECODE``, default off).
    ON => the ordered-frontier constrained tool-call config: phase-1 reasoning
    mask + truncate-after-<channel|> + converge-on-prefix + NaN-safe entropy +
    grammar verification and the ordered-frontier decoder (with atomic
    matcher primitives). OFF => the original unconstrained sampling behavior."""
    import os

    return os.environ.get("VLLM_DIFFUSION_CONSTRAINED_DECODE", "0") == "1"


def _bitmask_popcount(u: np.ndarray) -> int:
    """Allowed-token count of a uint32 next-token bitmask view (popcount)."""
    return int(np.unpackbits(u.view(np.uint8)).sum())


def _bitmask_allows(u: np.ndarray, tok: int) -> bool:
    """Whether ``tok`` is allowed by a uint32 next-token bitmask view."""
    w, bit = int(tok) // 32, int(tok) % 32
    return bool(0 <= w < u.shape[0] and ((int(u[w]) >> bit) & 1))


_REASONING_EOS_PAD_ID = 1  # canvas pad / eos id; allowed when allow-eos is on.


# Tokens ALLOWED during phase-1 reasoning: <|channel|>(100, open),
# <channel|>(101, close) and the string quote <|"|>(52). Every other
# special/control id is masked off until 101 is committed (the forbidden set is
# built from the tokenizer, minus these). The quote is allowed because base
# genuinely uses it inside closed reasoning (recapping call syntax / code
# blocks); it is always-safe — a lone quote can't form a parseable call, and the
# call tokens (<|tool_call>48 / <tool_call|>49 / <|tool_response>50) and the
# terminators (<eos>1 / <turn|>106) stay forbidden in reasoning.
_REASONING_ALLOWED_SPECIAL_IDS = frozenset({100, 101, _QUOTE_ID})


def _prompt_reasoning_preclosed(prompt_token_ids: list[int]) -> bool:
    """Whether the prompt ends with a closed reasoning channel — i.e. its
    TRAILING token is <channel|>(101).

    True for an enable_thinking-off prompt (the chat template ends the generation
    prompt with the pre-closed ``<|channel>thought\\n<channel|>`` scaffold, so
    generation begins already OUTSIDE reasoning); False for thinking-on (the
    gen-prompt ends ``<|turn>model\\n``). Keying on the TRAILING token (not "101
    anywhere in the prompt") avoids tripping on a prior turn's closed reasoning
    channel mid-prompt in a multi-turn thinking-on request."""
    return bool(prompt_token_ids) and prompt_token_ids[-1] == _CHANNEL_END_ID


class _ConstraintChecker:
    """llguidance-backed checker for the ordered-frontier decoder.

    The live matcher represents exactly the committed prefix; speculative canvas
    validation happens on copies.
    """

    def __init__(
        self,
        ll_tokenizer: Any,
        *,
        grammar: str | None = None,
        json_schema: Any | None = None,
    ) -> None:
        self._ll_tok = ll_tokenizer
        self._vocab = int(ll_tokenizer.vocab_size)
        self._committed: list[int] = []
        # Compile the grammar ONCE; rebuilds (truncate/reset/veto) only
        # re-instantiate a matcher from the cached compiled grammar — Lark
        # compilation is expensive and rebuild-on-shrink is hot.
        self._g = self._compile_grammar(grammar, json_schema)
        self._matcher: Any = self._new_matcher()
        # Lazily built pristine matcher used when a new call must be validated
        # independently of a carried, incomplete call.
        self._start_snapshot: Any = None
        # Per-block working matcher: a disposable deep copy of the committed
        # matcher, made once at each block start and advanced INCREMENTALLY per pass
        # (drift-resync) to feed the frontier mask — so masking stays O(delta)/pass
        # instead of a per-pass deep_copy+full-consume. INVALIDATED (set None) on any
        # committed-matcher mutation (advance_valid_prefix / reset); re-cloned lazily
        # by ``speculate``. ``_working_committed`` = the WITHIN-block tokens consumed
        # into it (for the drift-resync LCP). The committed matcher is never
        # speculatively advanced -> the carry can't be lost.
        self._working: Any = None
        self._working_committed: list[int] = []

    @staticmethod
    def _compile_grammar(grammar: str | None, json_schema: Any | None) -> Any:
        import llguidance

        if grammar is not None:
            return llguidance.LLMatcher.grammar_from_lark(grammar)
        import json as _json

        schema = json_schema
        if not isinstance(schema, str):
            schema = _json.dumps(schema)
        return llguidance.LLMatcher.grammar_from_json_schema(schema)

    def _new_matcher(self) -> Any:
        import llguidance

        # Raise parser limits for deep grammars and long bodies.
        limits_cls = getattr(llguidance, "LLParserLimits", None)
        if limits_cls is not None:
            limits = limits_cls(max_items_in_row=20000, step_max_items=600000)
            m = llguidance.LLMatcher(self._ll_tok, self._g, log_level=1, limits=limits)
        else:
            m = llguidance.LLMatcher(self._ll_tok, self._g, log_level=1)
        if m.is_error():
            raise RuntimeError(f"Constraint matcher build error: {m.get_error()}")
        return m

    def reset(self) -> None:
        """Drop committed state and rebuild from the cached grammar."""
        self._matcher = self._new_matcher()
        self._committed = []
        self._start_snapshot = None
        self._working = None  # committed matcher changed -> drop the working clone
        self._working_committed = []

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def advance_valid_prefix(self, ids: list[int]) -> int:
        """ORDERED-FRONTIER advance: consume the LONGEST VALID PREFIX of ``ids``
        and KEEP it committed (do NOT roll back the partial — the OPPOSITE of
        try_extend's atomic all-or-nothing rollback). Returns the count consumed
        (the frontier delta); the matcher is left advanced to just before the
        first grammar-illegal token. This is the core of the ordered frontier:
        the frontier jumps over the whole clean prefix to the first defect in one
        pass."""
        if not ids:
            return 0
        if self._matcher.is_stopped():
            return 0
        n = int(self._matcher.try_consume_tokens(list(ids)))
        if n > 0:
            self._committed.extend(list(ids)[:n])
        # committed matcher just advanced -> the per-block working clone is stale;
        # drop it so the next block re-clones from the new committed state.
        self._working = None
        self._working_committed = []
        return n

    def speculate(self, body: list[int]) -> tuple[Any, int]:
        """Advance a per-block working matcher over the realized canvas.

        Returns ``(working, vpl)``, where ``vpl`` is the number of valid tokens
        consumed within this block. The committed matcher is not mutated.

        Perf: the working clone is made ONCE per block (lazy deep_copy of the
        committed matcher) and advanced INCREMENTALLY each pass via drift-resync
        (rollback to the longest common prefix of the prior within-block tokens and
        this pass's body, then consume the new suffix) — O(delta)/pass, not a per-pass
        deep_copy+full-consume. Correctness is identical to a fresh-copy-per-pass: the
        clone always starts from the correct committed state and the drift-resync
        reconstructs the same longest-valid-prefix frontier. Invalidated on any
        committed-matcher mutation (advance_valid_prefix / reset)."""
        if self._working is None:
            self._working = self._matcher.deep_copy()
            self._working_committed = []
        wc = self._working_committed
        # drift-resync: rollback to the LCP of the prior within-block consumed
        # tokens and this pass's body, then advance the new suffix on the clone.
        common = 0
        for a, c in zip(wc, body):
            if a == c:
                common += 1
            else:
                break
        if common < len(wc):
            self._working.rollback(len(wc) - common)
            del wc[common:]
        new = body[len(wc) :]
        if new:
            n = int(self._working.try_consume_tokens(list(new)))
            wc.extend(new[:n])
        return self._working, len(wc)

    def validate_body(self, body: list[int], hit: int | None) -> tuple[int, bool, bool]:
        """Validate one converged block against the committed matcher.

        Returns the longest valid prefix, whether the block reaches an accepting
        state, and whether a leading call opener should abandon an incomplete
        call carried from the previous block.
        """
        m = self._matcher.deep_copy()
        vpl = int(m.try_consume_tokens(list(body)))
        abandon = False
        if vpl == 0 and body and body[0] == _TOOL_CALL_OPEN_ID:
            # committed matcher rejects the new opener -> validate as a NEW call from
            # grammar START (where 48 is legal); the prior never-closed call is left
            # as-is (matches base diffusion).
            abandon = True
            if self._start_snapshot is None:
                self._start_snapshot = self._new_matcher()
            m = self._start_snapshot.deep_copy()
            vpl = int(m.try_consume_tokens(list(body)))
        accepting = False
        if vpl == len(body):
            on_call = (_TOOL_CALL_OPEN_ID in body) or (
                not abandon and _TOOL_CALL_OPEN_ID in self._committed
            )
            if on_call:
                # call path: tool_calls_seq stop [50]; accepting iff the call closed.
                accepting = (int(m.try_consume_tokens([50])) == 1) and bool(
                    m.is_accepting()
                )
            elif hit in (_TURN_END_ID, _REASONING_EOS_PAD_ID):
                # settled NL: nl_refusal stop is (106|eos); 106 the consumable proxy.
                accepting = (int(m.try_consume_tokens([_TURN_END_ID])) == 1) and bool(
                    m.is_accepting()
                )
        return vpl, accepting, abandon


@dataclass
class _ConstraintSlotState:
    """Per-request grammar state for the constrained diffusion sampler."""

    checker: _ConstraintChecker
    # The matcher engages only after the reasoning block. ``boundary_pos=-1``
    # means the answer block begins at canvas position zero.
    engaged: bool = False
    boundary_pos: int = -1
    disabled: bool = False  # set if matcher setup/verify fails -> unconstrained
    # Count of re-noise attempts on the CURRENT in-flight call (the post-step
    # veto re-noises the first grammar-bad token instead of re-settling it).
    # Bounded by ``_CONSTRAINT_RENOISE_CAP``; reset to 0 at each block-per-call
    # boundary so every call gets a fresh re-noise budget.
    renoise_count: int = 0
    # A converged block is stashed until the following commit pass advances the
    # live matcher. This carries incomplete calls across canvas boundaries.
    _pending_commit_body: list[int] | None = None
    # A new opener may explicitly abandon an incomplete call from the prior block.
    _pending_abandon: bool = False


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
        # HF reference commits `argmax_canvas` (argmax of latest step's logits),
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

        # Per-slot self-conditioning soft embedding (probs @ embed_weight) from
        # the previous denoise step. Storing the [.., hidden] soft embed instead
        # of the full [.., vocab] distribution shrinks this buffer by
        # vocab/hidden (~170x) and moves the matmul to denoise time; the result
        # is identical (SC consumes probs @ embed_weight anyway).
        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

        # Per-request matchers. An empty mapping makes constrained verification
        # a no-op for ordinary generation requests.
        self.constraint_states: dict[int, _ConstraintSlotState] = {}

        # Phase-1 reasoning-mask (VLLM_DIFFUSION_CONSTRAINED_DECODE): per-slot flag,
        # True once <channel|>(101) has been committed (reasoning closed → phase
        # 2, no mask). False (default) => phase 1, forbidden specials masked.
        # Host-side numpy (set by _truncate_after_channel, read by the mask).
        self.reasoning_closed = np.zeros(max_num_reqs, dtype=bool)
        # Forbidden special-id tensor for phase-1 masking, populated by the model
        # state from the tokenizer when the flag is on; None => strict no-op.
        # Lives here so the (worker-side) sampler can read it via ``states``.
        self.reasoning_forbidden_ids: torch.Tensor | None = None

        # Host mirror of the active constrained canvases. One batched device-to-host
        # copy feeds verification and truncation, then remains valid for the next
        # frontier pass. Reused request slots are explicitly invalidated.
        self.argmax_host = np.zeros((max_num_reqs, canvas_length), dtype=np.int64)
        self._argmax_host_valid: set[int] = set()

    def snapshot_argmax_host(
        self, decode_slots: torch.Tensor, decode_slots_np: np.ndarray
    ) -> None:
        """Refresh the host mirror after the compiled sampling step."""
        self.argmax_host[decode_slots_np] = (
            self.argmax_canvas[decode_slots].to("cpu").numpy()
        )
        self._argmax_host_valid = {int(s) for s in decode_slots_np.tolist()}

    def read_argmax_host(
        self, active_slots: list[int], device: torch.device
    ) -> np.ndarray:
        """Return cached host argmax rows, refreshing newly active slots."""
        out = np.empty(
            (len(active_slots), self.canvas_length), dtype=self.argmax_host.dtype
        )
        fresh_pos = [
            i for i, s in enumerate(active_slots) if s not in self._argmax_host_valid
        ]
        for i, s in enumerate(active_slots):
            if s in self._argmax_host_valid:
                out[i] = self.argmax_host[s]  # carried (snapshot-validated)
        if fresh_pos:
            # FRESH-read non-validated slots from the LIVE argmax_canvas and do NOT
            # cache them: only snapshot_argmax_host establishes the cross-pass carry,
            # so a slot seeded directly into argmax_canvas (first pass, newly active,
            # or the CPU sim which skips the snapshot) always re-reads live — a stale
            # host row can never be served.
            fs = [active_slots[i] for i in fresh_pos]
            ft = torch.tensor(fs, device=device, dtype=torch.int64)
            fr = self.argmax_canvas[ft].to("cpu").numpy()
            for j, i in enumerate(fresh_pos):
                out[i] = fr[j]
        return out

    def argmax_host_row(self, slot: int, device: torch.device) -> np.ndarray:
        """Return one current host argmax row, copying it on cache miss."""
        if slot in self._argmax_host_valid:
            return self.argmax_host[slot]
        st = torch.tensor([slot], device=device, dtype=torch.int64)
        return self.argmax_canvas[st].to("cpu").numpy()[0]

    def set_constraint_checker(
        self, slot_idx: int, checker: _ConstraintChecker
    ) -> None:
        """Attach a per-slot grammar checker for a structured-output request."""
        self.constraint_states[slot_idx] = _ConstraintSlotState(checker=checker)

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Initialize canvas with random tokens for the given slots."""
        n = slot_indices_np.shape[0]
        self.canvas[slot_indices_np] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.init_canvas(torch.tensor([slot_idx], device=self.device))
        self.step[slot_idx] = 0
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0
        # Fresh request starts in phase-1 reasoning (channel not yet closed).
        self.reasoning_closed[slot_idx] = False
        # A reused slot must not inherit the previous request's host mirror.
        self._argmax_host_valid.discard(slot_idx)

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0
        # Drop any per-slot matcher so a reused slot cannot inherit grammar state
        # from the previous request.
        self.constraint_states.pop(slot_idx, None)
        # Clear phase-1 reasoning state so a reused slot starts fresh.
        self.reasoning_closed[slot_idx] = False
        # Invalidate the host mirror on free.
        self._argmax_host_valid.discard(slot_idx)


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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.supports_mm_inputs = encoder_cache is not None
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        if self.supports_mm_inputs:
            from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
            from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

            assert isinstance(encoder_cache, EncoderCache)
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

        # Per-step MM data produced by get_mm_embeddings and consumed by
        # prepare_inputs.  Stored as raw (mm_embeds, is_mm_embed) so that
        # prepare_inputs can call embed_input_ids directly into the
        # persistent _inputs_embeds_buf, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds: tuple[list[torch.Tensor], torch.Tensor] | None = None

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 32
        max_denoising_steps = (
            diffusion_config.max_denoising_steps if diffusion_config else 48
        )

        text_config = self.model_config.hf_text_config
        # Diffusion sampling params come straight from generation_config.json
        # (RC0.1 flat layout); the checkpoint is the source of truth.
        self.gen_config = self.model_config.try_get_generation_config()
        self.diffusion_states = DiffusionGemmaRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            vocab_size=self.model_config.get_vocab_size(),
            max_denoising_steps=max_denoising_steps,
            device=device,
            hidden_size=text_config.hidden_size,
            stability_threshold=self.gen_config["stability_threshold"],
        )
        self._req_id_to_index: dict[str, int] = {}
        # Build the llguidance tokenizer only when the first constrained request
        # arrives.
        self._ll_tokenizer: Any = None

        # Phase-1 reasoning-mask: build the forbidden special/control ids (all
        # specials MINUS {100,101}) ONCE from the tokenizer and stash on
        # diffusion_states for the sampler to read. Only when constrained decode
        # is on (else stays None => strict no-op).
        if _constrained_decode_enabled():
            self.diffusion_states.reasoning_forbidden_ids = (
                self._build_reasoning_forbidden_ids(device)
            )

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
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
        return (
            DiffusionSampler(
                sampler=sampler,
                diffusion_config=diffusion_config,
                vocab_size=self.model_config.get_vocab_size(),
                diffusion_states=self.diffusion_states,
                t_min=gen["t_min"],
                t_max=gen["t_max"],
                entropy_bound=entropy_bound,
                confidence_threshold=gen["confidence_threshold"],
                embed_weight=self.model.model.embed_tokens.weight,
                normalizer=self.model.model.normalizer,
            ),
            None,
        )

    def apply_staged_writes(self) -> None:
        pass

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        if new_req_data.req_id.startswith("_warmup_"):
            # Warmup requests provide placeholder prompt IDs for kernel capture,
            # but no meaningful chat prompt or tool grammar to inspect.
            return
        prompt_len = len(new_req_data.prompt_token_ids)
        self.diffusion_states.prompt_len[req_index] = prompt_len
        # Guardrail: if force-reasoning is on but THIS prompt's current-turn
        # reasoning channel is already closed (enable_thinking off => the
        # generation prompt ends with a pre-closed <|channel>...<channel|>
        # scaffold), the model goes straight to the answer/tool-call and would
        # NOT generate a 101. Init reasoning_closed from the prompt so the phase-1
        # mask no-ops for this request instead of suppressing the tool call. The
        # default-False init (states.add_request above) plus the generated-101
        # flip handle the thinking-on path; the disabled path retains the original
        # sampling behavior.
        if _constrained_decode_enabled() and _prompt_reasoning_preclosed(
            new_req_data.prompt_token_ids
        ):
            self.diffusion_states.reasoning_closed[req_index] = True
            logger.warning_once(
                "DiffusionGemma CONSTRAINED_DECODE on but a prompt arrived with its "
                "reasoning channel pre-closed (enable_thinking off?) -> mask inert "
                "for that request (no tool-call suppression)."
            )
        if _constrained_decode_enabled():
            self._maybe_add_constraint_checker(req_index, new_req_data)

    def _get_ll_tokenizer(self) -> Any:
        """Lazily build and cache the llguidance tokenizer for grammar checkers.

        DIVERGENCE (worker-side tokenizer): the structured-output backend builds
        an ``LLTokenizer`` in EngineCore; the diffusion sampler runs worker-side
        with no handle to it, so we build one here from the model config. A
        production version should share a single instance.
        """
        if self._ll_tokenizer is None:
            import llguidance.hf as llg_hf

            from vllm.tokenizers import cached_tokenizer_from_config

            hf_tok = cached_tokenizer_from_config(self.model_config)
            self._ll_tokenizer = llg_hf.from_tokenizer(
                hf_tok, max(hf_tok.vocab_size, len(hf_tok))
            )
        return self._ll_tokenizer

    def _build_reasoning_forbidden_ids(self, device: torch.device) -> torch.Tensor:
        """Build the phase-1 forbidden-id tensor = ALL special/control token ids
        MINUS the allowed set, derived from the tokenizer (NOT hardcoded):
        ``all_special_ids`` plus every ``added_tokens_decoder`` entry flagged
        special. Masked to -inf on every canvas row until 101 is committed.

        Allowed = {100,101} normally; with the allow-eos variant the eos pad id
        (1) is ALSO allowed so trailing positions can pad naturally (106/50/tool
        tokens stay forbidden)."""
        from vllm.tokenizers import cached_tokenizer_from_config

        hf_tok = cached_tokenizer_from_config(self.model_config)
        special: set[int] = set()
        with suppress(Exception):
            special.update(int(i) for i in (hf_tok.all_special_ids or []))
        added = getattr(hf_tok, "added_tokens_decoder", None) or {}
        for tid, tok in added.items():
            try:
                if getattr(tok, "special", False):
                    special.add(int(tid))
            except Exception:  # noqa: BLE001
                continue
        allowed = set(_REASONING_ALLOWED_SPECIAL_IDS)
        if _constrained_decode_enabled():
            allowed.add(_REASONING_EOS_PAD_ID)  # allow eos(1) pad at trailing pos
        forbidden = sorted(special - allowed)
        logger.info(
            "DiffusionGemma CONSTRAINED_DECODE: phase-1 masks %d special ids "
            "(allow_eos=%s, allowing %s): %s",
            len(forbidden),
            _constrained_decode_enabled(),
            sorted(allowed),
            forbidden,
        )
        return torch.tensor(forbidden, dtype=torch.int64, device=device)

    def _maybe_add_constraint_checker(self, req_index: int, new_req_data: Any) -> None:
        """Build a per-slot checker for structured-output requests.

        The tool-call grammar is carried out of band on
        ``sampling_params.extra_args["diffusion_toolcall_grammar"]`` (set by
        ``Gemma4ToolParser.adjust_request``), NOT via ``structured_outputs`` — so
        the request's ``use_structured_output`` stays False and it never enters
        the structured-output machinery (which triggers a non-logit EOS on the
        diffusion path).
        """
        sp = getattr(new_req_data, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) or {}
        grammar = extra.get("diffusion_toolcall_grammar")
        json_schema = None
        if grammar is None:
            return  # No worker-side grammar: verification is a no-op.
        try:
            ll_tok = self._get_ll_tokenizer()
            checker = _ConstraintChecker(
                ll_tok, grammar=grammar, json_schema=json_schema
            )
            self.diffusion_states.set_constraint_checker(req_index, checker)
        except Exception:  # noqa: BLE001
            logger.warning_once(
                "DiffusionGemma constrained decoding: failed to set up the grammar "
                "checker; "
                "this request will generate unconstrained."
            )

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def get_mm_embeddings(self, scheduled_encoder_inputs, input_batch):
        if not self.supports_mm_inputs:
            return None

        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np,
            input_batch.num_computed_prefill_tokens_np,
        )

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
            canvas = slice(start, end)
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
            )

        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # CUDA graph capture path — return a slice of the SAME persistent
        # inputs_embeds buffer that `prepare_inputs` writes to at runtime,
        # so the captured graph and runtime point to identical addresses.
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    def postprocess_state(self, idx_mapping, num_sampled) -> None:
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
    ) -> dict[str, Any]:
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
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
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
        )

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
        diffusion_states: DiffusionGemmaRequestStates | None = None,
        *,
        confidence_threshold: float,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
    ):
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        # Self-conditioning soft embed = probs @ embed_weight * normalizer,
        # computed in the sampler (see _compiled_sample_step).
        self.embed_weight = embed_weight
        self.normalizer = normalizer
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

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "DiffusionGemma does not support repetition/frequency/presence "
                "penalties; ignoring them for this request."
            )
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        # Diffusion applies no penalties. The runner reads
        # penalties_state.output_bin_counts, so expose a stub holding None;
        # post_update treats None bin counts as "no penalty bookkeeping".
        return _NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _handle_prefill(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> SamplerOutput:
        states = self.diffusion_states
        num_reqs = input_batch.num_reqs
        CL = self.canvas_length
        slots = input_batch.idx_mapping[:num_reqs]
        states.init_canvas(slots)
        self.req_states.draft_tokens[slots, :CL] = states.canvas[slots]
        states.is_encoder_phase.index_fill_(0, slots.long(), False)
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
    # Ordered-frontier grammar-constrained decoding
    # ------------------------------------------------------------------

    def _apply_reasoning_mask(
        self,
        logits: torch.Tensor,
        states: DiffusionGemmaRequestStates,
        decode_slots_np: np.ndarray,
        CL: int,
    ) -> torch.Tensor:
        """Phase-1 reasoning-mask (VLLM_DIFFUSION_CONSTRAINED_DECODE). For every
        decode slot still in reasoning (``reasoning_closed[slot]`` False), set the
        forbidden special/control ids (all specials MINUS {100,101}) to -inf on
        ALL CL canvas rows of that slot, so a forbidden special can never become
        the argmax until <channel|>(101) is committed. In-place; runs on denoise
        AND commit steps. ``logits`` is [num_decode*CL, V] (row = di*CL + pos)."""
        forbidden = states.reasoning_forbidden_ids
        if forbidden is None or forbidden.numel() == 0:
            return logits
        for di, slot in enumerate(decode_slots_np.tolist()):
            if states.reasoning_closed[int(slot)]:
                continue  # phase 2 — reasoning already closed, no mask
            base = di * CL
            logits[base : base + CL].index_fill_(1, forbidden, float("-inf"))
        return logits

    def _get_ll_executor(self) -> Any:
        """Return a process-wide executor for batched frontier masks.

        Older llguidance versions fall back to serial mask construction.
        """
        ex = getattr(self, "_ll_executor", None)
        if ex is None and not getattr(self, "_ll_executor_unavailable", False):
            try:
                import llguidance

                ex = llguidance.LLExecutor()
                self._ll_executor = ex
            except Exception:  # noqa: BLE001
                self._ll_executor_unavailable = True
                ex = None
        return ex

    def _fill_frontier_masks_batched(self, matchers: list, vocab: int) -> Any:
        """Compute next-token frontier masks, with a serial compatibility path."""
        import llguidance.torch as llg_torch

        n = len(matchers)
        bm = llg_torch.allocate_token_bitmask(n, vocab)
        used_par = False
        try:
            ex = self._get_ll_executor()
            if ex is not None:
                llg_torch.fill_next_token_bitmask_par(
                    ex, [(m, i) for i, m in enumerate(matchers)], bm
                )
                used_par = True
        except Exception:  # noqa: BLE001
            used_par = False
        if not used_par:
            for i, m in enumerate(matchers):
                with suppress(Exception):
                    llg_torch.fill_next_token_bitmask(m, bm, i)
        return bm

    def _apply_constraint_verify(
        self,
        logits: torch.Tensor,
        states: DiffusionGemmaRequestStates,
        decode_slots_np: np.ndarray,
        CL: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Apply grammar guidance at each constrained request's frontier.

        Matcher advancement is serial, while independent next-token masks are
        batched when llguidance exposes a compatible executor.
        """
        slot_list = [int(s) for s in decode_slots_np.tolist()]
        slot_to_di = {s: i for i, s in enumerate(slot_list)}
        active = [
            s
            for s in slot_list
            if s in states.constraint_states
            and not states.constraint_states[s].disabled
        ]
        if not active:
            return logits
        # Use the shared host snapshot to avoid repeated device synchronization.
        argmax_cpu = states.read_argmax_host(active, device)
        # --- Phase 1: advance every active matcher; collect STUCK slots ----------
        grammar: list[tuple[Any, int, int, int]] = []  # (st, di, frontier, body_tok)
        for row_i, slot in enumerate(active):
            st = states.constraint_states[slot]
            di = slot_to_di[slot]
            try:
                b = self._constraint_gate(st, bool(states.reasoning_closed[slot]))
                if b is None:
                    continue  # reasoning still open -> slot stays free this pass
                plan = self._constraint_advance(
                    st, argmax_cpu[row_i], di, b, CL, device
                )
            except Exception:  # noqa: BLE001
                st.disabled = True
                logger.warning_once(
                    "DiffusionGemma constrained decoding: per-slot verification "
                    "failed; request continues unconstrained."
                )
                continue
            if plan is None:
                continue
            if plan[0] == "stop":
                self._constraint_apply_mask(
                    logits, di, plan[1], [set(_CONSTRAINT_STOP_IDS)], CL, device
                )
            else:  # "grammar" -> needs the (batched) frontier mask
                # plan[3] = the SPECULATIVE deep_copy advanced to the frontier; the
                # batched fill reads ITS mask, not the committed matcher's.
                grammar.append((st, di, plan[1], plan[2], plan[3]))
        # --- Phase 2: ONE batched frontier-mask compute for all STUCK slots ------
        if grammar:
            vocab = grammar[0][0].checker.vocab_size
            matchers = [g[4] for g in grammar]
            bm = self._fill_frontier_masks_batched(matchers, vocab)
            # --- Phase 3: per-slot classify + apply from its mask row ------------
            for i, (st, di, frontier, body_tok, _m) in enumerate(grammar):
                try:
                    self._constraint_apply_frontier_from_mask(
                        logits, st, di, frontier, body_tok, bm[i : i + 1], device
                    )
                except Exception:  # noqa: BLE001
                    st.disabled = True
        return logits

    def _constraint_gate(
        self, st: _ConstraintSlotState, reasoning_closed: bool
    ) -> int | None:
        """Engage the grammar at the start of the first post-reasoning block."""
        if not st.engaged:
            if not reasoning_closed:
                return None
            st.boundary_pos = -1
            st.engaged = True
        return st.boundary_pos

    def _constraint_apply_mask(
        self,
        logits: torch.Tensor,
        di: int,
        start: int,
        per_pos: list[set[int]],
        CL: int,
        device: torch.device,
    ) -> None:
        """Mask uncommitted positions ``[start, start+len(per_pos))`` to their
        allowed-token sets (``-inf`` on everything else) so the compiled denoise
        can only commit grammar-valid + extendable tokens. An empty set => leave
        that position unconstrained (defensive; shouldn't occur when ok)."""
        if not per_pos:
            return
        if logits.shape[-1] != int(self.vocab_size):
            return  # Defensive no-op if model and tokenizer vocabularies differ.
        vocab = logits.shape[-1]
        words = (vocab + 31) // 32
        arr = np.zeros((len(per_pos), words), dtype=np.uint32)
        for i, toks in enumerate(per_pos):
            if not toks:
                arr[i, :] = np.uint32(0xFFFFFFFF)  # allow-all (no constraint)
                continue
            for t in toks:
                if 0 <= t < vocab:
                    arr[i, t // 32] |= np.uint32(1) << np.uint32(t % 32)
        import llguidance.torch as llg_torch

        bm = torch.from_numpy(arr.view(np.int32)).to(device)
        base = di * CL + start
        llg_torch.apply_token_bitmask_inplace(logits[base : base + len(per_pos)], bm)

    def _constraint_mask_frontier_grammar(
        self,
        logits: torch.Tensor,
        st: _ConstraintSlotState,
        di: int,
        frontier: int,
        bitmask: Any,
        device: torch.device,
    ) -> None:
        """Apply a precomputed grammar mask at one canvas frontier."""
        CL = self.canvas_length
        if not (0 <= frontier < CL):
            return
        if st.checker.vocab_size != logits.shape[-1]:
            return
        import llguidance.torch as llg_torch

        row = di * CL + frontier
        llg_torch.apply_token_bitmask_inplace(logits[row : row + 1], bitmask.to(device))

    def _constraint_apply_frontier_from_mask(
        self,
        logits: torch.Tensor,
        st: _ConstraintSlotState,
        di: int,
        frontier: int,
        body_tok: int,
        bm_row: Any,
        device: torch.device,
    ) -> None:
        """Classify and mask one ordered grammar frontier.

        A uniquely required tool-call closer is forced. Open string values retain
        all ordinary content choices while reserved control tokens are masked.
        """
        u = bm_row.reshape(-1).cpu().numpy().view(np.uint32)
        allowed_count = _bitmask_popcount(u)
        model_allowed = _bitmask_allows(u, body_tok)
        # Force only a uniquely legal completion token. A frontier that also
        # admits natural language remains free to choose that branch.
        if (not model_allowed) and allowed_count == 1:
            # Scope the lone-legal −∞ force to the completion-critical closer
            # <tool_call|>(49) ONLY (the only value-safe completion token). The lone
            # token IS 49 iff 49 is allowed here (allowed_count==1 already);
            # otherwise skip -> free-converge (veto + commit-repair carry validity).
            if not _bitmask_allows(u, _TOOL_CALL_CLOSE_ID):
                return
            # (3a) STRUCTURAL defect -> exact-mask the frontier (forbid illegal,
            # force the legal structural token(s)). The sampler re-picks legal next
            # pass; entropy collapses; the eb_mask accepts it -> it settles correct.
            self._constraint_mask_frontier_grammar(
                logits, st, di, frontier, bm_row, device
            )
            return
        # (3b) VALUE-position stall. If the committed prefix ends INSIDE an open
        # freeform string (close-quote 52 legal AND >1 tokens legal), apply the
        # grammar frontier mask too: the value-region char-class grammar forbids
        # EVERY reserved special inside an open string, so this forbids pad/eos/
        # closer/channel/... from converging into an open value (the empty-value
        # TRUNC) while leaving EVERY content token AND the close-quote (52) at their
        # model logits. This masks no content choices; token 52 is grammar-legal
        # only at a string boundary, so the branch does not fire outside a value.
        if _bitmask_allows(u, _QUOTE_ID) and allowed_count > 1:
            # If the model starts a new call while an earlier string remains open,
            # allow only that opener back into the mask. Commit-time validation
            # then abandons the incomplete call and restarts from the grammar root.
            if body_tok == _TOOL_CALL_OPEN_ID:
                word, bit = _TOOL_CALL_OPEN_ID // 32, _TOOL_CALL_OPEN_ID % 32
                bm_row[0, word] = bm_row[0, word] | (1 << bit)
            self._constraint_mask_frontier_grammar(
                logits, st, di, frontier, bm_row, device
            )
        # (4) tail beyond the frontier: untouched -> free-converge.

    def _constraint_advance(
        self,
        st: _ConstraintSlotState,
        snap: np.ndarray,
        di: int,
        b: int,
        CL: int,
        device: torch.device,
    ):
        """Speculate on one canvas without mutating the committed matcher.

        Returns:
          ``None`` — no body / whole body a valid prefix / failed (st.disabled);
          ``("stop", frontier)`` — speculative prefix accepting -> EOS-pad;
          ``("grammar", frontier, body_tok, m)`` — STUCK at ``body_tok`` -> the
          frontier needs a grammar mask, filled from the SPECULATIVE copy ``m``.
        The committed matcher advances only after a block is accepted.
        """
        start_body = b + 1
        if not (0 <= start_body < CL):
            return None
        # Realized body up to (and including) the first stop token, in CL space.
        body: list[int] = []
        for k in range(start_body, CL):
            t = int(snap[k])
            body.append(t)
            # Stop at this call's closer (49), inclusive;
            # positions after it stay FREE (no cross-call frontier mask), so one
            # (tc_body)+ matcher never spans >1 call.
            if t == _TOOL_CALL_CLOSE_ID or t in _CONSTRAINT_STOP_IDS:
                break
        if not body:
            return None
        try:
            # Speculate the body on a deep_copy of the committed matcher (its state
            # == the committed canvas of the call so far). ``cl`` = within-block
            # longest valid prefix; ``m`` is left at its frontier state so the
            # batched fill reads ITS mask. The committed matcher is NOT mutated.
            m, cl = st.checker.speculate(body)
        except Exception:  # noqa: BLE001
            st.disabled = True
            return None
        # Do not force arbitrary lone-legal tokens or mask value-interior commas:
        # both can corrupt the free value region. The apply phase only forces the
        # completion-critical closer (49); commit-time validation handles the rest.
        frontier = start_body + cl
        if frontier >= CL or cl >= len(body):
            return None  # whole (bounded) body is a valid prefix -> post-step decides
        # Pad after an already accepting prefix to avoid a livelock on unused
        # canvas positions.
        if m.is_accepting():
            return ("stop", frontier)
        # The frontier is stuck at body[cl], the first illegal token. The apply
        # phase classifies and masks it using the speculative matcher's mask.
        return ("grammar", frontier, body[cl], m)

    def _constraint_poststep_veto(
        self,
        states: DiffusionGemmaRequestStates,
        decode_slots_np: np.ndarray,
        is_committing: torch.Tensor,
        valid_len_np: np.ndarray,
        CL: int,
        device: torch.device,
    ) -> None:
        """Hard reliability backstop: never let a non-extendable canvas commit.

        For each constrained slot that just CONVERGED this denoise step (will
        commit next), validate the REALIZED post-boundary body from the grammar
        start. If valid + accepting -> allow. Else, while under the re-noise cap:
        suppress the convergence (is_encoder_phase=False) and RE-NOISE the whole
        invalid tail in BOTH states.canvas AND the model-input draft_tokens with
        the native forward noise (a real input write, not an eb_mask trick), then
        reset step + stability so the model keeps denoising and can close the call
        in-distribution. Once the cap is exhausted: commit through the last
        COMPLETE call's closer and drop only the invalid in-flight tail (never
        zero a span containing a 49 -> trailing complete calls survive). Re-runs
        every step.

        SCOPE: multi-block aware. A tool call longer than one CL-canvas block
        exhausts the block as a grammar-valid-but-open prefix; the veto ALLOWs
        that commit and the commit-advance extends the per-slot COMMITTED matcher
        by the block's body so the next block continues the SAME call from the
        committed-canvas state (like base diffusion, which spans long outputs
        across denoise blocks). The veto enforces that a committed
        block, if emitted, is either a complete grammar-valid output or a valid
        open prefix continued in the next block.
        """
        slot_list = [int(s) for s in decode_slots_np.tolist()]
        cand = [
            i
            for i, s in enumerate(slot_list)
            if s in states.constraint_states
            and not states.constraint_states[s].disabled
            and states.constraint_states[s].engaged
        ]
        if not cand:
            return
        cslots = [slot_list[i] for i in cand]
        ct = torch.tensor(cslots, device=device, dtype=torch.int64)
        enc_now = states.is_encoder_phase[ct].to("cpu").numpy().astype(bool)
        is_comm = is_committing.to("cpu").numpy().astype(bool)

        for j, i in enumerate(cand):
            slot = cslots[j]
            st = states.constraint_states[slot]
            # This is the only place the committed matcher is
            # mutated. The canvas was reinited by the compiled step before this veto,
            # so we advance by the body STASHED on the converge pass (ALLOW/CONTINUE).
            # An OPEN continuation (no closer) extends the committed matcher; a CLOSED
            # call (closer 49 in the body) or a terminal commit (no stash) resets it
            # to grammar START so the next call validates fresh. The body itself
            # signals whether matcher state should continue or reset.
            if bool(is_comm[i]):
                pend = st._pending_commit_body
                ab = st._pending_abandon
                st._pending_commit_body = None
                st._pending_abandon = False
                try:
                    if pend is None or _TOOL_CALL_CLOSE_ID in pend:
                        # terminal commit, or the body CLOSED the call -> next call
                        # validates from grammar START. (Single-block calls land here
                        # too: their body holds the closer.)
                        st.checker.reset()
                    else:
                        # OPEN continuation -> extend the committed matcher by this
                        # block's committed body. Cross-block carry is automatic:
                        # the matcher state equals the committed canvas of the call.
                        if ab:
                            # SETTLED ABANDON: the model started a NEW call after a
                            # never-closed value -> drop the prior call (reset to
                            # START) before advancing the new one (which was accepted
                            # from START in validate_body).
                            st.checker.reset()
                        st.checker.advance_valid_prefix(pend)
                except Exception:  # noqa: BLE001
                    st.disabled = True
                continue
            # just-converged: encoder phase now True, wasn't committing at entry.
            if not enc_now[j]:
                continue
            b = st.boundary_pos
            vlen = int(valid_len_np[i])
            if b + 1 >= vlen:
                continue
            raw = states.argmax_host_row(slot, device)[b + 1 : vlen].tolist()
            # Bound the body at the first tool-call closer (49), inclusive. A
            # complete call ends at its closer, so validation sees
            # only THIS call (calls 2..N live in later blocks after
            # _truncate_after_toolcall). `hit` = the terminator that stopped the
            # scan (the 49 closer, or an eos/turn/tool_response stop), excluded
            # from body unless it's the closer. Plain `body` is kept everywhere
            # else; only the bounded body is presented to the matcher.
            body = []
            hit = None
            for _t in raw:
                if _t == _TOOL_CALL_CLOSE_ID:
                    body.append(_t)
                    hit = _t
                    break  # complete call -> include the closer
                if _t in _CONSTRAINT_STOP_IDS:
                    hit = _t
                    break  # a terminator (eos/turn/tool_response) -> exclude
                body.append(_t)
            # Validate this block's realized ``body`` from
            # the COMMITTED matcher (== the committed canvas of the call so far) on a
            # deep_copy — uniform for every block (fresh, continuation, NL). ``vpl``
            # is the within-block valid prefix DIRECTLY; ``validate_body`` appends the
            # path-aware synthetic terminator internally. No prior replay is needed,
            # avoiding ambiguity between a continued call and natural language
            # parsed from the grammar root.
            vpl, accepting, abandon = st.checker.validate_body(body, hit)
            # CONTINUATION trigger: the full call is a grammar-VALID prefix
            # (``vpl == len(body)`` <=> the whole block validly continues) that ran
            # to the canvas end (``hit is None`` -> no closer(49), no stop) — i.e. a
            # tool call longer than one canvas block. Distinct from an illegal-token
            # tail (vpl < len(body)) and a premature stop (hit is not None), both of
            # which keep the old re-noise / terminal behavior.
            is_block_exhaust = hit is None and len(body) > 0 and vpl == len(body)
            # Ordered-frontier convergence gate (in-distribution):
            #  - VALID + accepting (complete call) -> ALLOW the commit.
            #  - block-exhausting valid-open prefix -> ALLOW the commit and CARRY
            #    the matcher; the next block continues the SAME call (multi-block
            #    continuation, like base diffusion). NO span cap — termination is
            #    the natural max_model_len/eos bound (each committed continuation
            #    advances the output by the block's valid body, so total length is
            #    bounded; the per-block re-noise cap below still bounds retries).
            #  - illegal token AND under the re-noise cap -> SUPPRESS convergence
            #    and RE-NOISE the whole invalid tail (states.canvas + the model-
            #    input draft_tokens) with native forward noise; reset step +
            #    history len so the model keeps denoising and can close the call.
            #  - caps exhausted -> TERMINAL safety net: commit through the last
            #    COMPLETE call's closer and drop only the invalid in-flight tail
            #    (never zero a span containing a 49).
            if accepting:
                # COMPLETE call (or NL answer) -> ALLOW the commit. No continuation
                # body to carry; the commit-advance resets the matcher to START for
                # the next call (pending body None -> reset; abandon irrelevant).
                st._pending_commit_body = None
                st._pending_abandon = False
            elif is_block_exhaust:
                # MULTI-BLOCK CONTINUATION. The body is a grammar-valid but
                # non-accepting OPEN prefix that exhausted the block (the call is
                # longer than one canvas block). ALLOW the commit: leave
                # is_encoder_phase True so the block commits, do NOT re-noise, do NOT
                # eos-pad. STASH the body so the commit-advance extends the committed
                # matcher by it -> the next block continues the same call from the
                # committed-canvas state. In-
                # distribution: base diffusion spans long outputs across denoise
                # blocks — we just stop resetting the grammar each block. Termination
                # holds via the natural max_model_len/eos bound (each committed
                # continuation advances the output by the block's valid body), so a
                # never-closing call still cannot hang.
                # ``abandon`` (a settled new opener the committed matcher rejected)
                # is carried to the commit so it resets the prior call BEFORE
                # advancing the new one.
                st._pending_commit_body = body
                st._pending_abandon = abandon
            elif st.renoise_count < _CONSTRAINT_RENOISE_CAP:
                # IN-DISTRIBUTION re-noise (replaces the eos-pad). Instead of
                # eos-padding the invalid tail (forces a stop / wipes the
                # still-open call), RE-NOISE the whole invalid tail
                # [b+1+vpl, vlen) with the model's NATIVE forward noise — full-
                # vocab uniform, the identical construction as init_canvas and the
                # per-pass random_tokens (NOT the Gumbel sampling trick) — and keep
                # denoising so the model can CLOSE the call naturally (stays
                # in-distribution). Fires on EVERY invalid convergence, before OR
                # at max (we reset step to grant fresh denoise passes), bounded by
                # _CONSTRAINT_RENOISE_CAP so this repair loop is bounded.
                # ⚠ The veto runs AFTER _compiled_sample_step synced
                # canvas -> draft_tokens (and __call__ does NOT re-sync), so the
                # model reads draft_tokens — a canvas-only write is INERT. Write
                # the SAME noise into BOTH states.canvas (denoise/convergence) AND
                # self.req_states.draft_tokens (the model input). ``vpl`` is the
                # hoisted longest-valid-prefix length (computed above).
                lo = b + 1 + vpl
                if vpl >= 0 and 0 <= lo < vlen:
                    span = vlen - lo
                    noise = torch.randint(
                        0,
                        states.vocab_size,
                        (span,),
                        dtype=states.canvas.dtype,
                        device=states.canvas.device,
                    )
                    states.canvas[slot, lo:vlen] = noise
                    self.req_states.draft_tokens[slot, lo:vlen] = noise
                # Keep denoising: suppress this commit, grant fresh denoise passes
                # (reset step) and break stability (reset history len) so the slot
                # cannot immediately re-converge onto the just-re-noised tail.
                states.is_encoder_phase[slot] = False
                states.step[slot] = 0
                states.accepted_canvas_history_len[slot] = 0
                st.renoise_count += 1
            else:
                # TERMINAL SAFETY NET (the re-noise cap is exhausted and the body
                # remains invalid): finish by committing through the last complete
                # call's closer (49)
                # and stop after it; drop ONLY the still-invalid in-flight tail.
                # NEVER zero a span containing a 49 (would wipe a complete call).
                # The dropped tail re-primes via _truncate_after_toolcall /
                # regenerates in the next block. For a block-FULL valid-open call
                # (no closer, cut == vlen, no room for a trailing stop) a last-
                # resort eos is stamped at vlen-1 below so termination is guaranteed
                # by the bounded re-noise fallback, not only by max_model_len.
                closers = [k for k, _t in enumerate(raw) if _t == _TOOL_CALL_CLOSE_ID]
                if closers:
                    cut = b + 1 + closers[-1] + 1  # keep through the last 49
                    vpl = -1
                else:
                    # No complete call -> drop the invalid in-flight single call
                    # at its longest grammar-valid prefix (span has no 49). ``vpl``
                    # is the hoisted longest-valid-prefix length (computed above).
                    cut = (b + 1 + vpl) if vpl >= 0 else (b + 1)
                if 0 <= cut < vlen:
                    states.argmax_canvas[slot, cut:vlen] = _CONSTRAINT_STOP_IDS[0]
                    states.argmax_host[slot, cut:vlen] = _CONSTRAINT_STOP_IDS[0]
                elif not closers and cut >= vlen and vlen > 0:
                    # LAST-RESORT termination (block-full NO-closer case): the
                    # grammar-valid prefix fills the ENTIRE canvas (cut == vlen) so
                    # there is no room to stamp a stop AFTER it. Force the FINAL
                    # position to eos after the re-noise budget is exhausted. This
                    # minimal out-of-distribution fallback is not used on the normal
                    # continuation path.
                    # ``not closers`` guard: when a trailing closer(49) sits at
                    # vlen-1 ``cut`` is also vlen, but the call IS complete through
                    # it — never overwrite a 49 (terminal-net invariant).
                    states.argmax_canvas[slot, vlen - 1] = _CONSTRAINT_STOP_IDS[0]
                    states.argmax_host[slot, vlen - 1] = _CONSTRAINT_STOP_IDS[0]

    # ------------------------------------------------------------------
    # Two-phase reasoning/answer boundary handling
    # ------------------------------------------------------------------

    def _truncate_after_channel(
        self,
        states: DiffusionGemmaRequestStates,
        decode_slots_np: np.ndarray,
        decode_idx_np: np.ndarray,
        is_committing: torch.Tensor,
        num_sampled: torch.Tensor,
        valid_canvas_len_np: np.ndarray,
        device: torch.device,
    ) -> dict[int, int]:
        """End a reasoning block at its first ``<channel|>`` token.

        The scheduler commits through the boundary and regenerates the answer in
        a fresh block, where the tool-call grammar can engage from position zero.

        No-op when: no 101 in the committed canvas; 101 is the last real token
        (p+1==valid_len); never scans past ``valid_len``. Multiple 101 -> first.
        This shares the verifier's host canvas snapshot.

        Returns ``{slot: trunc_len}`` so a later tool-call truncation does not
        overwrite the same request's boundary decision.
        """
        slot_list = [int(s) for s in decode_slots_np.tolist()]
        idx_list = [int(i) for i in decode_idx_np.tolist()]
        is_comm = is_committing.to("cpu").numpy().astype(bool)
        trunc: dict[int, int] = {}
        override_idx: list[int] = []
        override_len: list[int] = []
        for li, slot in enumerate(slot_list):
            if not is_comm[li]:
                continue
            vlen = int(valid_canvas_len_np[li])
            if vlen <= 0:
                continue
            row = states.argmax_host_row(slot, device)[:vlen]
            hits = np.where(row == _CHANNEL_END_ID)[0]
            if hits.size == 0:
                continue  # no boundary -> commit the whole block (no-op)
            p = int(hits[0])
            # 101 is being committed this block -> reasoning is closed (phase 2).
            # Set regardless of truncation (even if 101 is the last token).
            states.reasoning_closed[slot] = True
            if p + 1 >= vlen:
                continue  # boundary at the very end -> nothing to roll back
            override_idx.append(idx_list[li])
            override_len.append(p + 1)
            trunc[slot] = p + 1
        if override_idx:
            di = torch.tensor(override_idx, device=device, dtype=torch.int64)
            vals = torch.tensor(override_len, device=device, dtype=num_sampled.dtype)
            num_sampled[di] = vals
        return trunc

    def _truncate_after_toolcall(
        self,
        states: DiffusionGemmaRequestStates,
        decode_slots_np: np.ndarray,
        decode_idx_np: np.ndarray,
        is_committing: torch.Tensor,
        num_sampled: torch.Tensor,
        valid_canvas_len_np: np.ndarray,
        device: torch.device,
        skip_slots: set[int] | None = None,
    ) -> dict[int, int]:
        """Commit at most one complete tool call per diffusion block.

        For each slot committing this step, find the
        first tool-call closer ``<tool_call|>``(49) at position ``p``; if ANOTHER
        call opener ``<|tool_call>``(48) follows it within the valid canvas, cap
        ``num_sampled`` at ``p+1`` (GPU scatter) so the scheduler commits only
        call1 (tokens ``0..p``) and rolls back the rest. The rolled-back calls
        regenerate FRESH in the next block, each validated from grammar START (the
        per-slot reset below) — so one ``(tc_body)+`` matcher never spans >1 call
        (the parallel-call value-corruption / call-split bug).

        No-op when: no closer in the committed canvas; closer at the very end
        (``p+1>=vlen``); the closer is the LAST call (no following 48) -> commit
        whole, trailing stop kept -> ``finish_reason='tool_calls'`` unchanged.
        ``skip_slots`` skips slots already truncated by the channel pass this step
        (defensive: both write ``num_sampled``; in practice they fire in different
        blocks since calls only commit once reasoning is closed). Mirrors
        ``_truncate_after_channel``'s host D2H + GPU-scatter pattern.

        Returns ``{slot: trunc_len}`` for callers that coordinate truncations.
        """
        skip = skip_slots or set()
        slot_list = [int(s) for s in decode_slots_np.tolist()]
        idx_list = [int(i) for i in decode_idx_np.tolist()]
        is_comm = is_committing.to("cpu").numpy().astype(bool)
        trunc: dict[int, int] = {}
        override_idx: list[int] = []
        override_len: list[int] = []
        for li, slot in enumerate(slot_list):
            if not is_comm[li] or slot in skip:
                continue
            vlen = int(valid_canvas_len_np[li])
            if vlen <= 0:
                continue
            row = states.argmax_host_row(slot, device)[:vlen]
            closers = np.where(row == _TOOL_CALL_CLOSE_ID)[0]
            if closers.size == 0:
                continue  # no closer -> commit the whole block (no-op)
            p = int(closers[0])
            if p + 1 >= vlen:
                continue  # closer at the very end -> nothing follows
            # A following call exists iff an opener (48) appears AFTER this closer.
            if not bool(np.any(row[p + 1 :] == _TOOL_CALL_OPEN_ID)):
                continue  # last call -> commit whole (trailing stop kept)
            st = states.constraint_states.get(slot)
            override_idx.append(idx_list[li])
            override_len.append(p + 1)
            trunc[slot] = p + 1
            # Calls 2..N regenerate in the next block,
            # which must validate from grammar START. Keep reasoning_closed True
            # (next block re-engages at pos 0 via the tool-call gate); reset the
            # rest so committed_len=0 / boundary_pos=-1 / start_body=0.
            if st is not None:
                st.engaged = False
                st.boundary_pos = -1
                st.renoise_count = 0  # fresh re-noise budget for the next call
                st.checker.reset()
                # A parallel-call split resets the committed matcher to
                # grammar START so call2 validates fresh; continuing call1's state
                # would reject or corrupt call2's opener. Drop any pending
                # commit-advance from call1's block.
                st._pending_commit_body = None
                st._pending_abandon = False
        if override_idx:
            di = torch.tensor(override_idx, device=device, dtype=torch.int64)
            vals = torch.tensor(override_len, device=device, dtype=num_sampled.dtype)
            num_sampled[di] = vals
        return trunc

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
        constrained = _constrained_decode_enabled()

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill, init
        # canvas for any new prefills, and stage decode slot indices to GPU. ---
        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            ps = slots_np[prefill_indices_np]
            states.init_canvas(ps)
            self.req_states.draft_tokens[ps, :CL] = states.canvas[ps]
            ps_gpu = async_copy_to_gpu(
                ps.astype(np.int64), device=states.is_encoder_phase.device
            )
            states.is_encoder_phase.index_fill_(0, ps_gpu, False)

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
        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Phantom (padded) positions are zeroed → uniform logits → high
        # entropy (no premature convergence) and argmax 0 (stable); they are
        # never committed (num_sampled == real length).
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len  # row offset per req
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)] * valid.reshape(-1, 1).to(logits.dtype)

        # Cleared inside _compiled_sample_step so prefill/non-decode slots stay 0.
        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]

        all_slots = input_batch.idx_mapping[:num_reqs]

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit→False, converge→True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        # --- PRE-MASK entropy capture (constrained path) -----------------------
        # Snapshot the PRE-grammar-mask logits so _compiled_sample_step can
        # compute the accept/convergence entropy from the model's TRUE confidence
        # (premask_conv_entropy). Must be captured before constraint masking
        # applies the −inf mask. None on the base path lets the compiler prune
        # that branch. The clone is O(num_decode*CL*vocab), paid only when the
        # constrained path is on.
        premask_logits = None
        if constrained:
            premask_logits = logits.clone()

        # --- Grammar-constraint pre-step guide -----------------------------
        # For structured-output requests, mask THIS step's per-position logits to
        # grammar-valid + extendable tokens (from the bounded lookahead's
        # surviving beams) so the compiled denoise is biased toward extendable
        # trajectories. This is a GUIDE (marginal -> admits cross-beam combos),
        # NOT the reliability guarantee — that's the post-step veto below + the
        # commit-time validation. The guide is a no-op unless
        # VLLM_DIFFUSION_CONSTRAINED_DECODE=1 and a slot carries a checker. It reads
        # argmax_canvas with one GPU-to-CPU synchronization.
        if constrained and states.constraint_states and num_decode > 0:
            logits = self._apply_constraint_verify(
                logits, states, decode_slots_np, CL, device
            )

        # --- Phase-1 reasoning-mask (VLLM_DIFFUSION_CONSTRAINED_DECODE) -----------
        # Before the compiled step, for every decode slot still in reasoning
        # (101 not yet committed) forbid all special/control ids except {100,101}
        # so a forbidden special never becomes the argmax. Strict no-op unless
        # the flag is on (forbidden_ids None). Runs on denoise AND commit.
        if states.reasoning_forbidden_ids is not None and num_decode > 0:
            logits = self._apply_reasoning_mask(logits, states, decode_slots_np, CL)

        # --- Single compiled call: temp → sample → probs → post-process ---
        scaled = _compiled_sample_step(
            logits,
            decode_slots,
            decode_idx,
            all_slots,
            valid_canvas_len,
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
            CL=self.canvas_length,
            ST=states.stability_threshold,
            entropy_bound=self.entropy_bound,
            converge_on_prefix=constrained,
            nansafe_entropy=constrained,
            premask_conv_entropy=constrained,
            premask_logits=premask_logits,
        )

        # One host snapshot feeds verification, truncation, and the next frontier
        # pass without repeated device synchronization.
        if constrained and num_decode > 0:
            states.snapshot_argmax_host(decode_slots, decode_slots_np)

        # --- Grammar-constraint post-step convergence veto -----------------
        # The pre-step mask is only a guide (marginal). The "always-extendable"
        # backstop: after the compiled step, for any constrained slot that just
        # CONVERGED (about to commit+emit), validate the REALIZED joint committed
        # body; if it isn't grammar-valid-AND-complete, suppress the convergence
        # and DIRECTLY re-noise the invalid tail in states.canvas (a real canvas
        # write, not an eb_mask trick). At max steps, force a grammar-valid
        # completion instead of immediately emitting the invalid tail. Matcher
        # setup or runtime failures explicitly fail open to unconstrained decoding.
        if constrained and states.constraint_states and num_decode > 0:
            self._constraint_poststep_veto(
                states, decode_slots_np, is_committing, valid_canvas_len_np, CL, device
            )

        # --- Truncate committed block after <channel|>(101) (env-gated) --------
        # Cap num_sampled at the boundary so the scheduler rolls back the
        # post-101 tail. This runs whenever constrained decoding is enabled, even
        # if no per-request grammar is attached. Returns {slot:trunc_len}.
        trunc_map: dict[int, int] = {}
        if constrained and num_decode > 0:
            trunc_map = self._truncate_after_channel(
                states,
                decode_slots_np,
                decode_indices_np,
                is_committing,
                num_sampled,
                valid_canvas_len_np,
                device,
            )

        # --- Block-per-call: truncate committed block after a tool-call closer
        # <tool_call|>(49) when another call (opener 48) follows. This makes
        # each parallel call commits in its own block and validates from grammar
        # START -> no cross-call matcher mis-parse. skip_slots avoids re-writing
        # num_sampled for a slot the channel pass already truncated this step.
        if constrained and num_decode > 0:
            tc_trunc = self._truncate_after_toolcall(
                states,
                decode_slots_np,
                decode_indices_np,
                is_committing,
                num_sampled,
                valid_canvas_len_np,
                device,
                skip_slots=set(trunc_map),
            )
            trunc_map.update(tc_trunc)

        # --- Logprobs: stash on convergence, return on commit ---
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        is_decode_np = per_req_nlogits_np > 0

        logprobs_tensors = None
        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)
        if max_num_logprobs >= 0:
            # Denoise steps that just converged: the compiled step flipped
            # is_encoder_phase from False→True. Detect as slots where
            # is_encoder_phase is now True but is_committing was False.
            converged_mask = states.is_encoder_phase[decode_slots]
            just_converged = converged_mask & ~is_committing
            if just_converged.any():
                flat_logits = scaled.reshape(-1, scaled.shape[-1])
                argmax_tokens = scaled.argmax(dim=-1)
                for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                    li = local_idx.item()
                    slot = decode_slots[local_idx]
                    # Stash only the real canvas positions (== CL unless this
                    # canvas was truncated near max_model_len); padded tail
                    # positions are never emitted.
                    k_i = int(valid_canvas_len_np[li])
                    start = li * CL
                    self._pending_logprobs[slot.item()] = compute_topk_logprobs(
                        flat_logits[start : start + k_i],
                        max_num_logprobs,
                        argmax_tokens[local_idx][:k_i],
                    )

            # Commit steps: is_committing was True at entry. Reassemble
            # previously stashed logprobs and attach to SamplerOutput.
            if is_committing.any() and self._pending_logprobs:
                parts_ids, parts_lp, parts_ranks = [], [], []
                cu_gen: list[int] = []
                flat_offset = 0
                for i in range(num_reqs):
                    cu_gen.append(flat_offset)
                    slot = int(slots_np[i])
                    if is_decode_np[i] and slot in self._pending_logprobs:
                        lp = self._pending_logprobs.pop(slot)
                        parts_ids.append(lp.logprob_token_ids)
                        parts_lp.append(lp.logprobs)
                        parts_ranks.append(lp.selected_token_ranks)
                        flat_offset += lp.logprobs.shape[0]
                if parts_ids:
                    logprobs_tensors = LogprobsTensors(
                        logprob_token_ids=torch.cat(parts_ids),
                        logprobs=torch.cat(parts_lp),
                        selected_token_ranks=torch.cat(parts_ranks),
                        cu_num_generated_tokens=cu_gen,
                    )

        return self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            device,
            logprobs_tensors=logprobs_tensors,
        )
