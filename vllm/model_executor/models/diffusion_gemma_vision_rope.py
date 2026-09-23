# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerically stable RoPE kernels for the DiffusionGemma vision tower."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionAttention,
    apply_multidimensional_rope,
    eager_attention_forward,
)

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _round_bf16_to_fp32(value):
    # Preserve eager's intermediate BF16 rounding; casts alone are folded away.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["num_elements"])
def _gemma4_vision_rope_kernel(
    input_ptr,
    cosine_ptr,
    sine_ptr,
    output_ptr,
    num_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_axis_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_elements
    dimension = offsets % head_dim
    row = offsets // head_dim
    token = row // num_heads
    within_axis = dimension % (half_axis_dim * 2)
    first_half = within_axis < half_axis_dim
    rotated_dimension = tl.where(
        first_half,
        dimension + half_axis_dim,
        dimension - half_axis_dim,
    )

    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    rotated = tl.load(input_ptr + row * head_dim + rotated_dimension, mask=mask).to(
        tl.float32
    )
    cosine = tl.load(cosine_ptr + token * head_dim + dimension, mask=mask).to(
        tl.float32
    )
    sine = tl.load(sine_ptr + token * head_dim + dimension, mask=mask).to(tl.float32)

    sign = tl.where(first_half, -1.0, 1.0)
    direct_product = _round_bf16_to_fp32(value * cosine)
    rotated_product = _round_bf16_to_fp32(rotated * sign * sine)
    tl.store(output_ptr + offsets, direct_product + rotated_product, mask=mask)


@triton.jit(do_not_specialize=["num_elements"])
def _gemma4_vision_qk_rope_kernel(
    query_ptr,
    key_ptr,
    cosine_ptr,
    sine_ptr,
    query_output_ptr,
    key_output_ptr,
    num_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_axis_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_elements
    dimension = offsets % head_dim
    row = offsets // head_dim
    token = row // num_heads
    within_axis = dimension % (half_axis_dim * 2)
    first_half = within_axis < half_axis_dim
    rotated_dimension = tl.where(
        first_half,
        dimension + half_axis_dim,
        dimension - half_axis_dim,
    )
    rotated_offsets = row * head_dim + rotated_dimension

    query = tl.load(query_ptr + offsets, mask=mask).to(tl.float32)
    rotated_query = tl.load(query_ptr + rotated_offsets, mask=mask).to(tl.float32)
    key = tl.load(key_ptr + offsets, mask=mask).to(tl.float32)
    rotated_key = tl.load(key_ptr + rotated_offsets, mask=mask).to(tl.float32)
    cosine = tl.load(cosine_ptr + token * head_dim + dimension, mask=mask).to(
        tl.float32
    )
    sine = tl.load(sine_ptr + token * head_dim + dimension, mask=mask).to(tl.float32)

    sign = tl.where(first_half, -1.0, 1.0)
    query_product = _round_bf16_to_fp32(query * cosine)
    query_rotated_product = _round_bf16_to_fp32(rotated_query * sign * sine)
    key_product = _round_bf16_to_fp32(key * cosine)
    key_rotated_product = _round_bf16_to_fp32(rotated_key * sign * sine)
    tl.store(
        query_output_ptr + offsets,
        query_product + query_rotated_product,
        mask=mask,
    )
    tl.store(
        key_output_ptr + offsets,
        key_product + key_rotated_product,
        mask=mask,
    )


def _supports_diffusion_gemma_vision_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int,
) -> bool:
    return (
        current_platform.is_cuda()
        and value.device.type == "cuda"
        and cosine.device == value.device
        and sine.device == value.device
        and value.ndim == 4
        and cosine.shape == value.shape[:2] + value.shape[-1:]
        and sine.shape == cosine.shape
        and position_ids.ndim > 0
        and position_ids.shape[-1] == 2
        and unsqueeze_dim == 2
        and value.dtype == torch.bfloat16
        and cosine.dtype == torch.bfloat16
        and sine.dtype == torch.bfloat16
        and value.is_contiguous()
        and cosine.is_contiguous()
        and sine.is_contiguous()
        and value.shape[-1] % 4 == 0
        and 0 < value.numel() <= 2**31 - 1
    )


def diffusion_gemma_vision_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply Gemma4 vision RoPE while preserving eager bfloat16 rounding."""
    if not _supports_diffusion_gemma_vision_rope(
        value, cosine, sine, position_ids, unsqueeze_dim
    ):
        return apply_multidimensional_rope(
            value,
            cosine,
            sine,
            position_ids,
            unsqueeze_dim=unsqueeze_dim,
        )

    output = torch.empty_like(value)
    num_elements = value.numel()
    block_size = 256
    grid = (triton.cdiv(num_elements, block_size),)
    _gemma4_vision_rope_kernel[grid](
        value,
        cosine,
        sine,
        output,
        num_elements,
        num_heads=value.shape[-2],
        head_dim=value.shape[-1],
        half_axis_dim=value.shape[-1] // 4,
        block_size=block_size,
    )
    return output


def diffusion_gemma_vision_qk_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact Gemma4 vision RoPE to equal-shaped query and key tensors."""
    supported = query.shape == key.shape and _supports_diffusion_gemma_vision_rope(
        query, cosine, sine, position_ids, unsqueeze_dim
    )
    supported = supported and key.device == query.device
    supported = supported and key.dtype == query.dtype and key.is_contiguous()
    if not supported:
        return (
            diffusion_gemma_vision_rope(
                query, cosine, sine, position_ids, unsqueeze_dim
            ),
            diffusion_gemma_vision_rope(key, cosine, sine, position_ids, unsqueeze_dim),
        )

    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    num_elements = query.numel()
    block_size = 256
    grid = (triton.cdiv(num_elements, block_size),)
    _gemma4_vision_qk_rope_kernel[grid](
        query,
        key,
        cosine,
        sine,
        query_output,
        key_output,
        num_elements,
        num_heads=query.shape[-2],
        head_dim=query.shape[-1],
        half_axis_dim=query.shape[-1] // 4,
        block_size=block_size,
    )
    return query_output, key_output


class DiffusionGemmaVisionAttention(nn.Module):
    """Gemma4 vision attention using the exact pointwise RoPE fast path."""

    def __init__(self, source: Gemma4VisionAttention) -> None:
        super().__init__()
        self.layer_type = source.layer_type
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = source.is_causal
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.v_norm = source.v_norm
        self.train(source.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cosine, sine = position_embeddings
        if position_ids is None:
            raise ValueError("position_ids are required for vision RoPE")

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        query_states, key_states = diffusion_gemma_vision_qk_rope(
            query_states, key_states, cosine, sine, position_ids
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_norm(
            self.v_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


def replace_diffusion_gemma_vision_attentions(module: nn.Module) -> int:
    """Replace Gemma4 vision attention while retaining checkpoint parameters."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, Gemma4VisionAttention):
            setattr(module, child_name, DiffusionGemmaVisionAttention(child))
            replaced += 1
        else:
            replaced += replace_diffusion_gemma_vision_attentions(child)
    return replaced
