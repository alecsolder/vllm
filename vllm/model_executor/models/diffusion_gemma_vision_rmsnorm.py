# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerically stable RMSNorm kernels for the DiffusionGemma vision tower."""

import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit(do_not_specialize=["num_elements"])
def _gemma4_vision_square_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(output_ptr + offsets, value * value, mask=mask)


@triton.jit(do_not_specialize=["num_elements"])
def _gemma4_vision_apply_rms_kernel(
    input_ptr,
    weight_ptr,
    variance_ptr,
    output_ptr,
    num_elements,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    WITH_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    variance = tl.load(variance_ptr + offsets // hidden_size, mask=mask)
    output = value * tldevice.rsqrt(variance + epsilon)
    if WITH_WEIGHT:
        weight = tl.load(weight_ptr + offsets % hidden_size, mask=mask).to(tl.float32)
        output *= weight
    tl.store(output_ptr + offsets, output, mask=mask)


class DiffusionGemmaVisionRMSNorm(nn.Module):
    """Gemma4 vision RMSNorm with an opt-in, numerically exact fast path."""

    def __init__(self, source: Gemma4RMSNorm) -> None:
        super().__init__()
        self.eps = source.eps
        self.with_scale = source.with_scale
        self.training = source.training
        self.use_fast_path = True
        if self.with_scale:
            self.weight = source.weight
        else:
            self.register_parameter("weight", None)

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = hidden_states.float()
        mean_squared = normed_output.pow(2).mean(-1, keepdim=True) + self.eps
        normed_output = normed_output * torch.pow(mean_squared, -0.5)
        if self.weight is not None:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            not hidden_states.is_cuda
            or not current_platform.is_cuda()
            or not self.use_fast_path
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or hidden_states.ndim == 0
            or hidden_states.numel() == 0
            or hidden_states.numel() > 2**31 - 1
            or (
                self.weight is not None
                and (
                    self.weight.dtype != hidden_states.dtype
                    or self.weight.device != hidden_states.device
                    or not self.weight.is_contiguous()
                    or self.weight.ndim != 1
                    or self.weight.numel() != hidden_states.shape[-1]
                )
            )
        ):
            return self.forward_native(hidden_states)

        num_elements = hidden_states.numel()
        block_size = 256
        grid = (triton.cdiv(num_elements, block_size),)
        squared = torch.empty_like(hidden_states, dtype=torch.float32)
        _gemma4_vision_square_kernel[grid](
            hidden_states,
            squared,
            num_elements,
            BLOCK_SIZE=block_size,
        )
        variance = squared.mean(-1, keepdim=True)
        output = torch.empty_like(hidden_states)
        _gemma4_vision_apply_rms_kernel[grid](
            hidden_states,
            self.weight if self.weight is not None else hidden_states,
            variance,
            output,
            num_elements,
            hidden_size=hidden_states.shape[-1],
            epsilon=self.eps,
            WITH_WEIGHT=self.weight is not None,
            BLOCK_SIZE=block_size,
        )
        return output


def replace_diffusion_gemma_vision_rms_norms(module: nn.Module) -> int:
    """Replace Gemma4 vision norms while retaining checkpoint parameters."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, Gemma4RMSNorm):
            setattr(module, child_name, DiffusionGemmaVisionRMSNorm(child))
            replaced += 1
        else:
            replaced += replace_diffusion_gemma_vision_rms_norms(child)
    return replaced
