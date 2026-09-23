# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous utilities."""

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionAttention,
    apply_multidimensional_rope,
)

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.diffusion_gemma_vision_rope import (
    diffusion_gemma_vision_qk_rope,
    diffusion_gemma_vision_rope,
    replace_diffusion_gemma_vision_attentions,
)


def rotary_embedding_opcheck(
    rot,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding() is a in-place operation
    # that updates the query and key tensors.
    opcheck(
        torch.ops._C.rotary_embedding,
        (positions, query, key, rot.head_size, cos_sin_cache, rot.is_neox_style),
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_opcheck(
    default_vllm_config,
    dist_init,
    device,
    max_position,
    is_neox_style,
    rotary_dim,
    head_size,
    seq_len,
    use_key,
    head_stride_is_contiguous,
    dtype,
):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(
        batch_size, seq_len, num_heads, head_stride, dtype=dtype, device=device
    )
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if key is not None else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot,
            positions,
            query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if key is not None else None,
        )


@pytest.mark.parametrize("seq_len", [630, 1260, 2520])
@torch.inference_mode()
def test_diffusion_gemma_vision_rope_exact(seq_len: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260917 + seq_len)
    value = torch.randn(
        1,
        seq_len,
        16,
        72,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cosine = torch.randn(
        1,
        seq_len,
        72,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    sine = torch.randn(
        1,
        seq_len,
        72,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    position_ids = torch.zeros(1, seq_len, 2, dtype=torch.long, device=device)

    expected = apply_multidimensional_rope(
        value, cosine, sine, position_ids, unsqueeze_dim=2
    )
    actual = diffusion_gemma_vision_rope(
        value, cosine, sine, position_ids, unsqueeze_dim=2
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "head_dim"),
    [(1, 1, 1, 4), (2, 7, 3, 8), (1, 2520, 16, 72)],
)
@torch.inference_mode()
def test_diffusion_gemma_vision_qk_rope_exact(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        20260918 + batch_size + seq_len + num_heads + head_dim
    )
    shape = (batch_size, seq_len, num_heads, head_dim)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    cosine = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    sine = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    position_ids = torch.zeros(batch_size, seq_len, 2, dtype=torch.long, device=device)

    expected_query = apply_multidimensional_rope(
        query, cosine, sine, position_ids, unsqueeze_dim=2
    )
    expected_key = apply_multidimensional_rope(
        key, cosine, sine, position_ids, unsqueeze_dim=2
    )
    actual_query, actual_key = diffusion_gemma_vision_qk_rope(
        query, key, cosine, sine, position_ids, unsqueeze_dim=2
    )
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


@torch.inference_mode()
def test_diffusion_gemma_vision_rope_cpu_fallback() -> None:
    generator = torch.Generator().manual_seed(20260918)
    shape = (1, 7, 2, 8)
    query = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    cosine = torch.randn(1, 7, 8, dtype=torch.bfloat16, generator=generator)
    sine = torch.randn(1, 7, 8, dtype=torch.bfloat16, generator=generator)
    position_ids = torch.zeros(1, 7, 2, dtype=torch.long)

    expected_query = apply_multidimensional_rope(
        query, cosine, sine, position_ids, unsqueeze_dim=2
    )
    expected_key = apply_multidimensional_rope(
        key, cosine, sine, position_ids, unsqueeze_dim=2
    )
    actual_query, actual_key = diffusion_gemma_vision_qk_rope(
        query, key, cosine, sine, position_ids, unsqueeze_dim=2
    )
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


@torch.inference_mode()
def test_diffusion_gemma_vision_replacement_preserves_eval() -> None:
    """Replacing an evaluated attention module must not enable dropout."""
    config = Gemma4VisionConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        attention_dropout=0.5,
    )
    config._attn_implementation = "eager"
    source = Gemma4VisionAttention(config, layer_idx=0).eval()
    module = torch.nn.Sequential(source)
    hidden = torch.randn(1, 7, 8)
    embeddings = (torch.randn(1, 7, 4), torch.randn(1, 7, 4))
    positions = torch.zeros(1, 7, 2, dtype=torch.long)
    expected, _ = source(hidden, embeddings, position_ids=positions)

    assert replace_diffusion_gemma_vision_attentions(module) == 1
    assert not module[0].training
    assert all(
        actual is original
        for actual, original in zip(module[0].parameters(), source.parameters())
    )
    actual, _ = module[0](hidden, embeddings, position_ids=positions)
    assert torch.equal(actual, expected)
