# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import fp8_allclose, fp8_ulp_distance, opcheck
from vllm import ir
from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models.diffusion_gemma_text_norm import (
    fused_feedforward_inputs,
    fused_feedforward_output,
    fused_qkv_norm,
)
from vllm.model_executor.models.diffusion_gemma_vision_rmsnorm import (
    DiffusionGemmaVisionRMSNorm,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx90a

    on_mi250 = on_gfx90a()
else:
    on_mi250 = False

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def _rms_norm_tolerance(dtype: torch.dtype) -> dict[str, float]:
    return ir.ops.rms_norm.get_tolerance(dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = x.new_empty_strided(x.size(), x.stride()) if add_residual else None
    if residual is not None:
        residual.normal_(std=scale)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_weightless(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size, has_weight=False).to(dtype=dtype)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None

    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    tol = _rms_norm_tolerance(dtype)
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], **tol)
        torch.testing.assert_close(out[1], ref_out[1], **tol)
    else:
        torch.testing.assert_close(out, ref_out, **tol)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, None, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm,
            (out, x, None, layer.variance_epsilon),
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.accelerator.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if on_gfx950() and dtype == torch.float16 and not add_residual:
            # Fusion may round normalized FP16 across an E4M3 boundary on gfx950.
            assert fp8_allclose(out_quant_fused, out_quant, rtol=0.125, atol=2e-3)
            assert int(fp8_ulp_distance(out_quant_fused, out_quant).max()) <= 1
        else:
            # Fused and unfused FP8 paths can land on opposite sides of an E4M3
            # tie; tolerate a tiny number of isolated fp8 outliers on ROCm.
            ulp = fp8_ulp_distance(out_quant, out_quant_fused)
            max_outliers = ulp.numel() // 100_000 + 8
            num_outliers = int((ulp > 0).sum().item())
            assert num_outliers <= max_outliers, (
                f"FP8 quant mismatch: {num_outliers} fp8 outliers "
                f"(allowed {max_outliers})"
            )
    else:
        torch.testing.assert_close(
            out_quant.to(dtype=torch.float32),
            out_quant_fused.to(dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_gemma_rms_norm_mixed_input_weight_dtype(default_vllm_config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = CUDA_DEVICES[0]
    torch.set_default_device(device)

    num_tokens, hidden_size = 32, 1024
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    layer = GemmaRMSNorm(hidden_size, eps=1e-6).to(device=device)
    layer.weight.data.normal_(mean=0.0, std=0.1)

    # Gemma uses fp32 weight parameter while activations can be bf16.
    assert layer.weight.dtype == torch.float32
    out = layer(x)

    x_fp32 = x.float()
    weight_fp32 = layer.weight.data.float() + 1.0
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_fp32 * torch.rsqrt(variance + layer.variance_epsilon) * weight_fp32).to(
        x.dtype
    )

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("shape", "with_scale"),
    [
        ((1, 2520, 1152), True),
        ((1, 2520, 16, 72), True),
        ((1, 2520, 16, 72), False),
    ],
)
@torch.inference_mode()
def test_diffusion_gemma_vision_rms_norm_exact(
    shape: tuple[int, ...], with_scale: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = CUDA_DEVICES[0]
    source = Gemma4RMSNorm(shape[-1], eps=1e-6, with_scale=with_scale).to(
        device=device, dtype=torch.bfloat16
    )
    if with_scale:
        source.weight.data.normal_(mean=1.0, std=0.1)
    layer = DiffusionGemmaVisionRMSNorm(source)

    set_random_seed(0)
    hidden_states = torch.randn(shape, dtype=torch.bfloat16, device=device)
    row_scales = torch.logspace(
        -5,
        5,
        hidden_states.numel() // shape[-1],
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    hidden_states.mul_(row_scales.reshape(*shape[:-1], 1))

    expected = source(hidden_states)
    actual = layer(hidden_states)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("with_scale", [False, True])
def test_diffusion_gemma_vision_rms_norm_cpu_preserves_input_and_parameters(
    with_scale: bool,
) -> None:
    """The native fallback keeps HF's non-mutating FP32 contract."""
    source = Gemma4RMSNorm(8, eps=1e-6, with_scale=with_scale).eval()
    layer = DiffusionGemmaVisionRMSNorm(source)
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    original = hidden_states.clone()

    expected = source(hidden_states)
    actual = layer(hidden_states)

    assert torch.equal(hidden_states, original)
    assert torch.equal(actual, expected)
    assert layer.training == source.training
    assert layer.state_dict().keys() == source.state_dict().keys()
    if with_scale:
        assert layer.weight is source.weight


@pytest.mark.parametrize("with_scale", [False, True])
def test_diffusion_gemma_vision_rms_norm_cpu_bf16_on_cuda_platform(
    monkeypatch,
    with_scale: bool,
) -> None:
    """A CUDA-capable host must still accept CPU tensors through HF fallback."""
    from vllm.model_executor.models import diffusion_gemma_vision_rmsnorm

    monkeypatch.setattr(
        diffusion_gemma_vision_rmsnorm.current_platform, "is_cuda", lambda: True
    )
    source = Gemma4RMSNorm(8, eps=1e-6, with_scale=with_scale).to(torch.bfloat16)
    layer = DiffusionGemmaVisionRMSNorm(source)
    hidden_states = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 8
    original = hidden_states.clone()

    actual = layer(hidden_states)

    assert actual.device.type == "cpu"
    assert torch.equal(actual, source(original))
    assert torch.equal(hidden_states, original)


def _assert_bf16_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.is_contiguous()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 33, 1049])
@torch.inference_mode()
def test_diffusion_gemma_text_feedforward_inputs_preserves_bf16_rounding(
    num_tokens: int,
) -> None:
    """Dense, expert, and router inputs retain their distinct BF16 boundaries."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    set_random_seed(0)
    device = CUDA_DEVICES[0]
    hidden_size = 2816
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    if num_tokens == 33:
        x = (x.float() * torch.logspace(-12, 12, hidden_size, device=device)).to(
            x.dtype
        )
    w1, w2 = torch.randn(2, hidden_size, dtype=x.dtype, device=device).unbind()
    scale = torch.randn(hidden_size, dtype=torch.float32, device=device)
    root = torch.tensor(hidden_size**0.5, dtype=torch.float32, device=device)
    if num_tokens:
        dense = rms_norm_batch_invariant(x, w1)
        expert = rms_norm_batch_invariant(x, w2)
        router = rms_norm_batch_invariant(x, None)
        router = (router * root.to(x.dtype)) * scale.to(x.dtype)
        expected = dense, expert, router
    else:
        expected = tuple(torch.empty_like(x) for _ in range(3))

    actual = fused_feedforward_inputs(x, w1, w2, scale, root)
    for output, reference in zip(actual, expected):
        _assert_bf16_bitwise_equal(output, reference)


@pytest.mark.parametrize(
    ("num_tokens", "kv_heads", "head_dim"),
    [
        (0, 8, 256),
        (1, 2, 512),
        (33, 8, 256),
        (33, 2, 512),
        (1049, 8, 256),
        (1049, 2, 512),
    ],
)
@torch.inference_mode()
def test_diffusion_gemma_text_qkv_norm_preserves_packed_head_layout(
    num_tokens: int, kv_heads: int, head_dim: int
) -> None:
    """Normalize packed local/global QKV exactly, including strided split views."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    set_random_seed(0)
    device = CUDA_DEVICES[0]
    query_heads = 16
    widths = (query_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
    packed = torch.randn(num_tokens, sum(widths), dtype=torch.bfloat16, device=device)
    if num_tokens == 33:
        packed = (
            packed.float() * torch.logspace(-12, 12, sum(widths), device=device)
        ).to(packed.dtype)
    wq, wk = torch.randn(2, head_dim, dtype=packed.dtype, device=device).unbind()
    views = packed.split(widths, dim=-1)
    if num_tokens > 1:
        assert all(not value.is_contiguous() for value in views)
    expected: list[torch.Tensor] = []
    if num_tokens:
        for value, weight, width in zip(views, (wq, wk, None), widths):
            normalized = rms_norm_batch_invariant(
                value.view(num_tokens, -1, head_dim), weight
            )
            assert isinstance(normalized, torch.Tensor)
            expected.append(normalized.view(num_tokens, width))
    else:
        expected = [torch.empty_like(value) for value in views]

    actual = fused_qkv_norm(packed, wq, wk, query_heads, kv_heads, head_dim)
    for output, reference in zip(actual, expected):
        _assert_bf16_bitwise_equal(output, reference)


@pytest.mark.parametrize(
    ("num_tokens", "values"),
    [
        (0, "random"),
        (1, "random"),
        (33, "extreme"),
        (33, "cancellation"),
        (1049, "random"),
    ],
)
@torch.inference_mode()
def test_diffusion_gemma_text_feedforward_output_preserves_bf16_rounding(
    num_tokens: int, values: str
) -> None:
    """Keep three RMS reductions, residual addition, and layer scaling bitwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    set_random_seed(0)
    device = CUDA_DEVICES[0]
    hidden_size = 2816
    a, b, residual = torch.randn(
        3, num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    ).unbind()
    w1, w2, w3 = torch.randn(3, hidden_size, dtype=a.dtype, device=device).unbind()
    scale = torch.tensor([1.328125], dtype=a.dtype, device=device)
    if values == "extreme":
        factor = torch.logspace(-12, 12, hidden_size, device=device)
        a, b, residual = (
            (value.float() * factor).to(a.dtype) for value in (a, b, residual)
        )
    elif values == "cancellation":
        b = -a
        w2 = w1
    if num_tokens:
        first = rms_norm_batch_invariant(a, w1)
        second = rms_norm_batch_invariant(b, w2)
        combined = first + second
        norm = rms_norm_batch_invariant(combined, w3)
        expected = (norm + residual) * scale
    else:
        expected = torch.empty_like(a)

    actual = fused_feedforward_output(a, b, residual, w1, w2, w3, scale)
    _assert_bf16_bitwise_equal(actual, expected)
