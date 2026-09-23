# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 text normalization fusions preserving batch-invariant reduction order.

Keep 1024-element reduction chunks, default FP contraction, and the explicit
BF16 materialization boundaries aligned with ``rms_norm_batch_invariant``.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _triple_norm(X, W1, W2, SCALE, ROOT, O1, O2, OR, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols
    sum_sq = tl.zeros([1], tl.float32)
    for start in range(0, n_cols, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + base + col, col < n_cols, other=0.0).to(tl.float32)
        sq = x * x
        sum_sq += tl.sum(tl.where(col < n_cols, sq, 0.0))
    mean_sq = sum_sq / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    root = tl.load(ROOT).to(X.dtype.element_ty).to(tl.float32)
    for start in range(0, n_cols, BLOCK):
        col = start + tl.arange(0, BLOCK)
        valid = col < n_cols
        x = tl.load(X + base + col, valid, other=0.0).to(tl.float32)
        norm = x * inv_rms
        w1 = tl.load(W1 + col, valid, other=1.0).to(tl.float32)
        w2 = tl.load(W2 + col, valid, other=1.0).to(tl.float32)
        scale = (
            tl.load(SCALE + col, valid, other=1.0).to(X.dtype.element_ty).to(tl.float32)
        )
        out1 = (norm * w1).to(X.dtype.element_ty)
        out2 = (norm * w2).to(X.dtype.element_ty)
        router = norm.to(X.dtype.element_ty).to(tl.float32)
        router = (router * root).to(X.dtype.element_ty).to(tl.float32)
        router = (router * scale).to(X.dtype.element_ty)
        tl.store(O1 + base + col, out1, valid)
        tl.store(O2 + base + col, out2, valid)
        tl.store(OR + base + col, router, valid)


def fused_feedforward_inputs(x, w1, w2, scale, root, eps=1e-6):
    """Share one RMS reduction across dense, expert, and router inputs."""
    assert x.ndim == 2 and x.shape[1] > 0 and root.numel() == 1
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    assert all(w.is_contiguous() and w.numel() == x.shape[-1] for w in (w1, w2, scale))
    outputs = tuple(torch.empty_like(x) for _ in range(3))
    if x.numel() == 0:
        return outputs
    _triple_norm[(x.numel() // x.shape[-1],)](
        x,
        w1,
        w2,
        scale,
        root,
        *outputs,
        x.shape[-1],
        eps,
        BLOCK=1024,
    )
    return outputs


@triton.jit
def _packed_qkv_norm(
    X,
    WQ,
    WK,
    Q,
    K,
    V,
    eps,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // (HQ + 2 * HK)
    head = pid % (HQ + 2 * HK)
    col = tl.arange(0, BLOCK)
    valid = col < D
    x = tl.load(X + row * (HQ + 2 * HK) * D + head * D + col, valid, other=0.0).to(
        tl.float32
    )
    sum_sq = tl.zeros([1], tl.float32)
    sum_sq += tl.sum(tl.where(valid, x * x, 0.0))
    norm = x * (1.0 / tl.sqrt(sum_sq / D + eps))
    if head < HQ:
        weight = tl.load(WQ + col, valid, other=1.0).to(tl.float32)
        value = (norm * weight).to(X.dtype.element_ty)
        tl.store(Q + row * HQ * D + head * D + col, value, valid)
    elif head < HQ + HK:
        weight = tl.load(WK + col, valid, other=1.0).to(tl.float32)
        value = (norm * weight).to(X.dtype.element_ty)
        tl.store(K + row * HK * D + (head - HQ) * D + col, value, valid)
    else:
        value = norm.to(X.dtype.element_ty)
        tl.store(V + row * HK * D + (head - HQ - HK) * D + col, value, valid)


def fused_qkv_norm(x, wq, wk, query_heads, kv_heads, head_dim, eps=1e-6):
    """Normalize packed QKV into contiguous outputs without input copies."""
    assert x.ndim == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    assert 0 < head_dim <= 1024 and query_heads > 0 and kv_heads > 0
    assert x.shape[1] == (query_heads + 2 * kv_heads) * head_dim
    assert all(w.numel() == head_dim and w.is_contiguous() for w in (wq, wk))
    q = x.new_empty((x.shape[0], query_heads * head_dim))
    k = x.new_empty((x.shape[0], kv_heads * head_dim))
    v = torch.empty_like(k)
    if x.shape[0] == 0:
        return q, k, v
    _packed_qkv_norm[(x.shape[0] * (query_heads + 2 * kv_heads),)](
        x,
        wq,
        wk,
        q,
        k,
        v,
        eps,
        HQ=query_heads,
        HK=kv_heads,
        D=head_dim,
        BLOCK=1024,
    )
    return q, k, v


@triton.jit
def _combine(a, b, w1, w2, inv1, inv2):
    first = ((a * inv1) * w1).to(tl.bfloat16).to(tl.float32)
    second = ((b * inv2) * w2).to(tl.bfloat16).to(tl.float32)
    return (first + second).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _final_ffn(
    A, B, R, W1, W2, W3, SCALE, OUT, n_cols, eps1, eps2, eps3, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols
    sum_a = tl.zeros([1], tl.float32)
    sum_b = tl.zeros([1], tl.float32)
    for start in range(0, n_cols, BLOCK):
        col = start + tl.arange(0, BLOCK)
        valid = col < n_cols
        a = tl.load(A + base + col, valid, other=0.0).to(tl.float32)
        b = tl.load(B + base + col, valid, other=0.0).to(tl.float32)
        sum_a += tl.sum(tl.where(valid, a * a, 0.0))
        sum_b += tl.sum(tl.where(valid, b * b, 0.0))
    inv1 = 1.0 / tl.sqrt(sum_a / n_cols + eps1)
    inv2 = 1.0 / tl.sqrt(sum_b / n_cols + eps2)
    sum_combined = tl.zeros([1], tl.float32)
    for start in range(0, n_cols, BLOCK):
        col = start + tl.arange(0, BLOCK)
        valid = col < n_cols
        a = tl.load(A + base + col, valid, other=0.0).to(tl.float32)
        b = tl.load(B + base + col, valid, other=0.0).to(tl.float32)
        w1 = tl.load(W1 + col, valid, other=1.0).to(tl.float32)
        w2 = tl.load(W2 + col, valid, other=1.0).to(tl.float32)
        combined = _combine(a, b, w1, w2, inv1, inv2)
        sum_combined += tl.sum(tl.where(valid, combined * combined, 0.0))
    inv3 = 1.0 / tl.sqrt(sum_combined / n_cols + eps3)
    scale = tl.load(SCALE).to(tl.float32)
    for start in range(0, n_cols, BLOCK):
        col = start + tl.arange(0, BLOCK)
        valid = col < n_cols
        a = tl.load(A + base + col, valid, other=0.0).to(tl.float32)
        b = tl.load(B + base + col, valid, other=0.0).to(tl.float32)
        w1 = tl.load(W1 + col, valid, other=1.0).to(tl.float32)
        w2 = tl.load(W2 + col, valid, other=1.0).to(tl.float32)
        combined = _combine(a, b, w1, w2, inv1, inv2)
        w3 = tl.load(W3 + col, valid, other=1.0).to(tl.float32)
        norm = ((combined * inv3) * w3).to(tl.bfloat16).to(tl.float32)
        residual = tl.load(R + base + col, valid, other=0.0).to(tl.float32)
        added = (norm + residual).to(tl.bfloat16).to(tl.float32)
        output = (added * scale).to(tl.bfloat16)
        tl.store(OUT + base + col, output, valid)


def fused_feedforward_output(
    a, b, residual, w1, w2, w3, scale, eps1=1e-6, eps2=1e-6, eps3=1e-6, out=None
):
    """Fuse the no-PLE dense/MoE postprocessing while retaining BF16 boundaries."""
    assert a.shape == b.shape == residual.shape
    assert a.dtype == b.dtype == residual.dtype == torch.bfloat16
    assert all(value.is_contiguous() for value in (a, b, residual, w1, w2, w3))
    assert a.ndim == 2 and a.shape[1] > 0
    assert all(
        weight.ndim == 1 and weight.numel() == a.shape[1] for weight in (w1, w2, w3)
    )
    assert scale.numel() == 1 and scale.dtype == torch.bfloat16
    if out is None:
        out = torch.empty_like(a)
    assert out.shape == a.shape and out.dtype == a.dtype and out.is_contiguous()
    if a.shape[0]:
        _final_ffn[(a.shape[0],)](
            a,
            b,
            residual,
            w1,
            w2,
            w3,
            scale,
            out,
            a.shape[1],
            eps1,
            eps2,
            eps3,
            BLOCK=1024,
            enable_fp_fusion=True,
        )
    return out
