# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact BF16 DiffusionGemma text normalization with CUPTI.

CUDA graphs exclude Python launch overhead; cold L2 avoids warm-cache artifacts.
Reported minimum bytes count required activation reads/writes, excluding weights
and intermediate traffic. Example: --rows 33 --device cuda:0 --output norm.json
"""

import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
from vllm.model_executor.models.diffusion_gemma_text_norm import (
    fused_feedforward_inputs,
    fused_feedforward_output,
    fused_qkv_norm,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def reference_inputs(x, w1, w2, scale, root):
    dense = rms_norm_batch_invariant(x, w1)
    expert = rms_norm_batch_invariant(x, w2)
    router = rms_norm_batch_invariant(x, None)
    router = (router * root.to(x.dtype)) * scale.to(x.dtype)
    return dense, expert, router


def reference_output(a, b, residual, w1, w2, w3, scale):
    dense = rms_norm_batch_invariant(a, w1)
    expert = rms_norm_batch_invariant(b, w2)
    norm = rms_norm_batch_invariant(dense + expert, w3)
    return (norm + residual) * scale


def reference_qkv(x, wq, wk, query_heads, kv_heads, head_dim):
    heads = query_heads, kv_heads, kv_heads
    pieces = x.split([count * head_dim for count in heads], dim=-1)
    return tuple(
        rms_norm_batch_invariant(
            piece.unflatten(-1, (count, head_dim)), weight
        ).flatten(-2)
        for piece, count, weight in zip(pieces, heads, (wq, wk, None))
    )


def cases(rows, device):
    randn = partial(torch.randn, dtype=torch.bfloat16, device=device)
    x, b, residual = randn(3, rows, 2816).unbind()
    w1, w2, w3 = randn(3, 2816).unbind()
    root = torch.tensor(2816**-0.5, device=device)
    scale = torch.randn(2816, device=device)
    arguments = x, w1, w2, scale, root
    yield (
        "feedforward_inputs",
        partial(reference_inputs, *arguments),
        partial(fused_feedforward_inputs, *arguments),
        4 * x.numel() * x.element_size(),
    )
    scalar = torch.tensor([1.328125], dtype=x.dtype, device=device)
    arguments = x, b, residual, w1, w2, w3, scalar
    yield (
        "feedforward_output",
        partial(reference_output, *arguments),
        partial(fused_feedforward_output, *arguments),
        4 * x.numel() * x.element_size(),
    )
    for head_dim, kv_heads in ((256, 8), (512, 2)):
        packed = randn(rows, (16 + 2 * kv_heads) * head_dim)
        wq, wk = randn(2, head_dim).unbind()
        arguments = packed, wq, wk, 16, kv_heads, head_dim
        yield (
            f"qkv_heads16_kv{kv_heads}_dim{head_dim}",
            partial(reference_qkv, *arguments),
            partial(fused_qkv_norm, *arguments),
            2 * packed.numel() * packed.element_size(),
        )


@torch.inference_mode()
def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1049, 8392, 33568])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("--rows values must be positive")
    torch.accelerator.set_device_index(torch.device(args.device).index or 0)
    torch.manual_seed(9212026)
    records = []
    for rows in args.rows:
        for name, reference, candidate, minimum_bytes in cases(rows, args.device):
            expected, actual = reference(), candidate()
            expected = (expected,) if isinstance(expected, torch.Tensor) else expected
            actual = (actual,) if isinstance(actual, torch.Tensor) else actual
            for value, target in zip(actual, expected):
                assert value.dtype == target.dtype == torch.bfloat16
                torch.testing.assert_close(
                    value.view(torch.int16), target.view(torch.int16), atol=0, rtol=0
                )
            torch.accelerator.synchronize()
            timings = [
                statistics.median(
                    bench_gpu_time_with_cupti(
                        function,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        dry_run_iters=5,
                        repeat_iters=20,
                    )
                )
                for function in (reference, candidate)
            ]
            record = dict(
                operation=name,
                rows=rows,
                exact=True,
                reference_ms=timings[0],
                fused_ms=timings[1],
                speedup=timings[0] / timings[1],
                minimum_bytes=minimum_bytes,
                fused_gbps_minimum_bytes=minimum_bytes / (timings[1] * 1e6),
            )
            print(json.dumps(record), flush=True)
            records.append(record)
    if args.output:
        metadata = dict(
            device=args.device,
            gpu=torch.cuda.get_device_name(),
            seed=9212026,
            torch_version=torch.__version__,
            dtype="bfloat16",
            timing="CUPTI CUDA graphs, cold L2, median of 20 samples",
            bytes="Minimum activation traffic; excludes weights and intermediates",
        )
        args.output.write_text(
            json.dumps(dict(metadata=metadata, results=records), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
