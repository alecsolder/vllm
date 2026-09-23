# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DiffusionGemma attention launch parameters without kernel edits.

Uses seeded synthetic BF16 Q/K/V, shared prompt blocks, per-request causal flags,
and optional logical prompt/canvas splits. Timings cover only attention, with
CUDA graphs and CUPTI cold-L2 timing. Compilation, reference checks, allocation,
metadata preparation, and cache writes are excluded. Useful FLOPs count QK/PV
for permitted query/key pairs; ideal bytes count Q/output and unique KV blocks
once, not measured memory traffic. Every candidate is compared bitwise with the
current production launch and finite differences are reported, never hidden.

Example (CUPTI Python >= 13 must be importable):
    CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 .venv/bin/python \
        benchmarks/kernels/benchmark_diffusion_gemma_attention.py \
        --batches 3 --tails 0 --output /tmp/attention-tuning
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import regex as re
import torch

CONFIGS = {
    "current": {},
    "m16_w4_s1": {"num_warps": 4, "num_stages": 1},
    "m16_w4_s2": {"num_warps": 4, "num_stages": 2},
    "m16_w8_s1": {"num_warps": 8, "num_stages": 1},
    "m16_w8_s2": {"num_warps": 8, "num_stages": 2},
    "m16_w8_s3": {"num_warps": 8, "num_stages": 3},
    "m32_w8_s2": {"BLOCK_M": 32, "num_warps": 8, "num_stages": 2},
    "m32_w8_s3": {"BLOCK_M": 32, "num_warps": 8, "num_stages": 3},
}


class LaunchOverride:
    def __init__(self, kernel):
        self.kernel = kernel
        self.config: dict[str, int] = {}
        self.compiled = None
        self.launch: dict[str, Any] = {}

    def __getitem__(self, grid):
        def run(**kwargs):
            kwargs.update(self.config)
            assert not kwargs["IS_3D"]
            block_q = kwargs["BLOCK_M"] // kwargs["num_queries_per_kv"]
            kwargs["BLOCK_Q"] = block_q
            launch_grid = (
                kwargs["query_ptr"].shape[0] // block_q + kwargs["num_seqs"],
                grid[1],
            )
            self.launch = {
                "grid": launch_grid,
                **{
                    key: kwargs[key]
                    for key in ("BLOCK_M", "BLOCK_Q", "TILE_SIZE", "BLOCK_SIZE")
                },
            }
            self.compiled = self.kernel[launch_grid](**kwargs)
            return self.compiled

        return run


def make_case(args, batch, tail, head_dim, kv_heads, window):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    seq_len = args.prompt + args.canvas
    pages = math.ceil(seq_len / args.block_size)
    shared_pages = (args.prompt - tail) // args.block_size
    tables = []
    next_page = shared_pages
    for _ in range(batch):
        private = list(range(next_page, next_page + pages - shared_pages))
        tables.append(list(range(shared_pages)) + private)
        next_page += len(private)
    table = torch.tensor(tables, dtype=torch.int32, device="cuda")
    cache_shape = (next_page, args.block_size, kv_heads, 2 * head_dim)
    cache = torch.randn(
        cache_shape, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # Private tail pages still contain the same prompt KV for every child.
    for child in range(1, batch):
        for page in range(shared_pages, math.ceil(args.prompt / args.block_size)):
            length = min(args.block_size, args.prompt - page * args.block_size)
            cache[tables[child][page], :length].copy_(cache[tables[0][page], :length])
    key, value = cache.split(head_dim, dim=-1)
    query = torch.randn(
        (batch * (args.canvas + tail), 16 // args.tensor_parallel_size, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(query)
    query_lens = [tail, args.canvas] * batch if tail else [args.canvas] * batch
    seq_lens = [args.prompt, seq_len] * batch if tail else [seq_len] * batch
    flags = [True, False] * batch if tail else [False] * batch
    if tail:
        table = table.repeat_interleave(2, dim=0)
    cu_query = torch.tensor([0, *query_lens], dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    causal = torch.tensor(flags, dtype=torch.bool, device="cuda")
    kwargs = dict(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_query,
        max_seqlen_q=max(query_lens),
        seqused_k=kv_lens,
        max_seqlen_k=seq_len,
        softmax_scale=head_dim**-0.5,
        causal=causal,
        window_size=(window - 1, 0) if window else (-1, -1),
        block_table=table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    allowed_pairs = 0
    for q_len, k_len, is_causal in zip(query_lens, seq_lens, flags):
        for pos in range(k_len - q_len, k_len):
            lo = max(0, pos - window + 1) if window else 0
            hi = min(k_len, pos + window) if window else k_len
            if is_causal:
                hi = min(hi, pos + 1)
            allowed_pairs += hi - lo
    metrics = {
        "useful_flops": 4 * query.shape[1] * head_dim * allowed_pairs,
        "ideal_bytes": 2 * query.numel() * query.element_size()
        + 2 * key.numel() * key.element_size(),
    }
    return kwargs, (query_lens, seq_lens, flags), metrics


def reference_attention(kwargs, lengths, window):
    query_lens, seq_lens, flags = lengths
    outputs = []
    offset = 0
    for idx, (q_len, k_len, causal) in enumerate(zip(query_lens, seq_lens, flags)):
        indices = kwargs["block_table"][idx].long()
        key = kwargs["k"][indices].flatten(0, 1)[:k_len]
        value = kwargs["v"][indices].flatten(0, 1)[:k_len]
        groups = kwargs["q"].shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        value = value.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        query = kwargs["q"][offset : offset + q_len].transpose(0, 1).float()
        logits = torch.matmul(query, key.transpose(1, 2)) * kwargs["softmax_scale"]
        q_pos = torch.arange(k_len - q_len, k_len, device="cuda")[:, None]
        k_pos = torch.arange(k_len, device="cuda")[None, :]
        mask = torch.ones((q_len, k_len), dtype=torch.bool, device="cuda")
        if causal:
            mask &= k_pos <= q_pos
        if window:
            mask &= (q_pos - k_pos).abs() < window
        logits.masked_fill_(~mask, -float("inf"))
        outputs.append(torch.matmul(logits.softmax(-1), value).transpose(0, 1))
        offset += q_len
    return torch.cat(outputs).to(kwargs["out"].dtype)


def differences(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "different_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
        "max_abs": float(delta.max()),
    }


def save_kernel(wrapper, directory):
    kernel = wrapper.compiled
    tag = kernel.hash[:16]
    destination = directory / tag
    if not destination.exists():
        destination.mkdir(parents=True)
        for kind in ("ptx", "ttgir", "llir", "cubin"):
            code = kernel.asm[kind]
            path = destination / f"kernel.{kind}"
            path.write_bytes(code) if isinstance(code, bytes) else path.write_text(code)
        import triton

        tool = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
        sass = subprocess.check_output(
            [str(tool), "--dump-sass", str(destination / "kernel.cubin")], text=True
        )
        (destination / "kernel.sass").write_text(sass)
        resource = subprocess.check_output(
            [str(tool), "--dump-resource-usage", str(destination / "kernel.cubin")],
            text=True,
        )
        metadata = {
            "metadata": kernel.metadata._asdict(),
            "n_regs": kernel.n_regs,
            "n_spills": kernel.n_spills,
            "resource_usage": resource,
            "sass_instruction_counts": {
                op: len(re.findall(r"\b" + op + r"\b", sass))
                for op in ("LDL", "STL", "HMMA", "HGMMA")
            },
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
    return {
        "artifact": str(destination),
        **json.loads((destination / "metadata.json").read_text()),
    }


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 8])
    parser.add_argument("--tails", nargs="+", type=int, default=[0, 13, 31])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--prompt", type=int, default=895)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--tensor-parallel-size", type=int, choices=[1, 2, 4, 8], default=1
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["sliding", "global"],
        default=["sliding", "global"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    if os.environ["VLLM_BATCH_INVARIANT"] != "1":
        parser.error("VLLM_BATCH_INVARIANT must be 1")
    if any(tail < 0 or tail > args.prompt for tail in args.tails):
        parser.error("tails must be between zero and prompt length")
    from cupti import cupti  # noqa: F401
    from flashinfer.testing import bench_gpu_time_with_cupti

    from vllm.v1.attention.ops import triton_unified_attention as attention

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >= 13 is required")
    torch.accelerator.set_device_index(0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    original = attention.kernel_unified_attention
    wrapper = LaunchOverride(original)
    attention.kernel_unified_attention = wrapper
    source = Path(attention.__file__)
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "device": str(torch.cuda.get_device_properties(0)),
        "versions": {
            name: version(name) for name in ("torch", "triton", "cupti-python")
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "timing": "CUPTI; CUDA graph; cold L2; attention kernel only",
        "input_kind": "seeded synthetic BF16, not captured model activations",
        "cases": [],
    }
    output_path = args.output / "results.json"
    configs = {"current": {}, **{name: CONFIGS[name] for name in args.configs}}
    try:
        for batch in args.batches:
            for tail in args.tails:
                for kind, head_dim, kv_heads, window in (
                    ("sliding", 256, 8, 1024),
                    ("global", 512, 2, None),
                ):
                    if kind not in args.kinds:
                        continue
                    kv_heads = max(1, kv_heads // args.tensor_parallel_size)
                    name = f"{kind}_n{batch}_tail{tail}"
                    kwargs, lengths, metrics = make_case(
                        args, batch, tail, head_dim, kv_heads, window
                    )
                    case = {"name": name, **metrics, "configs": {}}
                    report["cases"].append(case)
                    wrapper.config = {}
                    attention.unified_attention(**kwargs)
                    baseline = kwargs["out"].clone()
                    reference = reference_attention(kwargs, lengths, window)
                    case["reference_check"] = differences(baseline, reference)
                    torch.testing.assert_close(
                        baseline, reference, atol=1.5e-2, rtol=1e-2
                    )
                    torch.save(baseline.cpu(), args.output / f"{name}-baseline.pt")
                    for config_name, config in configs.items():
                        row: dict[str, Any] = {"config": config, "samples_ms": []}
                        case["configs"][config_name] = row
                        wrapper.config = config
                        try:
                            attention.unified_attention(**kwargs)
                            row["correctness"] = differences(kwargs["out"], baseline)
                            if not row["correctness"]["finite"]:
                                raise ValueError("Nonfinite output")
                            row["launch"] = wrapper.launch
                            row["kernel"] = save_kernel(
                                wrapper, args.output / "generated"
                            )
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            print(f"{name} {config_name}: {row['error']}", flush=True)
                        write_report(output_path, report)
                    for round_idx in range(args.rounds):
                        order = list(configs)
                        if round_idx % 2:
                            order.reverse()
                        for config_name in order:
                            row = case["configs"][config_name]
                            if "error" in row:
                                continue
                            wrapper.config = configs[config_name]
                            measured = bench_gpu_time_with_cupti(
                                lambda kwargs=kwargs: attention.unified_attention(
                                    **kwargs
                                ),
                                dry_run_iters=args.warmup,
                                repeat_iters=args.repeat,
                                use_cuda_graph=True,
                                cold_l2_cache=True,
                            )
                            row["samples_ms"].extend(float(t) for t in measured)
                            row["median_us"] = (
                                statistics.median(row["samples_ms"]) * 1e3
                            )
                            row["tflops"] = metrics["useful_flops"] / (
                                row["median_us"] * 1e6
                            )
                            row["ideal_gbps"] = metrics["ideal_bytes"] / (
                                row["median_us"] * 1e3
                            )
                            print(
                                f"{name} {config_name} round={round_idx} "
                                f"us={row['median_us']:.3f} "
                                f"diff={row['correctness']['different_elements']} "
                                f"regs={row['kernel']['n_regs']} "
                                f"spills={row['kernel']['n_spills']}",
                                flush=True,
                            )
                            write_report(output_path, report)
    finally:
        attention.kernel_unified_attention = original
        report["source_unchanged"] = (
            report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        write_report(output_path, report)


if __name__ == "__main__":
    main()
