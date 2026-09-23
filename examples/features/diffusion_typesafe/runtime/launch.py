# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the consolidated H200 BF16 profile and TypeSafe endpoint."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--optimize-text-32",
        action="store_true",
        help="Use the H200 profile for 32 text questions with 1024-token canvases",
    )
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=50000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--enforce-eager", action="store_true")
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument("--trace", type=Path, help="Override the decision JSONL path")
    tracing.add_argument("--no-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("--gpu must be nonnegative and --port must be between 1 and 65535")
    if not 1 <= args.canvas_length <= 1024:
        parser.error("--canvas-length must be between 1 and 1024")
    if args.optimize_text_32 and (not args.text_only or args.canvas_length != 1024):
        parser.error("--optimize-text-32 requires --text-only --canvas-length 1024")
    if not args.canvas_length < args.max_model_len <= 50000:
        parser.error("--max-model-len must exceed --canvas-length and be at most 50000")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not (args.checkpoint / "config.json").is_file():
        parser.error("--checkpoint must contain config.json")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.trace = (
        args.trace.expanduser().resolve()
        if args.trace is not None
        else args.output_dir / "decisions.jsonl"
    )
    return args


def environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(("VLLM_DIFFUSION_",)):
            env.pop(name)
    for name in (
        "PYTHONHOME",
        "VLLM_TRITON_USE_TD",
        "VLLM_TYPESAFE_TRACE",
        "VLLM_TORCH_PROFILER_DIR",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_PLUGINS",
    ):
        env.pop(name, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=f"{HERE}:{SOURCE}",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        VLLM_TYPESAFE_H200_PROFILE="1",
        VLLM_TYPESAFE_TEXT_ONLY="1" if args.text_only else "0",
        VLLM_TYPESAFE_SOURCE_ROOT=str(SOURCE),
        VLLM_BATCH_INVARIANT="1",
        VLLM_BATCH_INVARIANT_FAST_LINEAR="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_DIFFUSION_READ_FAST_ADMISSION="1",
        VLLM_DIFFUSION_FUSE_READ_PREFILL="1",
        VLLM_DIFFUSION_MM_PREFIX_MODE="off" if args.text_only else "hf_both",
        VLLM_DIFFUSION_FUSE_VISION_RMSNORM="1",
        VLLM_DIFFUSION_FUSE_VISION_ROPE="1",
        VLLM_DIFFUSION_ATTENTION_STAGES2="1",
        VLLM_DIFFUSION_MEMO_READ_PREFIX="0",
        VLLM_TYPESAFE_TRIM_CANVAS_PADDING="1",
        VLLM_TYPESAFE_MINIMUM_CANVAS_PADDING="4",
        VLLM_TYPESAFE_FULL_CANVAS_TYPES="",
    )
    env["VLLM_TYPESAFE_OPTIMIZE_TEXT_32"] = (
        "1" if getattr(args, "optimize_text_32", False) else "0"
    )
    if getattr(args, "optimize_text_32", False):
        env["VLLM_DIFFUSION_FUSE_TEXT_NORM"] = "1"
    for name, directory in (
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        env[name] = str(args.output_dir / "cache" / directory)
    if not args.no_trace:
        env["VLLM_TYPESAFE_TRACE"] = str(args.trace)
    return env


def model_args(args: argparse.Namespace) -> list[str]:
    config = json.loads((HERE / "engine-args.json").read_text())
    config["diffusion_config"]["canvas_length"] = args.canvas_length
    config["max_model_len"] = args.max_model_len
    config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.text_only:
        config["language_model_only"] = True
    if getattr(args, "optimize_text_32", False):
        config["max_num_seqs"] = 32
        config["max_num_batched_tokens"] = 34816
        config["compilation_config"].update(
            cudagraph_capture_sizes=[
                64,
                128,
                256,
                512,
                1024,
                1056,
                2048,
                2112,
                4096,
                4224,
                8192,
                8448,
                16384,
                16896,
                17408,
                32768,
                33792,
                34816,
            ],
            max_cudagraph_capture_size=34816,
        )
    if args.enforce_eager:
        config.pop("compilation_config")
        config["enforce_eager"] = True
    argv = [
        "vllm",
        "serve",
        str(args.checkpoint),
        "--served-model-name",
        "diffusion-gemma",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--disable-uvicorn-access-log",
        "--disable-log-stats",
        "--enable-prompt-tokens-details",
    ]
    for name, value in config.items():
        argv.append("--" + name.replace("_", "-"))
        if not isinstance(value, bool):
            argv.append(
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
        elif not value:
            raise ValueError(f"Unsupported false CLI option: {name}")
    return argv


def main() -> None:
    args = parse_args()
    argv = model_args(args)
    if args.run:
        from h200_profile import install

        install()
        import torch

        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("This runtime profile requires an H200/SM90 GPU")
        sys.argv = argv
        runpy.run_module(
            "examples.features.diffusion_typesafe.inprocess", run_name="__main__"
        )
        return
    env = environment(args)
    record = {
        "source": str(SOURCE),
        "checkpoint": str(args.checkpoint),
        "gpu": args.gpu,
        "output_dir": str(args.output_dir),
        "trace": None if args.no_trace else str(args.trace),
        "argv": argv,
        "profile": (
            "H200 BF16 text only, TP1, batch32, fused text norms"
            if args.optimize_text_32
            else (
                "H200 BF16 text only, TP1, native8"
                if args.text_only
                else "H200 BF16 vision, TP1, native8, exact RoPE/RMSNorm"
            )
        ),
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    for name in (
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--gpu",
        str(args.gpu),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-dir",
        str(args.output_dir),
        "--max-model-len",
        str(args.max_model_len),
        "--canvas-length",
        str(args.canvas_length),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--run",
    ]
    if args.enforce_eager:
        child_args.append("--enforce-eager")
    if args.text_only:
        child_args.append("--text-only")
    if args.optimize_text_32:
        child_args.append("--optimize-text-32")
    if args.no_trace:
        child_args.append("--no-trace")
    else:
        child_args.extend(["--trace", str(args.trace)])
    os.chdir(SOURCE)
    os.execve(sys.executable, child_args, env)


if __name__ == "__main__":
    main()
