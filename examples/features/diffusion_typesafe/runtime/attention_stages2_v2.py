# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained SM90 BF16 attention configurations for the H200 profile."""

import functools
import hashlib
import os
from pathlib import Path

SOURCE_SHA256 = "e4a0d089c319091b9abcb77c018f1613929effa2d844cc11fc7cb281eda2e2cd"


@functools.cache
def _is_sm90(device):
    import torch

    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def _matches(kwargs):
    import torch

    head = kwargs.get("HEAD_SIZE")
    if head not in (256, 512):
        return False
    expected = {
        "HEAD_SIZE_PADDED": head,
        "num_query_heads": 16,
        "num_queries_per_kv": 2 if head == 256 else 8,
        "BLOCK_M": 16,
        "BLOCK_Q": 8 if head == 256 else 2,
        "TILE_SIZE": 32,
        "SLIDING_WINDOW": 1024 if head == 256 else 0,
        "IS_3D": False,
        "KV_QUANT_MODE": 0,
        "USE_PER_SEQ_CAUSAL": True,
        "USE_ALIBI_SLOPES": False,
        "USE_QQ_BIAS": False,
        "USE_SOFTCAP": False,
        "USE_SINKS": False,
        "USE_MM_PREFIX": False,
        "USE_R_SWA": False,
        "USE_FP8": False,
        "Q_IS_FP8": False,
        "USE_TD": False,
        "USE_TD_QO": False,
        "CHUNK_LOOKBACK": -1,
        "CHUNK_SIZE": -1,
    }
    block_sizes = (16, 32) if head == 256 else (32,)
    if kwargs.get("BLOCK_SIZE") not in block_sizes:
        return False
    if any(kwargs.get(key) != value for key, value in expected.items()):
        return False
    if kwargs.get("num_warps", 4) != 4 or kwargs.get("num_stages", 3) != 3:
        return False
    if kwargs.get("CANONICAL_BLOCK_Q", 0):
        return False
    for name in ("query_ptr", "key_cache_ptr", "value_cache_ptr", "output_ptr"):
        tensor = kwargs.get(name)
        if tensor is None or tensor.dtype != torch.bfloat16:
            return False
    return _is_sm90(kwargs["query_ptr"].device)


class Stages2Launch:
    def __init__(self, original, optimize_text_32=False):
        self.original = original
        self.optimize_text_32 = optimize_text_32

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def run(*args, **kwargs):
            if not args and _matches(kwargs):
                kwargs["num_stages"] = 2
                if self.optimize_text_32:
                    kwargs["CANONICAL_BLOCK_Q"] = kwargs["BLOCK_Q"]
                    kwargs["BLOCK_M"] = 32
                    kwargs["BLOCK_Q"] = 32 // kwargs["num_queries_per_kv"]
                    adjusted_grid = (
                        kwargs["query_ptr"].shape[0] // kwargs["BLOCK_Q"]
                        + kwargs["num_seqs"],
                        kwargs["key_cache_ptr"].shape[2],
                    )
                    return self.original[adjusted_grid](*args, **kwargs)
            return launch(*args, **kwargs)

        return run


def install():
    if os.environ.get("VLLM_DIFFUSION_ATTENTION_STAGES2") != "1":
        return
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise RuntimeError("The stages=2 profile requires batch invariance")
    optimize_text_32 = os.environ.get("VLLM_TYPESAFE_OPTIMIZE_TEXT_32") == "1"
    if optimize_text_32 and os.environ.get("VLLM_TYPESAFE_TEXT_ONLY") != "1":
        raise RuntimeError("The optimized query tiles require the text-only profile")
    from vllm.v1.attention.ops import triton_unified_attention as attention

    source = Path(attention.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Attention source changed since stages=2 validation")
    if not isinstance(attention.kernel_unified_attention, Stages2Launch):
        attention.kernel_unified_attention = Stages2Launch(
            attention.kernel_unified_attention, optimize_text_32=optimize_text_32
        )
        configuration = "stages=2, query tile=32" if optimize_text_32 else "stages=2"
        print(
            f"H200 BF16 attention {configuration} override installed",
            flush=True,
        )
