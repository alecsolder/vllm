# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the retained H200 attention and MoE settings in every worker."""

import importlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3]
_installed = False
_moe_override = None


def install() -> None:
    global _installed, _moe_override
    if _installed:
        return
    text_only = os.environ.get("VLLM_TYPESAFE_TEXT_ONLY", "0")
    if text_only not in {"0", "1"}:
        raise RuntimeError("VLLM_TYPESAFE_TEXT_ONLY must be 0 or 1")
    expected = {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_BATCH_INVARIANT_FAST_LINEAR": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DIFFUSION_FUSE_READ_PREFILL": "1",
        "VLLM_DIFFUSION_MM_PREFIX_MODE": "off" if text_only == "1" else "hf_both",
        "VLLM_DIFFUSION_FUSE_VISION_RMSNORM": "1",
        "VLLM_DIFFUSION_FUSE_VISION_ROPE": "1",
        "VLLM_DIFFUSION_READ_FAST_ADMISSION": "1",
        "VLLM_DIFFUSION_ATTENTION_STAGES2": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"The H200 profile requires {name}={value}")
    if Path(os.environ["VLLM_TYPESAFE_SOURCE_ROOT"]).resolve() != SOURCE:
        raise RuntimeError("The H200 profile must use its own source checkout")
    for name in (
        "vllm",
        "vllm.model_executor.models.diffusion_gemma",
        "vllm.model_executor.models.diffusion_gemma_vision_rope",
        "examples.features.diffusion_typesafe.server",
    ):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(SOURCE):
            raise RuntimeError(f"Imported {name} outside {SOURCE}")
    from vllm.model_executor.layers.fused_moe import override_config

    moe_config = json.loads((HERE / "moe-config.json").read_text())
    if os.environ.get("VLLM_TYPESAFE_OPTIMIZE_TEXT_32") == "1":
        if text_only != "1" or os.environ.get("VLLM_DIFFUSION_FUSE_TEXT_NORM") != "1":
            raise RuntimeError("The batch32 profile requires text-only fused norms")
        moe_config["BLOCK_SIZE_M"] = 128
    _moe_override = override_config(moe_config)
    _moe_override.__enter__()
    from attention_stages2_v2 import install as install_attention

    install_attention()
    _installed = True
