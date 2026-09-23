# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Propagate explicit H200 profile settings into spawned Python workers."""

import os
import traceback

if os.environ.get("VLLM_TYPESAFE_H200_PROFILE") == "1":
    try:
        from h200_profile import install

        install()
    except Exception:
        traceback.print_exc()
        os._exit(1)
