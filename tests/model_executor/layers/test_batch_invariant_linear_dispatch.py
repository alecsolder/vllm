# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.model_executor.layers import linear
from vllm.model_executor.layers import vocab_parallel_embedding as embedding


@pytest.mark.parametrize("fast_linear", [False, True])
def test_unquantized_linear_batch_invariant_dispatch(monkeypatch, fast_linear):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT_FAST_LINEAR", fast_linear)
    monkeypatch.setattr(linear.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(linear.current_platform, "is_xpu", lambda: False)

    method = object.__new__(linear.UnquantizedLinearMethod)
    method._gemm_impl = MagicMock(return_value="regular")
    invariant = MagicMock(return_value="invariant")
    monkeypatch.setattr(linear, "linear_batch_invariant", invariant)
    layer = SimpleNamespace(weight=object())

    result = method.apply(layer, object(), object())

    assert result == ("regular" if fast_linear else "invariant")
    assert method._gemm_impl.call_count == int(fast_linear)
    assert invariant.call_count == int(not fast_linear)


@pytest.mark.parametrize("fast_linear", [False, True])
def test_unquantized_lm_head_batch_invariant_dispatch(monkeypatch, fast_linear):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT_FAST_LINEAR", fast_linear)
    monkeypatch.setattr(embedding.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(embedding.current_platform, "is_xpu", lambda: False)

    regular = MagicMock(return_value="regular")
    invariant = MagicMock(return_value="invariant")
    monkeypatch.setattr(embedding, "dispatch_unquantized_gemm", lambda: regular)
    monkeypatch.setattr(embedding, "linear_batch_invariant", invariant)
    layer = SimpleNamespace(weight=object())

    result = embedding.UnquantizedEmbeddingMethod().apply(layer, object(), object())

    assert result == ("regular" if fast_linear else "invariant")
    assert regular.call_count == int(fast_linear)
    assert invariant.call_count == int(not fast_linear)
