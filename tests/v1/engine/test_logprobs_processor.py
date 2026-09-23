# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor.

These tests exercise the truncation invariant that the MRV2 sampler relies
on: when the sampler returns a row wider than a request's own
`num_logprobs + 1` (because another request in the batch needed a wider
row), the trailing positions are populated with sentinel values
(`token_id=0`, `logprob=-inf`). LogprobsProcessor must read only the first
`num_logprobs + 1` entries so those sentinels never reach the user.
"""

import numpy as np
import pytest

from vllm.logprobs import create_sample_logprobs
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists


def _make_processor(num_logprobs: int, *, flat_logprobs: bool) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=flat_logprobs),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_drops_trailing_sentinel_columns(flat_logprobs: bool):
    """A request that asked for 3 custom token logprobs but ended up in a
    batch padded to width 5 must not surface the trailing -inf entries."""
    processor = _make_processor(num_logprobs=3, flat_logprobs=flat_logprobs)

    sampled = 42
    # Layout: [sampled, custom_1, custom_2, custom_3, SENTINEL, SENTINEL]
    # Use float32-exact values so cumulative_logprob compares cleanly.
    token_ids = np.array([[sampled, 100, 200, 300, 0, 0]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    assert len(processor.logprobs) == 1
    pos = processor.logprobs[0]
    # Exactly sampled + 3 requested tokens; trailing sentinels dropped.
    assert set(pos.keys()) == {sampled, 100, 200, 300}
    assert 0 not in pos
    assert all(np.isfinite(lp.logprob) for lp in pos.values())
    # cumulative_logprob comes from the sampled token's logprob only.
    assert processor.cumulative_logprob == -0.5


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_accepts_exactly_sized_row(flat_logprobs: bool):
    """When the row is exactly num_logprobs+1, no truncation needed."""
    processor = _make_processor(num_logprobs=2, flat_logprobs=flat_logprobs)

    token_ids = np.array([[7, 11, 13]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.5, -2.5]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    pos = processor.logprobs[0]
    assert set(pos.keys()) == {7, 11, 13}


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_sparse_answer_rows_preserve_zero_padded_cumulative_logprob(
    flat_logprobs: bool,
):
    """Dropping neutral fixed rows preserves the read's cumulative score."""
    sparse = _make_processor(num_logprobs=1, flat_logprobs=flat_logprobs)
    expanded = _make_processor(num_logprobs=1, flat_logprobs=flat_logprobs)
    token_ids = np.array([[10, 11], [20, 21]], dtype=np.int32)
    answer_logprobs = np.array([[-0.25, -1.0], [-0.75, -2.0]], dtype=np.float32)
    answer_ranks = np.array([1, 1], dtype=np.int32)
    sparse._update_sample_logprobs(
        LogprobsLists(token_ids, answer_logprobs, answer_ranks)
    )

    expanded_token_ids = np.array(
        [[9, 9], [10, 11], [8, 8], [20, 21], [7, 7]], dtype=np.int32
    )
    expanded_logprobs = np.zeros((5, 2), dtype=np.float32)
    expanded_logprobs[[1, 3]] = answer_logprobs
    expanded._update_sample_logprobs(
        LogprobsLists(
            expanded_token_ids,
            expanded_logprobs,
            np.ones(5, dtype=np.int32),
        )
    )

    assert sparse.cumulative_logprob == expanded.cumulative_logprob == -1.0
    assert len(sparse.logprobs) == 2
    assert len(expanded.logprobs) == 5
