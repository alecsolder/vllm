# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for fused-read multimodal-range expansion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaModelState


class Staging:
    def __init__(self, size: int, dtype: np.dtype):
        self.np = np.zeros(size, dtype=dtype)

    def copy_to_uva(self) -> torch.Tensor:
        return torch.from_numpy(self.np)


def make_state():
    state = object.__new__(DiffusionGemmaModelState)
    state.max_num_reqs = 4
    state._split_query_staging = Staging(9, np.int32)
    state._split_logits_staging = Staging(4, np.int32)
    state._split_read_staging = Staging(4, np.bool_)
    state._split_query = torch.zeros(9, dtype=torch.int32)
    state._split_seq_lens = torch.zeros(8, dtype=torch.int32)
    state._split_causal = torch.zeros(8, dtype=torch.bool)
    state._split_block_tables = {}
    state._split_mm_prefix_range_buf = torch.full((8, 2, 2), 99, dtype=torch.int32)
    state._mm_prefix_range_copy_pending = True
    state._mm_prefix_range_copy_event = Mock()
    state.diffusion_states = SimpleNamespace(read_requests={0: object()})
    return state


def make_batch():
    return SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=1,
        query_start_loc_np=np.array([0, 8, 16], dtype=np.int32),
        cu_num_logits_np=np.array([0, 3, 5], dtype=np.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int64),
        seq_lens=torch.tensor([8, 16, 0, 0], dtype=torch.int32),
    )


def make_metadata(ranges: torch.Tensor, query_ranges=None):
    shared = SimpleNamespace(
        block_table=torch.arange(12, dtype=torch.int32).reshape(4, 3),
        query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
        seq_lens=torch.tensor([8, 16, 0, 0], dtype=torch.int32),
        causal=torch.tensor([True, True, False, False]),
        max_query_len=8,
        mm_prefix_range={0: [(2, 4)], 1: []},
        mm_prefix_range_tensor=ranges,
    )
    if query_ranges is not None:
        shared.mm_prefix_query_range_tensor = query_ranges
    return {"layer.0": shared, "layer.1": shared}, shared


def test_split_duplicates_ranges_for_both_pseudo_requests_and_reuses_buffer():
    state = make_state()
    batch = make_batch()
    causal = torch.tensor([True, True, False, False])
    original = torch.zeros(4, 2, 2, dtype=torch.int32)
    original[0, 0] = torch.tensor([2, 4])
    metadata, shared = make_metadata(original)

    state._split_read_attention(metadata, batch, causal, num_reqs=4)

    split = shared.mm_prefix_range_tensor
    pointer = split.data_ptr()
    assert split.tolist() == [
        [[2, 4], [0, 0]],
        [[2, 4], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    assert shared.mm_prefix_range == {
        0: [(2, 4)],
        1: [(2, 4)],
        2: [],
        3: [],
    }
    assert shared.query_start_loc.tolist() == [0, 5, 8, 14, 16, 16, 16, 16, 16]
    assert shared.seq_lens.tolist() == [5, 8, 14, 16, 0, 0, 0, 0]
    assert shared.causal.tolist() == [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert shared.block_table.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [3, 4, 5],
        [3, 4, 5],
        [6, 7, 8],
        [6, 7, 8],
        [9, 10, 11],
        [9, 10, 11],
    ]

    cleared = torch.zeros(4, 2, 2, dtype=torch.int32)
    metadata, shared = make_metadata(cleared)
    state._split_read_attention(metadata, batch, causal, num_reqs=4)
    assert shared.mm_prefix_range_tensor.data_ptr() == pointer
    assert shared.mm_prefix_range_tensor.count_nonzero().item() == 0
    assert state._mm_prefix_range_copy_event.record.call_count == 2


def test_split_rejects_flash_attention_query_range_metadata():
    state = make_state()
    batch = make_batch()
    ranges = torch.zeros(4, 2, 2, dtype=torch.int32)
    metadata, _ = make_metadata(
        ranges, query_ranges=torch.zeros(4, 2, dtype=torch.int32)
    )

    with pytest.raises(ValueError, match="requires Triton attention"):
        state._split_read_attention(
            metadata,
            batch,
            torch.tensor([True, True, False, False]),
            num_reqs=4,
        )
