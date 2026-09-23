# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU coverage for the opt-in diffusion image-attention metadata experiment."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import diffusion_gemma
from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaModelState


@pytest.fixture
def mixed_batch(monkeypatch):
    state = object.__new__(DiffusionGemmaModelState)
    state.max_model_len = 8192
    state._mm_prefix_mode = "hf_both"
    state._mm_prefix_trace_file = None
    state._full_attn_layer_idxs = {1}
    state._mm_prefix_range_cpu = torch.zeros(4, 2, 2, dtype=torch.int32)
    state._mm_prefix_range_np = state._mm_prefix_range_cpu.numpy()
    state._mm_prefix_range_buf = torch.zeros(4, 2, 2, dtype=torch.int32)
    state._mm_prefix_range_copy_event = Mock()
    state._mm_prefix_range_copy_pending = False
    state._causal_buf = torch.zeros(4, dtype=torch.bool)
    state._fuse_read_prefill = False
    state.diffusion_states = SimpleNamespace(
        is_encoder_phase=torch.tensor([True, False, True])
    )
    state.supports_mm_inputs = True
    state.model_config = SimpleNamespace(get_sliding_window=lambda: 1024)
    image = SimpleNamespace(
        modality="image",
        mm_position=SimpleNamespace(extract_embeds_range=lambda: [(2, 4)]),
    )
    state.encoder_cache = SimpleNamespace(
        mm_features={"image-prefill": [image], "image-denoise": [image]}
    )
    batch = SimpleNamespace(
        req_ids=["image-prefill", "text-prefill", "image-denoise"],
        num_reqs=3,
        num_reqs_after_padding=4,
        num_tokens=24,
        num_tokens_after_padding=32,
        idx_mapping=torch.tensor([2, 0, 1]),
        query_start_loc_np=np.array([0, 8, 16, 24], dtype=np.int32),
        query_start_loc=torch.tensor([0, 8, 16, 24], dtype=torch.int32),
        num_scheduled_tokens=np.array([8, 8, 8], dtype=np.int32),
        seq_lens=torch.tensor([8, 8, 16], dtype=torch.int32),
        is_prefilling_np=np.array([True, True, False]),
    )
    shared = SimpleNamespace(
        mm_prefix_range={0: [(2, 4)], 1: [], 2: []},
        mm_prefix_range_tensor=torch.tensor([[[2, 4]]]),
        mm_prefix_query_range_tensor=torch.tensor([[2, 4]]),
        causal=state._causal_buf[:3],
    )
    metadata = {
        "model.layers.0.self_attn.attn": shared,
        "model.layers.1.self_attn.attn": shared,
    }
    builder = Mock(return_value=metadata)
    monkeypatch.setattr(diffusion_gemma, "build_attn_metadata", builder)
    return state, batch, shared, builder


def prepare(state, batch, mode=CUDAGraphMode.NONE, for_capture=False):
    return state.prepare_attn(
        input_batch=batch,
        cudagraph_mode=mode,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
        for_capture=for_capture,
    )


def test_image_ranges_only_apply_to_prefill_and_preserve_mixed_causal_flags(
    mixed_batch,
):
    state, batch, shared, builder = mixed_batch
    metadata = prepare(state, batch)

    assert builder.call_args.kwargs["mm_req_doc_ranges"] is None
    assert builder.call_args.kwargs["causal"].tolist() == [True, True, False]
    assert all(item is shared for item in metadata.values())


def test_sliding_only_copies_shared_metadata_without_changing_causal_tensor(
    mixed_batch,
):
    state, batch, shared, _ = mixed_batch
    state._mm_prefix_mode = "sliding_only"
    metadata = prepare(state, batch)

    sliding, full = metadata.values()
    assert sliding is shared
    assert sliding.mm_prefix_range == {0: [(2, 4)], 1: [], 2: []}
    assert sliding.mm_prefix_range_tensor is not None
    assert sliding.mm_prefix_query_range_tensor is not None
    assert full is not shared
    assert full.mm_prefix_range is None
    assert full.mm_prefix_range_tensor is None
    assert full.mm_prefix_query_range_tensor is None
    assert full.causal is shared.causal


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_default_keeps_existing_metadata_and_graph_padding(mixed_batch, mode):
    state, batch, shared, builder = mixed_batch
    state._mm_prefix_mode = "off"
    metadata = prepare(state, batch, mode)

    assert builder.call_args.kwargs["mm_req_doc_ranges"] is None
    assert all(item is shared for item in metadata.values())
    expected = [True, True, False]
    if mode == CUDAGraphMode.FULL:
        expected.append(False)
    assert builder.call_args.kwargs["causal"].tolist() == expected


@pytest.mark.parametrize("text_only", [False, True])
def test_denoising_or_text_only_batches_do_not_add_mm_metadata(mixed_batch, text_only):
    state, batch, shared, builder = mixed_batch
    if text_only:
        state.encoder_cache.mm_features.clear()
    else:
        batch.is_prefilling_np[:] = False
    metadata = prepare(state, batch)

    assert builder.call_args.kwargs["mm_req_doc_ranges"] is None
    assert all(item is shared for item in metadata.values())
    assert shared.mm_prefix_range_tensor.count_nonzero().item() == 0


def test_experimental_mode_reuses_persistent_range_tensor_for_cuda_graph(
    mixed_batch,
):
    state, batch, _, _ = mixed_batch
    metadata = prepare(state, batch, CUDAGraphMode.FULL, for_capture=True)
    captured = next(iter(metadata.values())).mm_prefix_range_tensor

    assert captured.data_ptr() == state._mm_prefix_range_buf.data_ptr()
    assert captured.tolist() == [
        [[2, 4], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]

    batch.is_prefilling_np[:] = False
    metadata = prepare(state, batch, CUDAGraphMode.FULL)
    replay = next(iter(metadata.values())).mm_prefix_range_tensor
    assert replay.data_ptr() == captured.data_ptr()
    assert replay.count_nonzero().item() == 0
    state._mm_prefix_range_copy_event.synchronize.assert_called_once_with()
    assert state._mm_prefix_range_copy_event.record.call_count == 2


def test_persistent_range_tensor_clears_rows_when_graph_padding_shrinks_and_grows(
    mixed_batch,
):
    state, batch, _, _ = mixed_batch
    first = next(iter(prepare(state, batch, CUDAGraphMode.FULL).values()))
    pointer = first.mm_prefix_range_tensor.data_ptr()

    batch.req_ids = ["text-prefill"]
    batch.num_reqs = 1
    batch.num_reqs_after_padding = 2
    batch.is_prefilling_np = np.array([True])
    shrunk = next(iter(prepare(state, batch, CUDAGraphMode.FULL).values()))
    assert shrunk.mm_prefix_range_tensor.data_ptr() == pointer
    assert shrunk.mm_prefix_range_tensor.count_nonzero().item() == 0

    batch.req_ids = ["text-prefill", "image-prefill", "text-prefill"]
    batch.num_reqs = 3
    batch.num_reqs_after_padding = 4
    batch.is_prefilling_np = np.array([True, True, True])
    grown = next(iter(prepare(state, batch, CUDAGraphMode.FULL).values()))
    assert grown.mm_prefix_range_tensor.data_ptr() == pointer
    assert grown.mm_prefix_range_tensor.tolist() == [
        [[0, 0], [0, 0]],
        [[2, 4], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]


def test_multimodal_range_count_must_fit_configured_capacity(mixed_batch):
    state, batch, _, _ = mixed_batch
    state._mm_prefix_range_cpu = torch.zeros(4, 1, 2, dtype=torch.int32)
    state._mm_prefix_range_np = state._mm_prefix_range_cpu.numpy()
    state._mm_prefix_range_buf = torch.zeros(4, 1, 2, dtype=torch.int32)
    state.encoder_cache.mm_features["image-prefill"][0].mm_position = SimpleNamespace(
        extract_embeds_range=lambda: [(2, 4), (6, 8)]
    )

    with pytest.raises(ValueError, match="configured capacity is 1"):
        prepare(state, batch)


def test_long_image_range_is_preserved_for_per_layer_window_clamping(mixed_batch):
    state, batch, _, _ = mixed_batch
    state.encoder_cache.mm_features["image-prefill"][0].mm_position = SimpleNamespace(
        extract_embeds_range=lambda: [(2, 1200)]
    )

    metadata = prepare(state, batch)

    assert next(iter(metadata.values())).mm_prefix_range[0] == [(2, 1200)]
