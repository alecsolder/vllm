# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GrammarInputBuffers


def apply_grammar_bitmask(
    logits: torch.Tensor,
    req_ids: list[str],
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
    input_buffers: GrammarInputBuffers,
    scheduled_spec_decode_tokens: dict[str, list[int]] | None = None,
) -> None:
    """Apply grammar bitmask to logits using a Triton kernel.

    Handles both regular decoding and speculative decoding. For speculative
    decoding, the bitmask and logits have multiple consecutive entries per
    request.

    Args:
        logits: Output logits tensor [num_logits, vocab_size]
        req_ids: Request IDs in batch order (one per base logit position)
        grammar_req_ids: Request IDs that have grammar constraints
        grammar_bitmask: Bitmask array with consecutive entries per request
        input_buffers: Pre-allocated GPU buffers
        scheduled_spec_decode_tokens: Optional dict mapping req_id -> spec tokens
    """
    input_buffers.grammar_bitmask.np[: grammar_bitmask.shape[0]] = grammar_bitmask
    input_buffers.grammar_bitmask.copy_to_gpu(grammar_bitmask.shape[0])

    num_logits = logits.shape[0]

    if scheduled_spec_decode_tokens:
        # Speculative decoding: compute mapping accounting for multiple
        # logit/bitmask positions per request
        mapping = _compute_spec_decode_mapping(
            req_ids, grammar_req_ids, scheduled_spec_decode_tokens
        )
    else:
        # Simple case: 1:1 mapping between requests and logit positions
        grammar_req_id_to_idx = {req_id: i for i, req_id in enumerate(grammar_req_ids)}
        mapping = [grammar_req_id_to_idx.get(req_id, -1) for req_id in req_ids]

    input_buffers.bitmask_indices.np[:num_logits] = mapping
    input_buffers.bitmask_indices.copy_to_gpu(num_logits)

    vocab_size = logits.shape[-1]
    BLOCK_SIZE = 8192
    grid = (num_logits, triton.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        input_buffers.bitmask_indices.gpu[:num_logits],
        input_buffers.grammar_bitmask.gpu,
        input_buffers.grammar_bitmask.gpu.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def _compute_spec_decode_mapping(
    req_ids: list[str],
    grammar_req_ids: list[str],
    scheduled_spec_decode_tokens: dict[str, list[int]],
) -> list[int]:
    """Compute logit -> bitmask index mapping for speculative decoding.

    In speculative decoding:
    - Logits are laid out as: [req0_pos0, req0_pos1, ..., req1_pos0, ...]
    - Bitmask is laid out as: [grammar_req0_pos0, grammar_req0_pos1, ...]

    Returns a list where mapping[logit_idx] = bitmask_idx (or -1 if no bitmask).
    """
    # Build bitmask base index for each grammar request
    # (cumulative index accounting for num_positions per request)
    grammar_req_to_bitmask_base: dict[str, int] = {}
    cumulative_bitmask_idx = 0
    for req_id in grammar_req_ids:
        grammar_req_to_bitmask_base[req_id] = cumulative_bitmask_idx
        num_spec_tokens = len(scheduled_spec_decode_tokens.get(req_id, []))
        cumulative_bitmask_idx += 1 + num_spec_tokens

    # Build mapping for each logit position
    mapping: list[int] = []
    for req_id in req_ids:
        num_spec_tokens = len(scheduled_spec_decode_tokens.get(req_id, []))
        num_positions = 1 + num_spec_tokens

        if req_id in grammar_req_to_bitmask_base:
            bitmask_base = grammar_req_to_bitmask_base[req_id]
            for i in range(num_positions):
                mapping.append(bitmask_base + i)
        else:
            # No grammar constraint for this request
            for _ in range(num_positions):
                mapping.append(-1)

    return mapping


class StructuredOutputsWorker:
    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
