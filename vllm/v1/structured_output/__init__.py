# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    GuidanceGrammar,
    get_ll_executor,
)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        # `_grammar_bitmask` is the CURRENT ring slot, reassigned from
        # `_grammar_bitmask_ring` at the top of every `grammar_bitmask()` call.
        # The ring rotates so the buffer backing a returned (zero-copy) numpy
        # view is not overwritten by a later step's fills while a worker is
        # still reading it. `allocate_token_bitmask` is pinned host memory
        # (llguidance 1.7.0 & 1.7.5), so the returned view is itself a pinned
        # H2D source — no per-call `pin_memory()` (driver contention) and no
        # extra host copy. Depth must exceed the in-flight step count (async
        # scheduling / pipeline parallel).
        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self._grammar_bitmask_ring: list[torch.Tensor] | None = None
        self._pinned_ring_idx = 0
        _async_sched = vllm_config.scheduler_config.async_scheduling
        _pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self._pinned_ring_size = (
            _pp_size if _pp_size > 1 else (2 if _async_sched else 1)
        ) + 1

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        # Parallel-fill threshold: the non-spec `grammar_bitmask` parallel
        # path (via `LLExecutor.unsafe_compute_mask_ptr`) is gated on
        # `len(structured_output_request_ids) > threshold`. Set to 2 (any
        # batch > 1) — the Rust `len == 1` short-circuit in
        # `unsafe_compute_mask_ptr` handles the single-matcher case without
        # rayon overhead.
        self.fill_bitmask_parallel_threshold = 2
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask_ring is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a ring of bitmask buffers (one bitmask per token to
            # check: one per speculative position + one bonus/non-spec token).
            # `_grammar_bitmask` is rotated across these slots each step so a
            # returned zero-copy view survives until its slot is reused.
            self._grammar_bitmask_ring = [
                self.backend.allocate_token_bitmask(
                    max_batch_size * (1 + max_num_spec_tokens)
                )
                for _ in range(self._pinned_ring_size)
            ]
        self._grammar_bitmask = self._grammar_bitmask_ring[self._pinned_ring_idx]
        self._pinned_ring_idx = (self._pinned_ring_idx + 1) % self._pinned_ring_size

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for non-spec,
        # large-batch-size cases. `GuidanceGrammar` reqs go through a single
        # batched `LLExecutor.unsafe_compute_mask_ptr` call (fans out via
        # rayon par_iter with GIL released), mirroring the spec-decode
        # path's `_with_draft_token` call below. Non-`GuidanceGrammar`
        # backends don't expose a `_par` API, so they stay on the
        # threadpool path. Permissive-fill cases (`apply_bitmask=False` /
        # grammar terminated) skip the matcher entirely via a single-row
        # torch slice fill.
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch: list[tuple[StructuredOutputGrammar, int, bool]] = []
            # (matcher, row_index) accumulator for the batched _par call.
            batched_interpreters_ns: list[tuple[Any, int]] = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                # Force-block the Future before reading the property. The
                # property's 100µs poll-timeout in `_check_grammar_completion`
                # returns None on miss, which silently corrupts the bitmask
                # batch. By the time we land here the scheduler has already
                # gated this request past `is_grammar_ready` before promoting
                # to RUNNING, so the Future is ~µs from done and the blocking
                # unwrap closes the race window cheaply.
                if isinstance(structured_output_request._grammar, Future):
                    structured_output_request._grammar = (
                        structured_output_request._grammar.result()
                    )
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)

                if isinstance(grammar, GuidanceGrammar):
                    if not apply_bitmask or grammar.is_terminated():
                        # Permissive fill — no matcher work needed.
                        self._grammar_bitmask[cumulative_index].fill_(self._full_mask)
                    else:
                        # Accumulate for the batched `_par` dispatch after
                        # the loop. Each entry occupies a single bitmask row
                        # (non-spec path: n_rows == 1).
                        batched_interpreters_ns.append(
                            (grammar.ll_matcher, cumulative_index)
                        )
                else:
                    # Non-`GuidanceGrammar` backend (no `_par` API). Stay on
                    # the threadpool path verbatim.
                    batch.append((grammar, cumulative_index, apply_bitmask))
                    if len(batch) == self.fill_bitmask_parallel_batch_size:
                        promises.append(self._async_submit_fill_bitmask(batch))
                        batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Single batched `_par` dispatch across all `GuidanceGrammar`
            # reqs: one Rust-side parallel call (GIL released).
            if batched_interpreters_ns:
                row0_ns = self._grammar_bitmask[0]
                row_bytes_ns = row0_ns.numel() * row0_ns.element_size()
                get_ll_executor().unsafe_compute_mask_ptr(
                    batched_interpreters_ns,
                    self._grammar_bitmask.data_ptr(),
                    row_bytes_ns,
                    self._grammar_bitmask.shape[0],
                )

            # Wait for any non-`GuidanceGrammar` threadpool batches to finish.
            for promise in promises:
                promise.result()
        else:
            # Serial / spec-decode path. For each request, write
            # `1 + len(req_tokens)` bitmask rows: row 0 is the
            # sample/bonus-position mask; rows 1..N are after consuming each
            # successive draft. Spec-decode draft positions are filled via
            # the llguidance spec-decode API
            # (`unsafe_compute_mask_ptr_with_draft_token`) which advances
            # and rolls back internally on the Rust side. This avoids the
            # previous pattern of `accept_tokens([draft])` per token: that
            # pattern routed forbidden drafts through `consume_tokens`,
            # which could blow the lexer cost cap on complex tool grammars
            # and leave the matcher in `Stop: ParserTooComplex`. Other
            # backends retain the original advance+rollback path.
            #
            # `GuidanceGrammar` reqs with valid drafts are accumulated for a
            # single batched `LLExecutor.unsafe_compute_mask_ptr_with_draft_token`
            # call after the loop — fans out via rayon par_iter with the GIL
            # released, replacing N per-req calls with one batched dispatch.
            batched_interpreters: list[tuple[Any, int, list[int]]] = []
            # (matcher, row_index) accumulator for the no-draft / all-sentinel
            # sample-row cases below. Dispatched after the loop via the
            # batched non-draft `unsafe_compute_mask_ptr` (mirrors the non-spec
            # path 313-324), replacing the serial per-req `fill_next_token_bitmask`.
            sample_row_interpreters: list[tuple[Any, int]] = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                # Same race-window close as in the parallel branch above —
                # block on the Future rather than tolerating the property's
                # 100µs poll-timeout None return.
                if isinstance(structured_output_request._grammar, Future):
                    structured_output_request._grammar = (
                        structured_output_request._grammar.result()
                    )
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                n_rows = 1 + len(req_tokens)

                if not apply_bitmask or grammar.is_terminated():
                    # Fill rows permissive; nothing to advance through.
                    self._grammar_bitmask[
                        cumulative_index : cumulative_index + n_rows
                    ].fill_(self._full_mask)
                    cumulative_index += n_rows
                    continue

                if not isinstance(grammar, GuidanceGrammar):
                    # Backend doesn't support the batched `_par` API. Fall
                    # back to the advance+rollback path on the live matcher.
                    state_advancements = 0
                    for token in itertools.chain(req_tokens, (-1,)):
                        self._fill_bitmasks(
                            ((grammar, cumulative_index, apply_bitmask),)
                        )
                        if token == -1:
                            apply_bitmask = False
                        if apply_bitmask and not grammar.is_terminated():
                            accepted = grammar.accept_tokens(req_id, [token])
                            assert accepted, (
                                token,
                                req_id,
                                scheduled_spec_decode_tokens,
                            )
                            state_advancements += 1
                        cumulative_index += 1
                    if state_advancements > 0:
                        grammar.rollback(state_advancements)
                    continue

                # Pre-fill all rows permissive. The spec-decode API
                # OVERWRITES rows it computes; rows where it breaks early
                # (forbidden draft) stay permissive, letting the engine
                # fall back to natural sampling at those positions instead
                # of asserting on `accept_tokens`.
                self._grammar_bitmask[
                    cumulative_index : cumulative_index + n_rows
                ].fill_(self._full_mask)

                if not req_tokens:
                    # No drafts. Accumulate the single sample-position row for
                    # the batched `_par` dispatch after the loop (row already
                    # pre-filled permissive; the API overwrites it in place).
                    sample_row_interpreters.append(
                        (grammar.ll_matcher, cumulative_index)
                    )
                    cumulative_index += n_rows
                    continue

                # vLLM's draft-filter pipeline replaces drafts that fail
                # grammar validation with sentinel `-1` and pads spec slots
                # to a fixed length. The spec-decode API takes `Vec<u32>`,
                # so passing `-1` raises `OverflowError`. Truncate at first
                # sentinel — padded rows already got permissive masks.
                valid_drafts: list[int] = []
                for tok in req_tokens:
                    if tok < 0:
                        break
                    valid_drafts.append(tok)

                if valid_drafts:
                    # Accumulate for the batched call after the loop. Rows
                    # already pre-filled permissive; the API overwrites
                    # computed rows in place.
                    batched_interpreters.append(
                        (grammar.ll_matcher, cumulative_index, valid_drafts)
                    )
                else:
                    # All drafts are sentinels — accumulate the single
                    # sample-position row for the batched `_par` dispatch.
                    sample_row_interpreters.append(
                        (grammar.ll_matcher, cumulative_index)
                    )

                cumulative_index += n_rows

            # Single batched call across all accumulated `GuidanceGrammar`
            # reqs. Fans out via rayon par_iter with GIL released.
            if batched_interpreters:
                row0 = self._grammar_bitmask[0]
                row_bytes = row0.numel() * row0.element_size()
                get_ll_executor().unsafe_compute_mask_ptr_with_draft_token(
                    batched_interpreters,
                    self._grammar_bitmask.data_ptr(),
                    row_bytes,
                    self._grammar_bitmask.shape[0],
                )

            # Single batched non-draft dispatch across all accumulated no-draft
            # / all-sentinel sample rows: one Rust-side parallel call (GIL
            # released).
            if sample_row_interpreters:
                row0 = self._grammar_bitmask[0]
                get_ll_executor().unsafe_compute_mask_ptr(
                    sample_row_interpreters,
                    self._grammar_bitmask.data_ptr(),
                    row0.numel() * row0.element_size(),
                    self._grammar_bitmask.shape[0],
                )

        # Return a zero-copy numpy view of the current (pinned) ring slot. The
        # slot was rotated at the top of this call, so the next step writes a
        # different buffer and this view stays valid until its slot is reused
        # `_pinned_ring_size` steps later — by which point the worker H2D that
        # reads it has long completed.
        n = min(cumulative_index, self._grammar_bitmask.shape[0])
        return self._grammar_bitmask[:n].numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False
        # Unenforced-jump fail-open: a diverged request has grammar disabled
        # for the remainder of generation -> never advance its matcher.
        if (
            request.structured_output_request is not None
            and request.structured_output_request.disabled
        ):
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                return True

        return False

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
