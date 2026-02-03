# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and (
            "properties" in data or "patternProperties" in data
        ):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by guidance/llguidance."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # patternProperties is not supported by llguidance
        if "patternProperties" in obj:
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, self.vocab_size
        )
        self.ll_executor = llguidance.LLExecutor()

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def fill_bitmasks_batch(
        self,
        requests_data: list[tuple[StructuredOutputGrammar, int, bool, list[int]]],
        bitmask: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> None:
        """Fill bitmasks using native Rust parallelism.

        Automatically uses the optimized draft token API when spec_tokens is
        non-empty.

        Args:
            requests_data: List of (grammar, start_index, apply_bitmask,
                           spec_tokens) tuples
            bitmask: The bitmask tensor to fill
            full_mask: Scalar tensor with value -1 for non-constrained positions
        """
        # Separate requests into non-spec and spec groups
        non_spec_matchers = []
        spec_matchers = []

        for grammar, start_index, apply_bitmask, spec_tokens in requests_data:
            num_positions = len(spec_tokens) + 1

            if not apply_bitmask or grammar.is_terminated():
                for i in range(num_positions):
                    bitmask[start_index + i].fill_(full_mask)
            else:
                assert isinstance(grammar, GuidanceGrammar)
                if spec_tokens:
                    spec_matchers.append((grammar.ll_matcher, start_index, spec_tokens))
                else:
                    non_spec_matchers.append((grammar.ll_matcher, start_index))

        # Call appropriate parallel API based on what we have
        if non_spec_matchers:
            llguidance_torch.fill_next_token_bitmask_par(
                self.ll_executor, non_spec_matchers, bitmask
            )
        if spec_matchers:
            llguidance_torch.fill_next_token_bitmask_par_with_draft_tokens(
                self.ll_executor, spec_matchers, bitmask
            )

    def destroy(self):
        pass

    def accept_tokens_batch(
        self,
        requests: list[tuple[StructuredOutputGrammar, list[int]]],
    ) -> list[bool]:
        """Accept tokens for multiple grammars in parallel using Rust parallelism.

        For single-token requests, uses parallel consume_token_par().
        For multi-token requests (speculative decoding), falls back to serial
        processing until llguidance adds native multi-token batch support.

        Args:
            requests: List of (grammar, token_ids) tuples. token_ids can be
                      a single token or multiple tokens (for speculative decoding).

        Returns:
            List[bool]: Success/failure for each grammar (in order).
        """
        if not requests:
            return []

        # Separate single-token and multi-token requests
        single_token_requests: list[tuple] = []
        multi_token_requests: list[tuple[GuidanceGrammar, list[int]]] = []
        request_order: list[tuple[str, int]] = []  # Track original order

        for i, (grammar, token_ids) in enumerate(requests):
            assert isinstance(grammar, GuidanceGrammar)
            if len(token_ids) == 1:
                single_token_requests.append((grammar.ll_matcher, token_ids[0]))
                request_order.append(("single", len(single_token_requests) - 1))
            else:
                multi_token_requests.append((grammar, token_ids))
                request_order.append(("multi", len(multi_token_requests) - 1))

        # Process single-token requests in parallel
        single_results: list[bool] = []
        if single_token_requests:
            single_results = llguidance_torch.consume_token_par(
                self.ll_executor, single_token_requests
            )

        # Process multi-token requests by stepping through token positions
        # Instead of N serial accept_tokens calls, do max_len parallel batch calls
        multi_results: list[bool] = [True] * len(multi_token_requests)
        if multi_token_requests:
            max_tokens = max(len(tokens) for _, tokens in multi_token_requests)

            for pos in range(max_tokens):
                # Collect (matcher, token) for requests that have a token at this
                # position and haven't failed yet
                batch: list[tuple] = []
                batch_to_multi_idx: list[int] = []

                for multi_idx, (grammar, tokens) in enumerate(multi_token_requests):
                    if pos < len(tokens) and multi_results[multi_idx]:
                        batch.append((grammar.ll_matcher, tokens[pos]))
                        batch_to_multi_idx.append(multi_idx)

                if batch:
                    pos_results = llguidance_torch.consume_token_par(
                        self.ll_executor, batch
                    )
                    for i, success in enumerate(pos_results):
                        if not success:
                            multi_results[batch_to_multi_idx[i]] = False

        # Reassemble results in original order
        results: list[bool] = []
        for req_type, idx in request_order:
            if req_type == "single":
                results.append(single_results[idx])
            else:
                results.append(multi_results[idx])

        return results


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """
        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(
                        f"Trigger {begin} not found in triggers {triggers}"
                    )
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    )
                )
            if not tags:
                raise ValueError("No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    # if structured output is not enabled, there is nothing to validate
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
