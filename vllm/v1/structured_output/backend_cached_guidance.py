# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cached LLGuidance backend with simplified caching using lru_cache and ClassVar.

This module provides a cached version of the GuidanceBackend that:
1. Caches grammar validation using @lru_cache
2. Caches LLMatcher templates using a class-level dict
3. Returns deep_copy() of cached matchers for each request

When validation and compilation happen in the same process, the matcher
created during validation is reused during compilation (cache hit).
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    GuidanceGrammar,
    serialize_guidance_grammar,
)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance

else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")


@lru_cache(maxsize=64)
def _validate_grammar_impl(tp: StructuredOutputOptions, grm: str) -> None:
    """Validate grammar syntax (cached by raw key, not serialized).

    Uses (tp, grm) as cache key to skip serialization on cache hit.
    Raises ValueError on invalid grammar.
    """
    serialized_grammar = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(serialized_grammar, None)
    if err:
        raise ValueError(f"Grammar error: {err}")


def validate_cached_guidance_grammar(sampling_params: SamplingParams) -> None:
    """Validate grammar, skipping serialization for previously validated grammars."""
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    _validate_grammar_impl(tp, grm)  # raises on invalid, skips on cache hit


@dataclass
class CachedGuidanceBackend(GuidanceBackend):
    """GuidanceBackend with grammar caching.

    Uses a class-level cache to share LLMatcher templates across all instances.
    Returns deep_copy() of cached matchers for independence.
    """

    # Class-level cache: (request_type, grammar_spec) -> template matcher
    # Keyed on raw inputs to skip serialization on cache hit
    _matcher_templates: ClassVar[
        dict[tuple[StructuredOutputOptions, str], llguidance.LLMatcher]
    ] = {}

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        cache_key = (request_type, grammar_spec)

        if cache_key in self._matcher_templates:
            # Cache hit: skip serialization, return deep copy
            ll_matcher = self._matcher_templates[cache_key].deep_copy()
        else:
            # Cache miss: serialize and create matcher
            serialized_grammar = serialize_guidance_grammar(
                request_type,
                grammar_spec,
                self.disable_any_whitespace,
                self.disable_additional_properties,
            )
            ll_matcher = llguidance.LLMatcher(
                self.ll_tokenizer,
                serialized_grammar,
                log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
            )
            # Cache a template copy
            self._matcher_templates[cache_key] = ll_matcher.deep_copy()

        grammar = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )
        grammar.check_error()
        return grammar
