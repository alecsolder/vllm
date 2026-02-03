# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_cached_guidance import (
    CachedGuidanceBackend,
    _validate_grammar_impl,
    validate_cached_guidance_grammar,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "gpt2"


def test_cached_backend_compile_grammar_cache_hit():
    """Test that compiling the same grammar twice uses the cache."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="cached_guidance"),
    )

    # Clear the class-level cache before test
    CachedGuidanceBackend._matcher_templates.clear()

    backend = CachedGuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    schema = '{"type": "object"}'

    # First compilation - cache miss
    grammar1 = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    assert len(CachedGuidanceBackend._matcher_templates) == 1

    # Second compilation - cache hit (should not add to cache)
    grammar2 = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    assert len(CachedGuidanceBackend._matcher_templates) == 1

    # Both grammars should be functional
    assert not grammar1.is_terminated()
    assert not grammar2.is_terminated()


def test_cached_backend_compile_grammar_independence():
    """Test that cached matchers are independent copies."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="cached_guidance"),
    )

    # Clear the class-level cache before test
    CachedGuidanceBackend._matcher_templates.clear()

    backend = CachedGuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    schema = '{"type": "object"}'

    # Get two grammars from the same schema
    grammar1 = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    grammar2 = backend.compile_grammar(StructuredOutputOptions.JSON, schema)

    # Advance grammar1 to a different state
    prompt = tokenizer.encode('{"a": "b"}')
    for token in prompt:
        grammar1.accept_tokens("", [token])
    grammar1.accept_tokens("", [tokenizer.eos_token_id])

    # grammar1 should be terminated, but grammar2 should not be affected
    assert grammar1.is_terminated()
    assert not grammar2.is_terminated()


def test_validate_grammar_cached():
    """Test that validate_cached_guidance_grammar uses lru_cache."""
    # Clear the lru_cache before test
    _validate_grammar_impl.cache_clear()

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
        ),
    )

    # First validation - cache miss
    validate_cached_guidance_grammar(sampling_params)
    cache_info_1 = _validate_grammar_impl.cache_info()
    assert cache_info_1.misses == 1
    assert cache_info_1.hits == 0

    # Second validation - cache hit
    validate_cached_guidance_grammar(sampling_params)
    cache_info_2 = _validate_grammar_impl.cache_info()
    assert cache_info_2.misses == 1
    assert cache_info_2.hits == 1


def test_cached_backend_different_schemas():
    """Test that different schemas get separate cache entries."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="cached_guidance"),
    )

    # Clear the class-level cache before test
    CachedGuidanceBackend._matcher_templates.clear()

    backend = CachedGuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    schema1 = '{"type": "object"}'
    schema2 = '{"type": "string"}'
    schema3 = '{"type": "number"}'

    backend.compile_grammar(StructuredOutputOptions.JSON, schema1)
    assert len(CachedGuidanceBackend._matcher_templates) == 1

    backend.compile_grammar(StructuredOutputOptions.JSON, schema2)
    assert len(CachedGuidanceBackend._matcher_templates) == 2

    backend.compile_grammar(StructuredOutputOptions.JSON, schema3)
    assert len(CachedGuidanceBackend._matcher_templates) == 3

    # Reusing schema1 should not add new entry
    backend.compile_grammar(StructuredOutputOptions.JSON, schema1)
    assert len(CachedGuidanceBackend._matcher_templates) == 3
