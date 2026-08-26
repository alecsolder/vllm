# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for the Gemma4 tool-call grammar prototype.

The string-generation tests run without model assets. Matcher integration tests
use ``GEMMA4_TOKENIZER_PATH`` when a local Gemma4 tokenizer is available.
"""

from __future__ import annotations

import os

import pytest

from vllm.v1.structured_output.gemma_chat_grammar import (
    Gemma4SpecialTokens,
    generate_gemma4_chat_grammar,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]


def test_generates_gemma4_native_tool_envelope():
    grammar = generate_gemma4_chat_grammar(_TOOLS)

    assert grammar.startswith("%llguidance {}\n")
    assert "tc_lookup_weather_body" in grammar
    assert "<[48]>" in grammar  # <|tool_call>
    assert "<[49]>" in grammar  # <tool_call|>
    assert "<[52]> <[^52]>* <[52]>" in grammar  # Gemma4 string quoting
    assert "/days:/ /-?(?:0|[1-9][0-9]*)/" in grammar


def test_generates_nested_schema_and_parallel_call_rules():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "book",
                "parameters": {
                    "type": "object",
                    "$defs": {
                        "Passenger": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        }
                    },
                    "properties": {
                        "passengers": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Passenger"},
                        }
                    },
                    "required": ["passengers"],
                },
            },
        }
    ]

    grammar = generate_gemma4_chat_grammar(tools)

    assert "tool_calls_seq: ( tc_book_body )+" in grammar
    assert "tc_book_p0_passengers_item_params" in grammar
    assert "/\\[/" in grammar and "/\\]/" in grammar


def test_optional_response_modes_are_composed():
    grammar = generate_gemma4_chat_grammar(
        _TOOLS,
        allow_natural_language=True,
        allow_thinking=True,
    )

    assert "start: thinking_block?" in grammar
    assert "nl_then_calls" in grammar
    assert "nl_refusal_body" in grammar


def test_custom_special_tokens_and_invalid_inputs():
    custom = Gemma4SpecialTokens(tool_call_start=900, tool_call_end=901, quote=902)
    grammar = generate_gemma4_chat_grammar(_TOOLS, tokens=custom)
    assert "<[900]>" in grammar
    assert "<[901]>" in grammar
    assert "<[902]> <[^902]>* <[902]>" in grammar

    with pytest.raises(ValueError, match="At least one"):
        generate_gemma4_chat_grammar()

    colliding = [
        {"function": {"name": "a-b", "parameters": {}}},
        {"function": {"name": "a_b", "parameters": {}}},
    ]
    with pytest.raises(ValueError, match="collides"):
        generate_gemma4_chat_grammar(colliding)


@pytest.fixture(scope="module")
def matcher_runtime():
    tokenizer_path = os.environ.get("GEMMA4_TOKENIZER_PATH")
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        pytest.skip("Set GEMMA4_TOKENIZER_PATH to run matcher integration tests")

    llguidance = pytest.importorskip("llguidance")
    llguidance_hf = pytest.importorskip("llguidance.hf")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    return llguidance, llguidance_hf.from_tokenizer(tokenizer), tokenizer


def _matcher(grammar: str, llguidance, ll_tokenizer):
    serialized = llguidance.LLMatcher.grammar_from_lark(grammar)
    matcher = llguidance.LLMatcher(ll_tokenizer, serialized)
    assert not matcher.get_error()
    return matcher


def test_matcher_accepts_parallel_calls(matcher_runtime):
    llguidance, ll_tokenizer, tokenizer = matcher_runtime
    grammar = generate_gemma4_chat_grammar(_TOOLS)
    matcher = _matcher(grammar, llguidance, ll_tokenizer)
    text = (
        '<|tool_call>call:lookup_weather{city:<|"|>Paris<|"|>,days:2}'
        "<tool_call|>"
        '<|tool_call>call:lookup_weather{city:<|"|>Rome<|"|>}'
        "<tool_call|><eos>"
    )

    assert matcher.consume_tokens(tokenizer.encode(text, add_special_tokens=False))
    assert matcher.is_accepting() or matcher.is_stopped()


def test_mid_string_boundary_has_no_forced_token(matcher_runtime):
    """A string body remains open to either more text or its closing quote."""
    llguidance, ll_tokenizer, tokenizer = matcher_runtime
    grammar = generate_gemma4_chat_grammar(_TOOLS)
    matcher = _matcher(grammar, llguidance, ll_tokenizer)
    prefix = '<|tool_call>call:lookup_weather{city:<|"|>P'

    assert matcher.consume_tokens(tokenizer.encode(prefix, add_special_tokens=False))
    assert list(matcher.compute_ff_tokens()) == []
