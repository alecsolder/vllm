# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.structured_output.gemma_chat_grammar import (
    Gemma4SpecialTokens,
    generate_gemma4_chat_grammar,
)


def _tool(name: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters},
    }


def test_generates_tool_and_natural_language_branches() -> None:
    grammar = generate_gemma4_chat_grammar(
        tools=[
            _tool(
                "weather.lookup",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                },
            )
        ],
        allow_natural_language=True,
    )

    assert grammar.startswith("%llguidance {}\n")
    assert "start: tool_calls_seq | nl_then_calls | nl_refusal_body" in grammar
    assert r"call:weather\.lookup" in grammar
    assert "tc_weather_lookup_params" in grammar


def test_local_ref_expands_to_nested_object_rule() -> None:
    parameters = {
        "type": "object",
        "properties": {"primary_query": {"$ref": "#/$defs/Query"}},
        "required": ["primary_query"],
        "$defs": {
            "Query": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        },
    }

    grammar = generate_gemma4_chat_grammar([_tool("search", parameters)])

    assert "tc_search_p0_primary_query_params: /query:/" in grammar
    assert "/primary_query:/ open_value" not in grammar


def test_custom_special_tokens_are_used() -> None:
    tokens = Gemma4SpecialTokens(
        tool_call_start=900,
        tool_call_end=901,
        quote=902,
    )
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    grammar = generate_gemma4_chat_grammar([_tool("echo", parameters)], tokens=tokens)

    assert "<[900]>" in grammar
    assert "<[901]>" in grammar
    assert "<[902]>" in grammar


def test_requires_at_least_one_response_branch() -> None:
    with pytest.raises(ValueError, match="At least one"):
        generate_gemma4_chat_grammar()
