# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the example's HTTP and trace-view contracts without model inference."""

import asyncio
import hashlib
import json
import math
from contextlib import asynccontextmanager
from io import BytesIO, StringIO
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pybase64 as base64
import pytest
import regex as re
import xgrammar as xgr
from fastapi import FastAPI
from PIL import Image
from rich.console import Console

from examples.features.diffusion_typesafe import inprocess, server
from examples.features.diffusion_typesafe.inprocess import (
    DiffusionTypeSafeEndpointPlugin,
    _InProcessChatClient,
)
from examples.features.diffusion_typesafe.server import (
    Choice,
    DecisionReader,
    ImageInput,
    Noul,
    SystemOneRequest,
    create_app,
    prepare_images,
)
from examples.features.diffusion_typesafe.tui import (
    DetailScroll,
    Trace,
    canvas_grid_shape,
    canvas_panel,
    input_keys,
    parent_context,
    render,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse

TEST_CANVAS_LENGTH = 2048
CODE_TOKEN_IDS = {
    **{chr(index): index for index in range(65, 91)},
    **{
        a + b: 107 + index + (2 if index >= 94 else 0)
        for index, (a, b) in enumerate(
            (a, b) for a in "ABCD" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        if index < 100
    },
}


def reasoned_completion(body, probabilities):
    result = completion(body, probabilities)
    for choice in result["choices"]:
        choice["token_ids"] = [
            int(row["token"].removeprefix("token_id:"))
            for row in choice["logprobs"]["content"]
        ]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason_mode,canvas_format", [(False, "json"), (True, "chat_single_user")]
)
async def test_reason_ensemble_null_preserves_default_requests_and_responses(
    reason_mode, canvas_format
):
    """Opting out leaves the ordinary body, noise and response contract unchanged."""
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=reasoned_completion(body, [0.25, 0.75]))

    body = {
        "state": "The item was returned.",
        "questions": {"q": {"type": "noul"}},
        "reason_mode": reason_mode,
        "canvas_format": canvas_format,
        "reason_tokens": 8,
        "reason_steps": 3,
        "samples": 2,
        "seed": 17,
        "return_diagnostics": True,
    }
    replies = []
    async with api_client(respond) as client:
        for controls in ({}, {"reason_ensemble": None}, {}):
            reply = await client.post("/v1/systemone", json={**body, **controls})
            assert reply.status_code == 200, reply.text
            replies.append(reply.json())
    assert bodies[0] == bodies[1] == bodies[2]
    assert replies[0] == replies[1] == replies[2]
    assert "reason_ensemble" not in replies[0]["diagnostics"]


@pytest.mark.parametrize(
    "budgets,samples",
    [
        ([{"tokens": 1, "steps": 1}], 32),
        ([{"tokens": 512, "steps": 64}], 1),
        ([{"tokens": 1, "steps": 1}] * 32, 1),
        ([{"tokens": 8, "steps": 3}, {"tokens": 12, "steps": 5}], 16),
    ],
)
def test_reason_ensemble_accepts_exact_budget_and_draw_boundaries(budgets, samples):
    request = SystemOneRequest.model_validate(
        {
            "state": "x",
            "questions": {"q": {"type": "noul"}},
            "reason_mode": True,
            "canvas_format": "chat_single_user",
            "samples": samples,
            "reason_ensemble": budgets,
        }
    )
    assert [item.model_dump() for item in request.reason_ensemble] == budgets
    assert request.samples * len(request.reason_ensemble) <= 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"reason_ensemble": []},
        {"reason_ensemble": [{"tokens": 1, "steps": 1}] * 33},
        {"reason_ensemble": {"tokens": 8, "steps": 3}},
        {"reason_ensemble": [{"tokens": 0, "steps": 3}]},
        {"reason_ensemble": [{"tokens": 513, "steps": 3}]},
        {"reason_ensemble": [{"tokens": True, "steps": 3}]},
        {"reason_ensemble": [{"tokens": 8, "steps": 0}]},
        {"reason_ensemble": [{"tokens": 8, "steps": 65}]},
        {"reason_ensemble": [{"tokens": 8, "steps": 3.0}]},
        {"reason_ensemble": [{"tokens": 8, "steps": 3, "unknown": 1}]},
        {"reason_mode": False},
        {"canvas_format": "chat"},
        {"canvas_format": "json"},
        {"adaptive_samples": True},
        {"samples": 17},
        {"reason_schedule_steps": 4},
    ],
)
async def test_reason_ensemble_rejects_invalid_controls_before_transport(overrides):
    def respond(request):
        pytest.fail("Invalid ensemble controls must not reach upstream")

    body = {
        "state": "x",
        "questions": {"q": {"type": "noul"}},
        "reason_mode": True,
        "canvas_format": "chat_single_user",
        "reason_ensemble": [{"tokens": 8, "steps": 3}, {"tokens": 12, "steps": 5}],
    }
    async with api_client(respond) as client:
        reply = await client.post("/v1/systemone", json={**body, **overrides})
    assert reply.status_code == 422, reply.text


@pytest.mark.asyncio
async def test_reason_ensemble_preserves_each_budget_local_seed_and_effective_trace(
    tmp_path, monkeypatch
):
    """Two per-budget draws equal two ordinary requests and one grouped round."""
    bodies, contexts = [], []
    original_context = DecisionReader.shared_context

    def context(reader, request):
        value = original_context(reader, request)
        contexts.append(value)
        return value

    monkeypatch.setattr(DecisionReader, "shared_context", context)

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=reasoned_completion(body, [0.25, 0.75]))

    budgets = [{"tokens": 8, "steps": 3}, {"tokens": 12, "steps": 5}]
    common = {
        "state": "x",
        "questions": {"q": {"type": "noul"}},
        "reason_mode": True,
        "canvas_format": "chat_single_user",
        "samples": 2,
        "seed": 17,
        "steps": 2,
        "reason_initial_answer_noise": "candidates",
        "reason_answer_noise": "vocab",
        "reason_schedule_steps": 5,
        "reason_entropy_bound": 0.01,
        "return_diagnostics": True,
    }
    trace = tmp_path / "ensemble.jsonl"
    async with api_client(respond, trace=trace) as client:
        for budget in budgets:
            reply = await client.post(
                "/v1/systemone",
                json={
                    **common,
                    "reason_tokens": budget["tokens"],
                    "reason_steps": budget["steps"],
                },
            )
            assert reply.status_code == 200, reply.text
        reply = await client.post(
            "/v1/systemone",
            json={
                **common,
                "reason_steps": 64,
                "reason_ensemble": budgets,
            },
        )
        assert reply.status_code == 200, reply.text
    assert (
        len(bodies) == len(contexts) == 3 and contexts[0] == contexts[1] == contexts[2]
    )
    assert (
        bodies[2]["diffusion_reads"]
        == bodies[0]["diffusion_reads"] + bodies[1]["diffusion_reads"]
    )
    assert bodies[2]["n"] == 4
    assert bodies[2]["max_tokens"] == max(
        bodies[0]["max_tokens"], bodies[1]["max_tokens"]
    )
    assert (
        bodies[0]["diffusion_reads"][0]["seed_canvas"]
        != bodies[0]["diffusion_reads"][1]["seed_canvas"]
    )
    diagnostics = reply.json()["diagnostics"]
    assert diagnostics["reason_ensemble"] == budgets
    assert "reason_tokens" not in diagnostics and "reason_steps" not in diagnostics
    assert diagnostics["samples"] == 2 and diagnostics["questions"]["q"]["samples"] == 4
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    parent = [row for row in events if row["event"] == "parent"][-1]
    samples = [row for row in events if row["event"] == "sample"][-4:]
    assert [row["sample"] for row in samples] == [0, 1, 0, 1]
    assert [row["denoise_steps"] for row in samples] == [5, 5, 7, 7]
    assert [
        (row["reason_tokens"], row["reason_steps"]) for row in parent["children"]
    ] == [(8, 3), (8, 3), (12, 5), (12, 5)]
    assert reply.json()["usage"]["input_tokens"] == 4 * 40


@pytest.mark.asyncio
async def test_reason_ensemble_mixed_primitives_keep_id_order_and_semantic_means():
    """Question routing and arrival order cannot mix another budget or primitive."""
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        vectors = []
        for read in body["diffusion_reads"]:
            short = len(read["reasoning_positions"]) == 8
            vectors.append(
                ([0.2, 0.8] if short else [0.6, 0.4])
                if len(read["logprob_token_ids"]) == 2
                else ([0.8, 0.1, 0.1] if short else [0.1, 0.2, 0.7])
            )
        result = reasoned_completion(body, vectors)
        result["choices"].reverse()
        return httpx.Response(200, json=result)

    questions = {
        "noul": {"type": "noul"},
        "choice": {"type": "choice", "criteria": {"a": None, "b": None}},
        "score": {"type": "score", "criteria": ["none", "some", "all"]},
    }
    body = {
        "state": "x",
        "reason_mode": True,
        "canvas_format": "chat_single_user",
        "reason_ensemble": [{"tokens": 8, "steps": 3}, {"tokens": 12, "steps": 5}],
        "return_diagnostics": True,
    }
    layouts = [
        questions,
        {
            "renamed_score": questions["score"],
            "opaque☃": questions["noul"],
            "other": questions["choice"],
        },
    ]
    replies = []
    async with api_client(respond) as client:
        for layout in layouts:
            reply = await client.post(
                "/v1/systemone", json={**body, "questions": layout}
            )
            assert reply.status_code == 200, reply.text
            replies.append(reply.json())
    assert bodies[0]["messages"] == bodies[1]["messages"]
    answers = replies[0]["answers"]
    assert answers["noul"]["noul"] == pytest.approx(0.4)
    assert answers["choice"]["probabilities"] == pytest.approx({"a": 0.4, "b": 0.6})
    assert answers["score"]["probabilities"] == pytest.approx(
        {"0": 0.45, "1": 0.15, "2": 0.4}
    )
    assert answers["score"]["score"] == pytest.approx(0.95)
    for old, new, first, second in (
        ("noul", "opaque☃", 0, 1),
        ("choice", "other", 1, 2),
        ("score", "renamed_score", 2, 0),
    ):
        assert answers[old] == replies[1]["answers"][new]
        assert (
            bodies[0]["diffusion_reads"][first * 2 : first * 2 + 2]
            == bodies[1]["diffusion_reads"][second * 2 : second * 2 + 2]
        )
    assert len(bodies) == 2 and all(body["n"] == 6 for body in bodies)


@pytest.mark.asyncio
async def test_reason_ensemble_validates_all_budget_capacities_before_transport():
    """A later oversized budget rejects the entire batch before the first read."""
    async with decision_reader() as reader:
        template = reader.template("q", Noul(type="noul"), 8, "chat_single_user")
        exact_length = template.content_tokens + 2

    def respond(request):
        pytest.fail("A partly prepared ensemble must not send any upstream request")

    async with api_client(respond, canvas_length=exact_length) as client:
        reply = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "reason_mode": True,
                "canvas_format": "chat_single_user",
                "reason_ensemble": [
                    {"tokens": 8, "steps": 3},
                    {"tokens": 9, "steps": 3},
                ],
            },
        )
    assert reply.status_code == 422, reply.text
    assert "Nothing was truncated" in reply.json()["detail"]


@pytest.mark.parametrize("padding", [0, 4, 253])
@pytest.mark.parametrize("reasoning", [False, True])
def test_showcase_accepts_complete_scaffolds_with_optional_padding(padding, reasoning):
    """The physical canvas may be trimmed without changing the fixed scaffold."""
    pytest.importorskip("typesafe_sdk")
    from examples.features.diffusion_typesafe.evaluate import valid_canvas

    record = {
        "canvas_tokens": [10, 99, 106] + [0] * padding,
        "output_tokens": [10, 201, 106] + [0] * padding,
        "answer_position": 1,
        "question_canvas_tokens": 3,
        "candidate_token_ids": [201, 202],
    }
    if reasoning:
        record.update(
            {
                "canvas_tokens": [40, 99, 106] + [0] * padding,
                "output_tokens": [50, 201, 106] + [0] * padding,
                "reasoning_positions": [0],
                "reasoning_forbidden_token_ids": [106],
                "reasoning_vocab_size": 512,
            }
        )
    assert valid_canvas(record)


@pytest.mark.parametrize(
    "overrides",
    [
        {"canvas_tokens": [], "output_tokens": []},
        {"canvas_tokens": [10] * 257, "output_tokens": [10] * 257},
        {"output_tokens": [10, 201]},
        {"canvas_tokens": [10, 99], "output_tokens": [10, 201]},
        {"output_tokens": [11, 201, 106]},
        {"output_tokens": [10, 203, 106]},
        {"answer_position": 3},
    ],
)
def test_showcase_rejects_truncated_or_invalid_canvases(overrides):
    """Trimming padding must not hide missing scaffold tokens or illegal edits."""
    pytest.importorskip("typesafe_sdk")
    from examples.features.diffusion_typesafe.evaluate import valid_canvas

    record = {
        "canvas_tokens": [10, 99, 106],
        "output_tokens": [10, 201, 106],
        "answer_position": 1,
        "question_canvas_tokens": 3,
        "candidate_token_ids": [201, 202],
    }
    assert not valid_canvas({**record, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_tokens": [11, 50, 51, 201, 106]},
        {"output_tokens": [10, 106, 51, 201, 106]},
        {"reasoning_positions": [1, 5]},
        {"reasoning_positions": [1, 3]},
    ],
)
def test_showcase_rejects_reasoning_that_changes_forbidden_or_fixed_tokens(overrides):
    """A reasoning span does not permit edits elsewhere or invalid generated IDs."""
    pytest.importorskip("typesafe_sdk")
    from examples.features.diffusion_typesafe.evaluate import valid_canvas

    record = {
        "canvas_tokens": [10, 40, 41, 99, 106],
        "output_tokens": [10, 50, 51, 201, 106],
        "answer_position": 3,
        "question_canvas_tokens": 5,
        "candidate_token_ids": [201, 202],
        "reasoning_positions": [1, 2],
        "reasoning_forbidden_token_ids": [106],
        "reasoning_vocab_size": 512,
    }
    assert not valid_canvas({**record, **overrides})


def token_vocabulary():
    vocabulary = [f"unused{index}".encode() for index in range(512)]
    vocabulary[256:] = [bytes([index]) for index in range(256)]
    for token, spelling in {
        0: "<pad>",
        106: "<turn|>",
        201: "yes",
        202: "no",
        249: "<|turn>",
        250: "user",
        251: "\n",
        252: "\n\n",
        **{token: code for code, token in CODE_TOKEN_IDS.items()},
    }.items():
        vocabulary[token] = spelling.encode()
    return vocabulary


class Tokenizer:
    """Byte-level fixture with whole-token answer codes and real JSON grammar."""

    pad_token_id = 0
    all_special_ids = [106, 249]
    vocabulary = token_vocabulary()

    def __len__(self):
        return len(self.vocabulary)

    def encode(self, text, add_special_tokens=False):
        if text == "<turn|>":
            return [106]
        if text == "\n\n":
            return [252]
        if text.startswith("<|turn>user\n"):
            return [249, 250, 251] + self.encode(text[len("<|turn>user\n") :])
        matches = list(re.finditer(r'"answer"\s*:\s*"(yes|no|[A-Z]{1,2})"', text))
        if not matches:
            matches = list(re.finditer(r'0: "?(yes|no|[A-Z]{1,2}|[1-9])"?$', text))
        if not matches:
            return [256 + byte for byte in text.encode()]
        match = matches[-1]
        code = match.group(1)
        value = (
            256 + ord(code)
            if code.isdecimal()
            else {"yes": 201, "no": 202, **CODE_TOKEN_IDS}[code]
        )
        return (
            [256 + byte for byte in text[: match.start(1)].encode()]
            + [value]
            + [256 + byte for byte in text[match.end(1) :].encode()]
        )

    def decode(self, tokens):
        return b"".join(self.vocabulary[token] for token in tokens).decode(
            errors="replace"
        )


def grammar_compiler(overrides=None):
    vocabulary = token_vocabulary()
    for index, spelling in (overrides or {}).items():
        vocabulary[index] = spelling.encode()
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocabulary, xgr.VocabType.RAW, stop_token_ids=[106]),
        max_threads=1,
    )


@asynccontextmanager
async def decision_reader(
    compiler=None,
    canvas_length=TEST_CANVAS_LENGTH,
    tokenizer=None,
    trace=None,
    trim_canvas_padding=False,
    minimum_canvas_padding=0,
    full_canvas_types=frozenset(),
):
    reader = DecisionReader(
        tokenizer if tokenizer is not None else Tokenizer(),
        "http://upstream",
        "diffusion-gemma",
        canvas_length,
        trace=trace,
        grammar_compiler=compiler if compiler is not None else grammar_compiler(),
        trim_canvas_padding=trim_canvas_padding,
        minimum_canvas_padding=minimum_canvas_padding,
        full_canvas_types=full_canvas_types,
    )
    try:
        yield reader
    finally:
        await reader.close()


@asynccontextmanager
async def api_client(respond, **reader_kwargs):
    async with decision_reader(**reader_kwargs) as reader:
        await reader.client.aclose()
        reader.client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://upstream/"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(reader)),
            base_url="http://test",
        ) as client:
            yield client


def completion(body, probabilities):
    if not isinstance(probabilities[0], list):
        probabilities = [probabilities] * body["n"]
    choices = []
    for index, (read, values) in enumerate(zip(body["diffusion_reads"], probabilities)):
        candidates = read["logprob_token_ids"]
        canvas = read["seed_canvas"].copy()
        position = read["answer_positions"][0]
        winner = max(range(len(values)), key=values.__getitem__)
        canvas[position] = candidates[winner]
        top = [
            {"token": f"token_id:{token}", "logprob": math.log(probability)}
            for token, probability in zip(candidates, values)
        ]
        content = [
            {
                "token": f"token_id:{token}",
                "logprob": math.log(values[winner]) if i == position else -0.1,
                "top_logprobs": top if i == position else [],
            }
            for i, token in enumerate(canvas)
        ]
        choices.append(
            {
                "index": index,
                "finish_reason": "length",
                "logprobs": {"content": content},
            }
        )
    return {
        "id": "chatcmpl-test",
        "choices": choices,
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": sum(
                len(read["seed_canvas"]) for read in body["diffusion_reads"]
            ),
        },
    }


def compact_completion(body, probabilities):
    result = completion(body, probabilities)
    for choice, read in zip(result["choices"], body["diffusion_reads"]):
        position = read["answer_positions"][0]
        content = choice["logprobs"]["content"]
        choice["logprobs"]["content"] = [content[position]]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 26, 27, 100])
async def test_xgrammar_accepts_every_complete_fixed_slot_category(count):
    """All 100 codes occupy one legal slot and preserve their option metadata."""
    async with decision_reader() as reader:
        question = Choice(
            type="choice", criteria={f"label_{index}": None for index in range(count)}
        )
        template = reader.template("routing", question)
        all_codes = tuple(CODE_TOKEN_IDS)
        codes = all_codes[:count] if count <= 26 else all_codes[26 : 26 + count]
        assert template.candidate_ids == tuple(CODE_TOKEN_IDS[code] for code in codes)
        assert template.codes == codes
        document = json.loads(
            reader.tokenizer.decode(template.tokens[: template.content_tokens - 1])
        )
        assert document["options"] == {
            code: f"label_{index}" for index, code in enumerate(codes)
        }
        assert document["name"] == "decision"


@pytest.mark.asyncio
async def test_xgrammar_rejects_a_tokenizer_with_an_invalid_fixed_prefix():
    async with decision_reader(grammar_compiler({256 + ord("{"): "["})) as reader:
        with pytest.raises(
            ValueError, match="XGrammar rejected the fixed answer prefix"
        ):
            reader.template("q", Noul(type="noul"))


@pytest.mark.asyncio
async def test_xgrammar_rejects_prefix_valid_candidate_that_cannot_complete_suffix():
    """A next-token mask alone cannot validate a one-token answer scaffold."""
    compiler = grammar_compiler({201: "y"})
    matcher = xgr.GrammarMatcher(
        compiler.compile_grammar('root ::= "\\"yes\\"" | "\\"no\\""')
    )
    assert matcher.accept_token(256 + ord('"'))
    assert matcher.accept_token(201)
    assert not matcher.accept_token(256 + ord('"'))
    async with decision_reader(compiler) as reader:
        with pytest.raises(ValueError, match="answer candidate with the fixed suffix"):
            reader.template("q", Noul(type="noul"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["noul", "choice"])
async def test_full_json_context_must_encode_every_answer_as_one_shared_slot(kind):
    """A valid JSON answer that needs two tokens cannot become a one-slot read."""

    class SplitAnswerTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            tokens = super().encode(text, add_special_tokens=add_special_tokens)
            token = 202 if kind == "noul" else CODE_TOKEN_IDS["AA"]
            if token in tokens:
                index = tokens.index(token)
                tokens[index : index + 1] = [256 + ord("A"), 256 + ord("A")]
            return tokens

    async with decision_reader(tokenizer=SplitAnswerTokenizer()) as reader:
        with pytest.raises(ValueError, match="same single token position"):
            question = (
                Noul(type="noul")
                if kind == "noul"
                else Choice(type="choice", criteria={str(i): None for i in range(27)})
            )
            reader.template("q", question)


@pytest.mark.asyncio
async def test_hundred_choices_return_every_probability_and_the_final_option():
    """The HTTP contract preserves all labels, including the last two-letter code."""

    def respond(request):
        body = json.loads(request.content)
        read = body["diffusion_reads"][0]
        assert len(set(read["logprob_token_ids"])) == 100
        assert len(read["answer_positions"]) == 1
        return httpx.Response(
            200, json=compact_completion(body, [0.001] * 99 + [0.901])
        )

    async with api_client(respond) as client:
        question: dict[str, Any] = {
            "type": "choice",
            "criteria": {f"q{i}": None for i in range(100)},
        }
        response = await client.post(
            "/v1/systemone", json={"state": "x", "questions": {"q": question}}
        )
        assert response.status_code == 200, response.text
        answer = response.json()["answers"]["q"]
        assert answer["choice"] == "q99"
        assert list(answer["probabilities"]) == list(question["criteria"])
        assert sum(answer["probabilities"].values()) == pytest.approx(1)
        question["criteria"]["q100"] = None
        rejected = await client.post(
            "/v1/systemone", json={"state": "x", "questions": {"q": question}}
        )
        assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_compact_answer_rows_reconstruct_the_validated_canvas():
    def respond(request):
        body = json.loads(request.content)
        return httpx.Response(200, json=compact_completion(body, [0.25, 0.75]))

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["answers"]["q"]["noul"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_inprocess_plugin_exposes_system_one_contract():
    class Models:
        async def show_available_models(self):
            class Response:
                def model_dump(self):
                    return {
                        "object": "list",
                        "data": [{"id": "diffusion-gemma", "object": "model"}],
                    }

            return Response()

    class Reader:
        async def system_one(self, request, request_id):
            assert request.state == "x"
            assert request.system_prompt == "Follow the caller's policy."
            assert request.steps == 3
            assert request.self_conditioning is False
            assert request.self_conditioning_scope == "canvas"
            assert request.return_diagnostics is True
            assert request_id.startswith("ts-")
            assert inprocess._RAW_REQUEST.get() is not None
            assert inprocess._RAW_REQUEST.get().app.state.server_load_metrics == 1
            return {"model": "diffusion-gemma", "usage": {}, "answers": {}}

    app = FastAPI()

    @app.get("/v1/models")
    async def core_models():
        return {"object": "list", "data": []}

    DiffusionTypeSafeEndpointPlugin().attach_router(app)
    app.state.diffusion_typesafe_reader = Reader()
    app.state.openai_serving_models = Models()
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "system_prompt": "Follow the caller's policy.",
                "steps": 3,
                "self_conditioning": False,
                "self_conditioning_scope": "canvas",
                "return_diagnostics": True,
            },
        )
        models = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.headers["x-typesafe-request-id"].startswith("ts-")
    assert response.json()["model"] == "diffusion-gemma"
    assert models.json()["data"][0]["id"] == "diffusion-gemma"
    assert models.json()["models"][0]["name"] == "diffusion-gemma"
    assert app.state.server_load_metrics == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, secret",
    [
        (
            {
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "samples": "private-sample-value",
            },
            "private-sample-value",
        ),
        (
            {
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "images": [{"url": {"private": "private-image-url"}}],
            },
            "private-image-url",
        ),
        (
            {
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "system_prompt": {"private": "private-system-prompt"},
            },
            "private-system-prompt",
        ),
    ],
)
async def test_inprocess_plugin_sanitizes_validation_errors(body, secret):
    app = FastAPI()
    DiffusionTypeSafeEndpointPlugin().attach_router(app)
    app.state.enable_server_load_tracking = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/systemone", json=body)

    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()["detail"][0]["loc"][0] == "body"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, content_type",
    [
        (b'{"state":"private-invalid-json"', "application/json"),
        (
            b'{"state":"x","questions":{"q":{"type":"noul"}}}',
            "text/plain",
        ),
    ],
)
async def test_inprocess_plugin_rejects_non_json_requests(content, content_type):
    app = FastAPI()
    DiffusionTypeSafeEndpointPlugin().attach_router(app)
    app.state.enable_server_load_tracking = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/systemone",
            content=content,
            headers={"content-type": content_type},
        )

    assert response.status_code == 422
    assert "private-invalid-json" not in response.text
    assert response.json()["detail"][0]["loc"] == ["body"]


@pytest.mark.asyncio
async def test_inprocess_client_preserves_chat_completion_wire_contract():
    class Handler:
        async def create_chat_completion(self, request, raw_request=None):
            assert raw_request is None
            body = request.model_dump(exclude_none=True)
            assert all(read["steps"] == 3 for read in body["diffusion_reads"])
            assert all(
                read["self_conditioning"] is False for read in body["diffusion_reads"]
            )
            assert all(
                read["self_conditioning_scope"] == "canvas"
                for read in body["diffusion_reads"]
            )
            payload = compact_completion(body, [0.25, 0.75])
            payload["model"] = request.model
            payload["usage"]["total_tokens"] = sum(payload["usage"].values())
            for choice in payload["choices"]:
                choice["message"] = {"role": "assistant", "content": ""}
            return ChatCompletionResponse.model_validate(payload)

    async with decision_reader() as reader:
        await reader.client.aclose()
        reader.client = _InProcessChatClient(Handler())  # type: ignore[arg-type]
        result = await reader.system_one(
            SystemOneRequest(
                state="x",
                questions={"q": {"type": "noul"}},
                steps=3,
                self_conditioning=False,
                self_conditioning_scope="canvas",
            ),
            "ts-test",
        )

    assert result["answers"]["q"]["noul"] == pytest.approx(0.25)


@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive", [False, True])
async def test_inprocess_adaptive_rounds_do_not_reuse_callers_parent_id(
    tmp_path, adaptive
):
    """Outer request correlation survives distinct internal adaptive parents."""
    parent_ids: list[str] = []
    trace_path = tmp_path / "inprocess-parents.jsonl"

    class Handler:
        async def create_chat_completion(self, request, raw_request):
            assert raw_request.headers["x-client-trace"] == "outer-trace"
            assert raw_request.headers.get("x-request-id") == (
                None if adaptive else "caller-id"
            )
            parent_id = inprocess.OpenAIServingChat._base_request_id(
                raw_request, request.request_id
            )
            payload = compact_completion(request.model_dump(), [0.5, 0.5])
            payload["id"] = f"chatcmpl-{parent_id}"
            payload["model"] = request.model
            payload["usage"]["total_tokens"] = sum(payload["usage"].values())
            for choice in payload["choices"]:
                choice["message"] = {"role": "assistant", "content": ""}
            parent_ids.append(payload["id"])
            return ChatCompletionResponse.model_validate(payload)

    async with decision_reader(trace=trace_path) as reader:
        await reader.client.aclose()
        reader.client = _InProcessChatClient(Handler())  # type: ignore[arg-type]
        app = FastAPI()
        DiffusionTypeSafeEndpointPlugin().attach_router(app)
        app.state.diffusion_typesafe_reader = reader
        app.state.enable_server_load_tracking = False
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/systemone",
                headers={"x-request-id": "caller-id", "x-client-trace": "outer-trace"},
                json={
                    "state": "x",
                    "questions": {"q": {"type": "noul"}},
                    "adaptive_samples": adaptive,
                    "max_samples": 2,
                },
            )
    assert response.status_code == 200, response.text
    assert len(parent_ids) == (2 if adaptive else 1)
    assert len(set(parent_ids)) == len(parent_ids)
    if not adaptive:
        assert parent_ids == ["chatcmpl-caller-id"]
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert {event["request_id"] for event in events} == {
        response.headers["x-typesafe-request-id"]
    }
    if adaptive:
        assert events[-1]["parent_ids"] == parent_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("trim_padding", [False, True])
async def test_inprocess_plugin_initializes_from_online_renderer(
    monkeypatch, trim_padding
):
    monkeypatch.delenv("VLLM_TYPESAFE_TRACE", raising=False)
    monkeypatch.setenv("VLLM_TYPESAFE_TRIM_CANVAS_PADDING", str(int(trim_padding)))
    monkeypatch.setenv(
        "VLLM_TYPESAFE_MINIMUM_CANVAS_PADDING", "4" if trim_padding else "0"
    )
    monkeypatch.setenv(
        "VLLM_TYPESAFE_FULL_CANVAS_TYPES", "choice" if trim_padding else ""
    )
    captured: dict[str, Any] = {}

    class Client:
        async def aclose(self):
            pass

    class Reader:
        def __init__(self, tokenizer, upstream, model, **kwargs):
            captured.update(tokenizer=tokenizer, model=model, kwargs=kwargs)
            self.client = Client()

    tokenizer = Tokenizer()
    state = SimpleNamespace(
        openai_serving_chat=object(),
        online_renderer=SimpleNamespace(renderer=SimpleNamespace(tokenizer=tokenizer)),
        vllm_config=SimpleNamespace(
            diffusion_config=SimpleNamespace(canvas_length=256)
        ),
    )
    args = SimpleNamespace(
        served_model_name=["diffusion-gemma"],
        model="/checkpoint",
    )
    monkeypatch.setattr(inprocess, "DecisionReader", Reader)

    await DiffusionTypeSafeEndpointPlugin().init_state(object(), state, args)

    assert captured == {
        "tokenizer": tokenizer,
        "model": "diffusion-gemma",
        "kwargs": {
            "canvas_length": 256,
            "trace": None,
            "trim_canvas_padding": trim_padding,
            "minimum_canvas_padding": 4 if trim_padding else 0,
            "full_canvas_types": frozenset({"choice"}) if trim_padding else frozenset(),
        },
    }
    assert isinstance(state.diffusion_typesafe_reader.client, _InProcessChatClient)


@pytest.mark.asyncio
async def test_typed_answers_use_every_candidate_and_zero_based_scores():
    """A response preserves opaque labels/rubrics and the full distribution."""
    upstream_requests = []

    def respond(request):
        body = json.loads(request.content)
        upstream_requests.append(body)
        values = [
            [0.25, 0.75] if len(read["logprob_token_ids"]) == 2 else [0.1, 0.3, 0.6]
            for read in body["diffusion_reads"]
        ]
        return httpx.Response(200, json=completion(body, values))

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": {"ticket": ["example", None]},
                "cache_salt": "same-cache-namespace",
                "questions": {
                    "yes?": {"type": "noul", "instructions": "Is this relevant?"},
                    "opaque:name\n": {
                        "type": "choice",
                        "criteria": {"a": None, "b": None, "返金": None},
                    },
                    "severity": {
                        "type": "score",
                        "criteria": ["none", {"level": "some"}, "high"],
                    },
                },
            },
        )
        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["yes?"]["noul"] == pytest.approx(0.25)
        assert answers["opaque:name\n"]["choice"] == "返金"
        assert answers["opaque:name\n"]["probabilities"] == pytest.approx(
            {"a": 0.1, "b": 0.3, "返金": 0.6}
        )
        assert answers["severity"]["score"] == pytest.approx(1.5)
        assert answers["severity"]["legend"]["1"] == {"level": "some"}
        assert response.json()["usage"] == {
            "input_tokens": 120,
            "output_tokens": 3 * TEST_CANVAS_LENGTH,
        }
        assert response.headers["x-typesafe-request-id"].startswith("ts-")
        assert len(upstream_requests) == 1
        body = upstream_requests[0]
        assert body["n"] == 3
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert body["cache_salt"] == "same-cache-namespace"
        assert "logprob_token_ids" not in body and "vllm_xargs" not in body
        assert all(
            len(read["answer_positions"]) == 1 for read in body["diffusion_reads"]
        )
        assert all(read["answer_only_logprobs"] for read in body["diffusion_reads"])
        assert all(read["steps"] == 1 for read in body["diffusion_reads"])
        assert all(read["self_conditioning"] for read in body["diffusion_reads"])
        assert all(
            "self_conditioning_scope" not in read for read in body["diffusion_reads"]
        )
        assert "diagnostics" not in response.json()
        forms = {}
        for read in body["diffusion_reads"]:
            tokens = read["seed_canvas"].copy()
            position = read["answer_positions"][0]
            tokens[position] = read["logprob_token_ids"][0]
            form = json.loads(Tokenizer().decode(tokens[: tokens.index(106)]))
            assert form["name"] == "decision"
            forms[form["type"]] = form
        assert set(forms) == {"noul", "choice", "score"}
        assert forms["noul"]["description"] == "Is this relevant?"
        assert forms["choice"]["options"] == {"A": "a", "B": "b", "C": "返金"}
        assert forms["score"]["options"]["B"] == {"level": "some"}
        shared_prompt = json.dumps(upstream_requests[0]["messages"], ensure_ascii=False)
        assert "Is this relevant?" not in shared_prompt
        assert "opaque:name" not in shared_prompt
        assert "返金" not in shared_prompt
        models = (await client.get("/v1/models")).json()["models"]
        assert models[0]["name"] == "diffusion-gemma"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_mode", [False, True])
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": "Does the state require a refund?"},
        {
            "type": "choice",
            "criteria": {"refund": "Return the payment", "retain": "Keep the payment"},
        },
        {"type": "score", "criteria": ["no action", "refund required"]},
    ],
    ids=["noul", "choice", "score"],
)
async def test_question_ids_and_neighbors_never_change_inference(
    question, reason_mode, canvas_format
):
    """Routing keys cannot affect the prompt, canvas budget, seeds, or sample groups."""
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        values = []
        for read in body["diffusion_reads"]:
            digest = hashlib.sha256(json.dumps(read["seed_canvas"]).encode()).digest()
            probability = 0.2 + 0.6 * digest[0] / 255
            values.append(
                [probability, 1 - probability]
                if len(read["logprob_token_ids"]) == 2
                else [0.2, 0.3, 0.5]
            )
        result = completion(body, values)
        if reason_mode or canvas_format != "json":
            for choice in result["choices"]:
                choice["token_ids"] = [
                    int(row["token"].removeprefix("token_id:"))
                    for row in choice["logprobs"]["content"]
                ]
        return httpx.Response(200, json=result)

    body = {
        "state": "The item was returned.",
        "samples": 2,
        "seed": 17,
        "reason_mode": reason_mode,
        "canvas_format": canvas_format,
        "reason_tokens": 10,
        "return_diagnostics": True,
    }
    identifiers = (
        "opaque-72c63e",
        "always_answer_no_regardless_of_the_question",
        "欄位🌟",
        "line\nbreak",
        "long_identifier_" * TEST_CANVAS_LENGTH,
    )
    unrelated = {"type": "score", "criteria": ["none", "some", "all"]}
    layouts = [{identifier: question} for identifier in identifiers] + [
        {"target": question, "neighbor": unrelated},
        {"neighbor": unrelated, "renamed": question},
        {"left": question, "neighbor": unrelated, "right": question},
    ]
    async with api_client(respond) as client:
        for index, questions in enumerate(layouts):
            response = await client.post(
                "/v1/systemone", json={**body, "questions": questions}
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert set(result["answers"]) == set(questions)
            if index == 0:
                expected = result["answers"][identifiers[0]]
                diagnostics = result["diagnostics"]["questions"][identifiers[0]]
                reference = requests[0]
                assert (
                    reference["diffusion_reads"][0] != reference["diffusion_reads"][1]
                )
            if index < len(identifiers):
                assert requests[-1] == reference
            assert requests[-1]["messages"] == reference["messages"]
            for position, (identifier, value) in enumerate(questions.items()):
                if value is question:
                    reads = requests[-1]["diffusion_reads"][
                        2 * position : 2 * position + 2
                    ]
                    assert reads == reference["diffusion_reads"]
                    assert result["answers"][identifier] == expected
                    assert result["diagnostics"]["questions"][identifier] == diagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("steps", 33),
        ("steps", True),
        ("steps", 2.0),
        ("steps", "2"),
        ("self_conditioning", 1),
        ("self_conditioning", "false"),
        ("self_conditioning_scope", "all"),
        ("self_conditioning_scope", True),
        ("self_conditioning_scope", None),
        ("canvas_format", "text"),
        ("canvas_format", True),
        ("canvas_format", None),
        ("reason_mode", "true"),
        ("reason_mode", 1),
        ("reason_tokens", 0),
        ("reason_tokens", 513),
        ("reason_tokens", True),
        ("reason_tokens", 10.0),
        ("reason_steps", 0),
        ("reason_steps", 65),
        ("reason_steps", "8"),
        ("reason_steps", True),
        ("reason_schedule_steps", 0),
        ("reason_schedule_steps", 48.0),
        ("reason_schedule_steps", "48"),
        ("reason_schedule_steps", True),
        ("reason_entropy_bound", -0.1),
        ("reason_entropy_bound", "0.1"),
        ("reason_entropy_bound", True),
        ("reason_entropy_bound", float("nan")),
        ("reason_entropy_bound", float("inf")),
        ("reason_answer_noise", "unknown"),
        ("reason_answer_noise", True),
        ("reason_initial_answer_noise", "unknown"),
        ("reason_initial_answer_noise", True),
        ("return_diagnostics", 0),
        ("return_diagnostics", "true"),
    ],
)
async def test_request_controls_reject_coercion_before_upstream(field, value):
    def respond(request):
        pytest.fail("Invalid request controls must not reach the model")

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            content=json.dumps(
                {"state": "x", "questions": {"q": {"type": "noul"}}, field: value}
            ),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_steps", [8, 64])
async def test_reason_schedule_cannot_end_before_requested_passes(reason_steps):
    def respond(request):
        pytest.fail("An invalid reasoning schedule must not reach the model")

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "reason_mode": True,
                "reason_steps": reason_steps,
                "reason_schedule_steps": reason_steps - 1,
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("self_conditioning", [False, True])
async def test_request_controls_leave_defaults_and_initial_noise_unchanged(
    tmp_path, self_conditioning
):
    """Per-call tuning cannot leak through the cached question template."""
    requests = []
    trace = tmp_path / "controls.jsonl"

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=compact_completion(body, [0.25, 0.75]))

    body = {"state": "x", "questions": {"q": {"type": "noul"}}}
    async with api_client(respond, trace=trace) as client:
        responses = [
            await client.post("/v1/systemone", json={**body, **controls})
            for controls in (
                {},
                {
                    "steps": 32,
                    "self_conditioning": self_conditioning,
                    "self_conditioning_scope": "canvas",
                    "return_diagnostics": True,
                },
                {},
            )
        ]
    assert all(response.status_code == 200 for response in responses)
    before, tuned, after = [response.json() for response in responses]
    assert before == after
    assert "diagnostics" not in before
    assert tuned["answers"] == before["answers"]
    assert tuned["diagnostics"] == {
        "samples": 1,
        "steps": 32,
        "self_conditioning": self_conditioning,
        "self_conditioning_scope": "canvas",
        "seed": 0,
        "questions": {
            "q": {
                "samples": 1,
                "standard_errors": {"yes": None, "no": None},
                "argmax_agreement": 1.0,
            }
        },
    }
    assert requests[0] == requests[2]
    tuned_read = requests[1]["diffusion_reads"][0]
    assert tuned_read == {
        **requests[0]["diffusion_reads"][0],
        "steps": 32,
        "self_conditioning": self_conditioning,
        "self_conditioning_scope": "canvas",
    }
    samples = [
        row
        for line in trace.read_text().splitlines()
        if (row := json.loads(line))["event"] == "sample"
    ]
    assert [row["denoise_steps"] for row in samples] == [1, 32, 1]
    assert [row["self_conditioning"] for row in samples] == [
        True,
        self_conditioning,
        True,
    ]
    assert [row["self_conditioning_scope"] for row in samples] == [
        "answer",
        "canvas",
        "answer",
    ]


def reason_completion(body, probabilities, compact=True):
    result = completion(body, probabilities)
    for index, (choice, read) in enumerate(
        zip(result["choices"], body["diffusion_reads"])
    ):
        content = choice["logprobs"]["content"]
        for position in read["reasoning_positions"]:
            content[position]["token"] = f"token_id:{256 + ord('a') + index}"
        choice["token_ids"] = [
            int(row["token"].removeprefix("token_id:")) for row in content
        ]
        if compact:
            choice["logprobs"]["content"] = [content[read["answer_positions"][0]]]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("answer_noise", ["candidates", "vocab"])
@pytest.mark.parametrize("canvas_format", ["json", "chat"])
@pytest.mark.parametrize("reason_steps", [4, 48, 64])
async def test_reason_mode_returns_generated_strings_per_draw_and_preserves_scaffold(
    tmp_path, compact, answer_noise, canvas_format, reason_steps
):
    """Final reasoning comes from generated token IDs, never the initial noise."""
    requests = []
    trace = tmp_path / "reason.jsonl"
    schedule_steps = max(48, reason_steps)

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["return_token_ids"] is True
        for read in body["diffusion_reads"]:
            assert len(read["reasoning_positions"]) == 10
            assert read["reasoning_steps"] == reason_steps
            assert read["reasoning_schedule_steps"] == schedule_steps
            assert read["reasoning_entropy_bound"] == 0
            assert read["steps"] == 3
            assert read["reasoning_positions"][-1] < read["answer_positions"][0]
            assert read.get("reasoning_answer_noise", "candidates") == answer_noise
            if answer_noise == "candidates":
                assert (
                    read["seed_canvas"][read["answer_positions"][0]]
                    in read["logprob_token_ids"]
                )
            forbidden = set(read["reasoning_forbidden_token_ids"])
            assert {0, 106} <= forbidden
            punctuation = {256 + ord('"'), 256 + ord("\\"), 256 + 10}
            assert (punctuation <= forbidden) == (canvas_format == "json")
            assert all(
                read["seed_canvas"][position] not in forbidden
                for position in read["reasoning_positions"]
            )
        return httpx.Response(200, json=reason_completion(body, [0.75, 0.25], compact))

    body = {
        "state": "x",
        "questions": {"first": {"type": "noul"}, "second": {"type": "noul"}},
        "samples": 2,
        "reason_mode": True,
        "canvas_format": canvas_format,
        "reason_tokens": 10,
        "reason_steps": reason_steps,
        "reason_schedule_steps": schedule_steps,
        "reason_entropy_bound": 0,
        "reason_answer_noise": answer_noise,
        "steps": 3,
        "return_diagnostics": True,
    }
    async with api_client(respond, trace=trace) as client:
        response = await client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["answers"]["first"]["reasoning"] == ["a" * 10, "b" * 10]
    assert answer["answers"]["second"]["reasoning"] == ["c" * 10, "d" * 10]
    assert answer["answers"]["first"]["noul"] == pytest.approx(0.75)
    assert answer["diagnostics"]["reason_tokens"] == 10
    assert answer["diagnostics"]["reason_steps"] == reason_steps
    assert answer["diagnostics"]["reason_schedule_steps"] == schedule_steps
    assert answer["diagnostics"]["reason_entropy_bound"] == 0
    assert answer["diagnostics"]["reason_answer_noise"] == answer_noise
    assert len(requests) == 1
    if answer_noise == "vocab":
        assert any(
            read["seed_canvas"][read["answer_positions"][0]]
            not in read["logprob_token_ids"]
            for read in requests[0]["diffusion_reads"]
        )
    samples = [
        json.loads(line)
        for line in trace.read_text().splitlines()
        if json.loads(line)["event"] == "sample"
    ]
    assert [sample["reasoning"] for sample in samples] == [
        letter * 10 for letter in "abcd"
    ]
    assert all(sample["denoise_steps"] == reason_steps + 3 for sample in samples)
    for sample in samples:
        assert sample["reason_schedule_steps"] == schedule_steps
        assert sample["reason_entropy_bound"] == 0
        positions = set(sample["reasoning_positions"]) | {sample["answer_position"]}
        assert all(
            a == b
            for index, (a, b) in enumerate(
                zip(sample["canvas_tokens"], sample["output_tokens"])
            )
            if index not in positions
        )
        assert [
            Tokenizer().decode([token]) for token in sample["canvas_tokens"]
        ] == sample["canvas_text"]


@pytest.mark.asyncio
async def test_reason_mode_does_not_change_legacy_templates_or_noise():
    """Reasoning budgets and template caching cannot leak into disabled requests."""
    async with decision_reader() as reader:
        base = SystemOneRequest(state="x", questions={"q": Noul(type="noul")})
        context = reader.shared_context(base)
        original = reader.prepare_read(base, "q", base.questions["q"], 0, context)
        reason = base.model_copy(update={"reason_mode": True, "reason_tokens": 10})
        reason_context = reader.shared_context(reason)
        first = reader.prepare_read(
            reason, "q", reason.questions["q"], 0, reason_context
        )
        repeated = reader.prepare_read(
            reason, "q", reason.questions["q"], 0, reason_context
        )
        second = reader.prepare_read(
            reason, "q", reason.questions["q"], 1, reason_context
        )
        after = reader.prepare_read(base, "q", base.questions["q"], 0, context)
        disabled = base.model_copy(
            update={
                "reason_tokens": 50,
                "reason_steps": 32,
                "reason_schedule_steps": 1,
                "reason_entropy_bound": 1.0,
                "reason_answer_noise": "vocab",
                "reason_initial_answer_noise": "candidates",
                "canvas_format": "json",
            }
        )
        assert original == after
        assert reader.shared_context(disabled) == context
        assert (
            reader.prepare_read(disabled, "q", disabled.questions["q"], 0, context)
            == original
        )
        assert first == repeated
        assert first.canvas != second.canvas
        assert first.template == second.template
        assert len(first.template.reasoning_positions) == 10
        assert reason_context.prompt_hash != context.prompt_hash
        assert "reasoning_positions" not in original.descriptor()
        document = json.loads(
            reader.tokenizer.decode(
                first.template.tokens[: first.template.content_tokens - 1]
            )
        )
        assert document["reasoning"] == "x" * 10


@pytest.mark.asyncio
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
async def test_reason_sampler_controls_preserve_prefix_and_each_draw_seed(
    canvas_format,
):
    """Changing the sampler must isolate its effect from prompt and initial noise."""
    async with decision_reader() as reader:
        request = SystemOneRequest(
            state="x",
            questions={"q": Noul(type="noul")},
            canvas_format=canvas_format,
            reason_mode=True,
            reason_tokens=10,
            reason_steps=4,
        )
        context = reader.shared_context(request)
        for sample_index in (0, 3):
            baseline = reader.prepare_read(
                request, "q", request.questions["q"], sample_index, context
            )
            for controls in (
                {"reason_schedule_steps": None, "reason_entropy_bound": None},
                {"reason_steps": 64},
                {"reason_schedule_steps": 4},
                {"reason_schedule_steps": 48, "reason_entropy_bound": 0},
                {"reason_entropy_bound": 1.0},
            ):
                tuned = SystemOneRequest(**{**request.model_dump(), **controls})
                assert reader.shared_context(tuned) == context
                read = reader.prepare_read(
                    tuned, "q", tuned.questions["q"], sample_index, context
                )
                assert read.canvas == baseline.canvas
                assert read.template == baseline.template
                assert read.descriptor() == {
                    **baseline.descriptor(),
                    **{
                        key.replace("reason_", "reasoning_", 1): value
                        for key, value in controls.items()
                        if value is not None
                    },
                }
            assert "reasoning_schedule_steps" not in baseline.descriptor()
            assert "reasoning_entropy_bound" not in baseline.descriptor()


@pytest.mark.asyncio
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
async def test_reason_mode_budget_must_fit_without_truncating_question(canvas_format):
    """Reasoning reserves canvas positions without raising output limits."""
    async with decision_reader() as reader:
        template = reader.template(
            "q", Noul(type="noul"), reason_tokens=10, canvas_format=canvas_format
        )
        exact_length = template.content_tokens + (
            2 if canvas_format == "chat_single_user" else 0
        )
    async with decision_reader(canvas_length=exact_length) as reader:
        assert (
            len(
                reader.template(
                    "q",
                    Noul(type="noul"),
                    reason_tokens=10,
                    canvas_format=canvas_format,
                ).tokens
            )
            == exact_length
        )
        with pytest.raises(ValueError, match="Nothing was truncated"):
            reader.template(
                "q", Noul(type="noul"), reason_tokens=11, canvas_format=canvas_format
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
async def test_initial_answer_override_preserves_reason_noise_and_followup_control(
    canvas_format,
):
    """Initial answer support is independent of the native follow-up noise flag."""
    async with decision_reader() as reader:
        request = SystemOneRequest(
            state="state",
            questions={"q": Noul(type="noul")},
            canvas_format=canvas_format,
            reason_mode=True,
            reason_tokens=10,
        )
        context = reader.shared_context(request)
        variants = [
            request,
            request.model_copy(update={"reason_initial_answer_noise": None}),
            request.model_copy(
                update={
                    "reason_initial_answer_noise": "candidates",
                    "reason_answer_noise": "vocab",
                }
            ),
            request.model_copy(update={"reason_initial_answer_noise": "vocab"}),
            request.model_copy(update={"reason_answer_noise": "vocab"}),
        ]
        for index in range(4):
            reads = [
                reader.prepare_read(value, "q", request.questions["q"], index, context)
                for value in variants
            ]
            assert reads[0] == reads[1]
            assert reads[0].canvas == reads[2].canvas
            assert reads[3].canvas == reads[4].canvas
            assert (
                reads[0].descriptor().get("reasoning_answer_noise", "candidates")
                == "candidates"
            )
            assert reads[2].descriptor()["reasoning_answer_noise"] == "vocab"
            positions = reads[0].template.reasoning_positions
            assert all(
                [read.canvas[pos] for pos in positions]
                == [reads[0].canvas[pos] for pos in positions]
                for read in reads
            )
            assert (
                reads[0].canvas[reads[0].template.answer_position]
                in reads[0].template.candidate_ids
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,reason_tokens,reason_steps,steps,self_conditioning,scope,with_image",
    [
        (1, 0, 1, 1, True, "answer", False),
        (26, 1, 1, 32, False, "canvas", False),
        (27, 32, 8, 4, True, "answer", False),
        (100, 256, 32, 1, True, "answer", False),
        (2, 512, 64, 1, True, "answer", False),
        (2, 32, 64, 8, False, "canvas", True),
    ],
)
async def test_single_user_chat_preserves_original_draws_and_native_reasoning_seed(
    tmp_path,
    count,
    reason_tokens,
    reason_steps,
    steps,
    self_conditioning,
    scope,
    with_image,
):
    """Only the turn boundary, positions and explicit pre-transform seed change."""
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        probabilities = [
            [1 / len(read["logprob_token_ids"])] * len(read["logprob_token_ids"])
            for read in body["diffusion_reads"]
        ]
        result = completion(body, probabilities)
        for choice in result["choices"]:
            choice["token_ids"] = [
                int(row["token"].removeprefix("token_id:"))
                for row in choice["logprobs"]["content"]
            ]
        return httpx.Response(200, json=result)

    body = {
        "state": "A shared state.",
        "questions": {
            "opaque\n☃": {
                "type": "choice",
                "instructions": "Choose the supported option.",
                "criteria": {f"v{i}": None for i in range(count)},
            }
        },
        "reason_mode": reason_tokens > 0,
        "reason_tokens": max(1, reason_tokens),
        "reason_steps": reason_steps,
        "steps": steps,
        "samples": 2,
        "seed": 17,
        "self_conditioning": self_conditioning,
        "self_conditioning_scope": scope,
        "reason_initial_answer_noise": "candidates",
        "reason_answer_noise": "vocab",
    }
    if with_image:
        body["images"] = [{"url": inline_image()[0]}]
    trace = tmp_path / "single-user.jsonl"
    answers = []
    async with api_client(respond, trace=trace) as client:
        for canvas_format in ("chat", "chat_single_user", "chat"):
            response = await client.post(
                "/v1/systemone", json={**body, "canvas_format": canvas_format}
            )
            assert response.status_code == 200, response.text
            answers.append(response.json()["answers"])
    assert requests[0] == requests[2]
    assert answers[0] == answers[1] == answers[2]
    expected = json.loads(json.dumps(requests[0]))
    expected["continue_final_message"] = True
    if with_image:
        expected["messages"][1]["content"].append({"type": "text", "text": ""})
    for old, read in zip(requests[0]["diffusion_reads"], expected["diffusion_reads"]):
        canvas = old["seed_canvas"]
        assert canvas[:3] == [249, 250, 251]
        read["seed_canvas"] = [252, *canvas[3:], 0, 0]
        read["answer_positions"] = [old["answer_positions"][0] - 2]
        if reason_tokens:
            read["reasoning_positions"] = [p - 2 for p in old["reasoning_positions"]]
            read["reasoning_seed"] = int.from_bytes(
                hashlib.sha256(json.dumps(canvas).encode()).digest()[:8], "little"
            )
        assert "reasoning_seed" not in old
        assert len(read["seed_canvas"]) == len(canvas)
    assert requests[1] == expected
    samples = [
        row
        for line in trace.read_text().splitlines()
        if (row := json.loads(line))["event"] == "sample"
    ]
    assert samples[0]["shared_prompt_hash"] != samples[2]["shared_prompt_hash"]
    assert len({row["seed_prompt_hash"] for row in samples}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("canvas_format", ["chat", "chat_single_user"])
async def test_chat_thoughts_return_plain_text_and_preserve_semantic_answers(
    tmp_path, canvas_format
):
    """A shared parent returns real per-draw notes and original Choice/Score keys."""
    note = 'a"\\\t\n12345'
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["n"] == 6
        assert body["add_generation_prompt"] is False
        assert body["return_token_ids"] is True
        assert body["max_tokens"] == max(
            len(row["seed_canvas"]) for row in body["diffusion_reads"]
        )
        assert (
            body["messages"][0]["content"]
            == "<|think|>\n" + server.THOUGHT_INSTRUCTIONS
        )
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body.get("continue_final_message", False) == (
            canvas_format == "chat_single_user"
        )
        values = [[0.75, 0.25]] * 2 + [[0.1, 0.2, 0.7]] * 2 + [[0.6, 0.4]] * 2
        result = reason_completion(body, values)
        for choice, read in zip(result["choices"], body["diffusion_reads"]):
            assert len(read["seed_canvas"]) % 16 == 0
            for position, letter in zip(read["reasoning_positions"], note):
                choice["token_ids"][position] = 256 + ord(letter)
        assert [
            Tokenizer().decode([token])
            for token in body["diffusion_reads"][2]["logprob_token_ids"]
        ] == ["1", "2", "3"]
        result["choices"].reverse()
        return httpx.Response(200, json=result)

    trace = tmp_path / "chat.jsonl"
    async with api_client(respond, trace=trace) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": {"document": "shared"},
                "canvas_format": canvas_format,
                "reason_mode": True,
                "reason_tokens": 10,
                "samples": 2,
                "questions": {
                    "n": {"type": "noul", "instructions": "Relevant?"},
                    "s": {"type": "score", "criteria": ["low", "mid", "high"]},
                    "c": {
                        "type": "choice",
                        "criteria": {"accept": None, "reject": None},
                    },
                },
            },
        )
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    assert answers["n"]["noul"] == pytest.approx(0.75)
    assert answers["s"]["probabilities"] == pytest.approx(
        {"0": 0.1, "1": 0.2, "2": 0.7}
    )
    assert answers["c"]["probabilities"] == pytest.approx(
        {"accept": 0.6, "reject": 0.4}
    )
    assert all(answer["reasoning"] == [note, note] for answer in answers.values())
    assert len(requests) == 1
    samples = [
        record
        for line in trace.read_text().splitlines()
        if (record := json.loads(line))["event"] == "sample"
    ]
    assert all(record["canvas_format"] == canvas_format for record in samples)
    assert all(
        ("reasoning_seed" in record) == (canvas_format == "chat_single_user")
        for record in samples
    )
    assert len({record["shared_prompt_hash"] for record in samples}) == 1
    assert len({record["seed_prompt_hash"] for record in samples}) == 1
    assert samples[0]["shared_prompt_hash"] != samples[0]["seed_prompt_hash"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "fixed"])
@pytest.mark.parametrize("canvas_format", ["chat", "chat_single_user"])
async def test_chat_without_reasoning_requires_the_complete_fixed_canvas(
    damage, canvas_format
):
    """Answer-only logprobs cannot conceal alterations to chat role markers."""

    def respond(request):
        body = json.loads(request.content)
        result = completion(body, [0.75, 0.25])
        choice = result["choices"][0]
        content = choice["logprobs"]["content"]
        if damage != "missing":
            choice["token_ids"] = [
                int(row["token"].removeprefix("token_id:")) for row in content
            ]
            choice["token_ids"][0] = 106
        choice["logprobs"]["content"] = [
            content[body["diffusion_reads"][0]["answer_positions"][0]]
        ]
        return httpx.Response(200, json=result)

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "canvas_format": canvas_format,
            },
        )
    assert response.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    ["missing", "short", "bool", "quote", "special", "fixed", "answer", "content"],
)
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
async def test_reason_mode_rejects_unverifiable_or_illegal_generated_canvas(
    damage, canvas_format
):
    """Compact answer scores cannot hide missing reasoning or altered JSON metadata."""

    def respond(request):
        body = json.loads(request.content)
        result = reason_completion(body, [0.75, 0.25], compact=damage != "content")
        choice = result["choices"][0]
        read = body["diffusion_reads"][0]
        position = read["reasoning_positions"][0]
        if damage == "missing":
            del choice["token_ids"]
        elif damage == "short":
            choice["token_ids"].pop()
        elif damage == "bool":
            choice["token_ids"][position] = True
        elif damage == "quote":
            choice["token_ids"][position] = (
                256 + ord('"') if canvas_format == "json" else 256 + ord("\x01")
            )
        elif damage == "special":
            choice["token_ids"][position] = 106
        elif damage == "fixed":
            choice["token_ids"][0] = 256 + ord("[")
        elif damage == "answer":
            choice["token_ids"][read["answer_positions"][0]] = read[
                "logprob_token_ids"
            ][1]
        else:
            choice["logprobs"]["content"][position]["token"] = (
                f"token_id:{256 + ord('b')}"
            )
        return httpx.Response(200, json=result)

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "reason_mode": True,
                "canvas_format": canvas_format,
                "reason_tokens": 10,
            },
        )
    assert response.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", ["legacy", "compact", "mixed"])
async def test_trimmed_padding_keeps_grouped_reads_and_reports_physical_usage(
    wire_format,
):
    upstream_requests = []

    def respond(request):
        body = json.loads(request.content)
        upstream_requests.append(body)
        result = completion(body, [0.25, 0.75])
        for index, (choice, read) in enumerate(
            zip(result["choices"], body["diffusion_reads"])
        ):
            if wire_format == "compact" or (wire_format == "mixed" and index == 1):
                position = read["answer_positions"][0]
                choice["logprobs"]["content"] = [
                    choice["logprobs"]["content"][position]
                ]
        return httpx.Response(200, json=result)

    request = {
        "state": {"item": "x"},
        "questions": {
            "short": {"type": "noul"},
            "described": {"type": "noul", "instructions": "Is x present?"},
        },
    }
    async with api_client(respond, trim_canvas_padding=True) as client:
        response = await client.post("/v1/systemone", json=request)
    assert response.status_code == 200, response.text
    body = upstream_requests[0]
    lengths = [len(read["seed_canvas"]) for read in body["diffusion_reads"]]
    assert body["n"] == 2
    assert body["max_tokens"] == TEST_CANVAS_LENGTH
    assert lengths[0] < lengths[1]
    assert max(lengths) < TEST_CANVAS_LENGTH
    assert all(read["answer_only_logprobs"] for read in body["diffusion_reads"])
    assert response.json()["usage"]["output_tokens"] == sum(lengths)
    assert response.json()["answers"]["short"]["noul"] == pytest.approx(0.25)
    assert response.json()["answers"]["described"]["noul"] == pytest.approx(0.25)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["legacy_capacity", "compact_row_usage"])
async def test_trimmed_reads_reject_lengths_unrelated_to_physical_canvases(failure):
    def respond(request):
        body = json.loads(request.content)
        if failure == "compact_row_usage":
            result = compact_completion(body, [0.25, 0.75])
            result["usage"]["completion_tokens"] = body["n"]
        else:
            result = completion(body, [0.25, 0.75])
            content = result["choices"][0]["logprobs"]["content"]
            content.extend([content[-1]] * (TEST_CANVAS_LENGTH - len(content)))
        return httpx.Response(200, json=result)

    async with api_client(respond, trim_canvas_padding=True) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}},
        )
    assert response.status_code == 502, response.text
    expected = (
        "completion usage" if failure == "compact_row_usage" else "decision canvas"
    )
    assert expected in response.json()["detail"]


@pytest.mark.asyncio
async def test_trimmed_canvas_can_retain_a_minimum_padding_suffix():
    async with decision_reader(
        trim_canvas_padding=True, minimum_canvas_padding=17
    ) as reader:
        template = reader.template("short", Noul(type="noul"))
        assert len(template.tokens) == template.content_tokens + 17
        assert template.tokens[-17:] == (reader.tokenizer.pad_token_id,) * 17


@pytest.mark.asyncio
async def test_trimmed_canvas_can_preserve_selected_question_types():
    async with decision_reader(
        trim_canvas_padding=True,
        minimum_canvas_padding=4,
        full_canvas_types=frozenset({"choice"}),
    ) as reader:
        choice = reader.template(
            "route", Choice(type="choice", criteria={"left": None, "right": None})
        )
        noul = reader.template("safe", Noul(type="noul"))
        assert len(choice.tokens) == TEST_CANVAS_LENGTH
        assert len(noul.tokens) == noul.content_tokens + 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "canvas_format,reason_tokens,canvas_length",
    [("json", 0, 256), ("chat_single_user", 512, 1024)],
)
async def test_oversized_question_rejects_whole_request_before_any_model_read(
    canvas_format, reason_tokens, canvas_length
):
    """A valid first question must not run when a later canvas cannot fit."""
    upstream_requests = []

    def respond(request):
        upstream_requests.append(json.loads(request.content))
        return httpx.Response(500)

    async with api_client(respond, canvas_length=canvas_length) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "shared",
                "canvas_format": canvas_format,
                "reason_mode": bool(reason_tokens),
                "reason_tokens": max(1, reason_tokens),
                "questions": {
                    "small": {"type": "noul"},
                    "large": {"type": "noul", "instructions": "x" * 1024},
                },
            },
        )
        assert response.status_code == 422
        assert "large" in response.json()["detail"]
        assert "requires" in response.json()["detail"]
        assert f"holds {canvas_length}" in response.json()["detail"]
        assert upstream_requests == []


@pytest.mark.asyncio
async def test_missing_candidate_scores_fail_instead_of_fabricating_zero_probability():
    """An incomplete upstream distribution cannot become a successful answer."""

    def respond(request):
        body = json.loads(request.content)
        result = completion(body, [0.25, 0.75])
        position = body["diffusion_reads"][0]["answer_positions"][0]
        result["choices"][0]["logprobs"]["content"][position]["top_logprobs"] = []
        return httpx.Response(200, json=result)

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}},
        )
        assert response.status_code == 502


@pytest.mark.asyncio
async def test_changed_fixed_canvas_token_is_rejected():
    """Valid slot scores cannot hide an upstream edit to the answer scaffold."""

    def respond(request):
        result = completion(json.loads(request.content), [0.25, 0.75])
        result["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:99"
        return httpx.Response(200, json=result)

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}},
        )
        assert response.status_code == 502
        assert "changed a fixed canvas token" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("logprob", [math.nan, math.inf, -math.inf])
async def test_nonfinite_candidate_score_is_rejected(logprob):
    """Even an unselected candidate must have a finite, usable model score."""

    def respond(request):
        body = json.loads(request.content)
        result = completion(body, [0.25, 0.75])
        position = body["diffusion_reads"][0]["answer_positions"][0]
        row = result["choices"][0]["logprobs"]["content"][position]
        row["top_logprobs"][0]["logprob"] = logprob
        return httpx.Response(
            200,
            content=json.dumps(result),
            headers={"content-type": "application/json"},
        )

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}},
        )
        assert response.status_code == 502
        assert "finite logprob for every candidate" in response.json()["detail"]


@pytest.mark.asyncio
async def test_samples_average_each_question_without_cross_question_coupling():
    """Adding another question changes neither target draws nor their average."""
    upstream_requests: dict[int, list[dict[str, Any]]] = {2: [], 3: []}
    draws = {2: [[0.2, 0.8], [0.6, 0.4]], 3: [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]}

    def respond(request):
        body = json.loads(request.content)
        probabilities = []
        for read in body["diffusion_reads"]:
            count = len(read["logprob_token_ids"])
            sample_index = len(upstream_requests[count]) % 2
            upstream_requests[count].append(read)
            probabilities.append(draws[count][sample_index])
        result = completion(body, probabilities)
        result["choices"].reverse()
        return httpx.Response(200, json=result)

    target = {"type": "choice", "criteria": {"a": None, "b": None}}
    body: dict[str, Any] = {
        "state": {"item": "x"},
        "seed": 17,
        "samples": 2,
        "steps": 3,
        "self_conditioning": False,
        "return_diagnostics": True,
        "questions": {
            "target": target,
            "unrelated": {"type": "score", "criteria": ["none", "some", "all"]},
        },
    }
    async with api_client(respond) as client:
        together = await client.post("/v1/systemone", json=body)
        assert together.status_code == 200, together.text
        result = together.json()
        assert result["answers"]["target"]["probabilities"] == pytest.approx(
            {"a": 0.4, "b": 0.6}
        )
        assert result["answers"]["unrelated"]["probabilities"] == pytest.approx(
            {"0": 0.45, "1": 0.15, "2": 0.4}
        )
        assert result["answers"]["unrelated"]["score"] == pytest.approx(0.95)
        diagnostics = result["diagnostics"]
        assert diagnostics["samples"] == 2
        assert diagnostics["steps"] == 3
        assert diagnostics["self_conditioning"] is False
        assert diagnostics["seed"] == 17
        assert diagnostics["questions"]["target"]["standard_errors"] == pytest.approx(
            {"a": 0.2, "b": 0.2}
        )
        assert diagnostics["questions"]["unrelated"][
            "standard_errors"
        ] == pytest.approx({"0": 0.35, "1": 0.05, "2": 0.3})
        assert all(
            question["samples"] == 2 and question["argmax_agreement"] == 0.5
            for question in diagnostics["questions"].values()
        )
        assert all(
            read["steps"] == 3 and read["self_conditioning"] is False
            for reads in upstream_requests.values()
            for read in reads
        )
        assert result["usage"] == {
            "input_tokens": 160,
            "output_tokens": 4 * TEST_CANVAS_LENGTH,
        }

        alone = await client.post(
            "/v1/systemone", json={**body, "questions": {"target": target}}
        )
        assert alone.status_code == 200, alone.text
        assert alone.json()["answers"]["target"] == result["answers"]["target"]
        assert (
            alone.json()["diagnostics"]["questions"]["target"]
            == diagnostics["questions"]["target"]
        )
        assert len(upstream_requests[2]) == 4
        assert len(upstream_requests[3]) == 2
        assert upstream_requests[2][:2] == upstream_requests[2][2:]

        reordered = await client.post(
            "/v1/systemone",
            json={
                **body,
                "questions": {
                    "unrelated": body["questions"]["unrelated"],
                    "target": target,
                },
            },
        )
        assert reordered.status_code == 200, reordered.text
        assert reordered.json()["answers"]["target"] == result["answers"]["target"]
        assert upstream_requests[2][:2] == upstream_requests[2][4:]


@pytest.mark.asyncio
async def test_adaptive_validation_finishes_before_any_model_read():
    def respond(request):
        pytest.fail("Invalid adaptive controls or templates must not reach the model")

    body: dict[str, Any] = {"state": "x", "questions": {"q": {"type": "noul"}}}
    async with api_client(respond) as client:
        for field, value in (
            ("adaptive_samples", "true"),
            ("max_samples", True),
            ("max_samples", 0),
            ("max_samples", 33),
            ("adaptive_threshold", "0.1"),
            ("adaptive_threshold", True),
            ("adaptive_threshold", -0.1),
            ("adaptive_threshold", 1.1),
            ("adaptive_threshold", math.nan),
            ("adaptive_threshold", math.inf),
        ):
            response = await client.post(
                "/v1/systemone",
                content=json.dumps({**body, field: value}),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422, response.text
            assert response.json()["detail"][0]["loc"] == ["body", field]
        response = await client.post(
            "/v1/systemone",
            json={**body, "adaptive_samples": True, "samples": 5},
        )
        assert response.status_code == 422
        response = await client.post(
            "/v1/systemone",
            json={
                **body,
                "adaptive_samples": True,
                "questions": {
                    **body["questions"],
                    "invalid": {"type": "noul", "instructions": "x" * 3000},
                },
            },
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_adaptive_sampling_uses_mean_entropy_and_accounts_for_each_round(
    tmp_path,
):
    """Conflicting confident draws trigger more work only for their question."""
    requests: list[dict[str, Any]] = []
    trace = tmp_path / "adaptive.jsonl"

    def respond(request):
        body = json.loads(request.content)
        round_index = len(requests)
        requests.append(body)
        probabilities = (
            [[0.99, 0.01], [0.01, 0.99], [0.99, 0.005, 0.005]]
            if round_index == 0
            else [[0.8, 0.2], [0.6, 0.4]]
        )
        if round_index == 0:
            probabilities.append(probabilities[-1])
        result = compact_completion(body, probabilities)
        result["id"] = f"chatcmpl-round-{round_index}"
        result["usage"]["prompt_tokens"] = 40 + 20 * round_index
        result["choices"].reverse()
        return httpx.Response(200, json=result)

    async with api_client(respond, trace=trace) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "shared",
                "samples": 2,
                "adaptive_samples": True,
                "max_samples": 4,
                "adaptive_threshold": 0.6,
                "steps": 3,
                "self_conditioning": False,
                "return_diagnostics": True,
                "questions": {
                    "uncertain": {"type": "noul"},
                    "confident": {"type": "score", "criteria": ["a", "b", "c"]},
                },
            },
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert [body["n"] for body in requests] == [4, 2]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert all(
        read["steps"] == 3 and read["self_conditioning"] is False
        for body in requests
        for read in body["diffusion_reads"]
    )
    assert result["answers"]["uncertain"]["noul"] == pytest.approx(0.6)
    assert result["usage"] == {
        "input_tokens": 280,
        "output_tokens": 6 * TEST_CANVAS_LENGTH,
    }
    diagnostics = result["diagnostics"]
    assert diagnostics["adaptive_samples"] is True
    assert diagnostics["max_samples"] == 4
    assert diagnostics["adaptive_threshold"] == 0.6
    assert diagnostics["questions"]["uncertain"]["samples"] == 4
    assert diagnostics["questions"]["confident"]["samples"] == 2
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert {event["request_id"] for event in events} == {
        response.headers["x-typesafe-request-id"]
    }
    parents = [event for event in events if event["event"] == "parent"]
    assert [parent["round"] for parent in parents] == [0, 1]
    assert parents[1]["children"] == [
        {"choice_index": index, "question_id": "uncertain", "sample": index + 2}
        for index in range(2)
    ]
    assert [event["round"] for event in events if event["event"] == "sample"] == [
        0,
        0,
        0,
        0,
        1,
        1,
    ]
    assert events[-1]["parent_ids"] == [parent["parent_id"] for parent in parents]
    assert events[-1]["rounds"] == 2 and events[-1]["child_count"] == 6
    viewer = Trace(trace)
    viewer.read(final=True)
    for sample in viewer.samples:
        context = parent_context(viewer, sample)
        parent = parents[sample["round"]]
        assert context["parent_id"] == parent["parent_id"]
        assert context["child_count"] == parent["child_count"]
        assert context["children"] == parent["children"]


@pytest.mark.asyncio
@pytest.mark.parametrize("canvas_format", ["json", "chat_single_user"])
async def test_adaptive_draw_prefix_is_independent_of_rounds_neighbors_and_requests(
    canvas_format,
):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        values = [
            [0.4, 0.6] if len(read["logprob_token_ids"]) == 2 else [0.99, 0.005, 0.005]
            for read in body["diffusion_reads"]
        ]
        result = (
            reason_completion(body, values)
            if canvas_format == "chat_single_user"
            else compact_completion(body, values)
        )
        return httpx.Response(200, json=result)

    target = {"type": "noul"}
    unrelated = {"type": "score", "criteria": ["a", "b", "c"]}
    body = {
        "state": "shared",
        "seed": 17,
        "return_diagnostics": True,
        "questions": {"target": target},
        "canvas_format": canvas_format,
        "reason_mode": canvas_format == "chat_single_user",
        "reason_tokens": 10,
    }
    batches = []
    results = []
    async with api_client(respond) as client:
        for controls in (
            {"samples": 3, "max_samples": 1},
            {"adaptive_samples": True, "max_samples": 3},
            {
                "adaptive_samples": True,
                "max_samples": 3,
                "questions": {"target": target, "unrelated": unrelated},
            },
            {
                "adaptive_samples": True,
                "max_samples": 3,
                "questions": {"unrelated": unrelated, "target": target},
            },
            {
                "adaptive_samples": True,
                "max_samples": 3,
                "questions": {"unrelated": unrelated, "renamed": target},
            },
            {},
        ):
            start = len(requests)
            response = await client.post("/v1/systemone", json={**body, **controls})
            assert response.status_code == 200, response.text
            results.append(response.json())
            questions = cast(dict[str, Any], controls.get("questions", {}))
            target_id = "renamed" if "renamed" in questions else "target"
            assert results[-1]["answers"][target_id]["noul"] == pytest.approx(0.4)
            batches.append(
                [
                    read
                    for request in requests[start:]
                    for read in request["diffusion_reads"]
                    if len(read["logprob_token_ids"]) == 2
                ]
            )
    assert batches[0] == batches[1] == batches[2] == batches[3] == batches[4]
    assert batches[5] == batches[0][:1]
    assert results[-1]["diagnostics"]["questions"]["target"]["samples"] == 1
    assert "adaptive_samples" not in results[-1]["diagnostics"]


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [{"adaptive_threshold": 1.0}, {"max_samples": 1}])
async def test_adaptive_sampling_avoids_an_empty_followup_round(controls):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=compact_completion(body, [0.5, 0.5]))

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "adaptive_samples": True,
                **controls,
            },
        )
    assert response.status_code == 200, response.text
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_adaptive_second_round_failure_never_returns_initial_answers(
    tmp_path, cancel
):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []
    trace = tmp_path / "adaptive-failure.jsonl"

    async def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(200, json=compact_completion(body, [0.5, 0.5]))
        if not cancel:
            return httpx.Response(500)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with api_client(respond, trace=trace) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                json={
                    "state": "x",
                    "questions": {"q": {"type": "noul"}},
                    "adaptive_samples": True,
                },
            )
        )
        if cancel:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
        else:
            response = await task
            assert response.status_code == 502 and "answers" not in response.json()
    assert len(calls) == 2
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert not any(event["event"] == "response" for event in events)
    assert events[-1]["event"] == "parent"
    assert events[-1]["round"] == 1 and events[-1]["status"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_group_trace_maps_six_children_without_multiplying_parent_usage(
    tmp_path, compact
):
    """One parent owns all draws, but only logical SDK usage repeats the prompt."""
    trace = tmp_path / "group.jsonl"
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        result = completion(body, [[0.2, 0.8]] * 2 + [[0.1, 0.2, 0.7]] * 4)
        if compact:
            for choice, read in zip(result["choices"], body["diffusion_reads"]):
                position = read["answer_positions"][0]
                choice["logprobs"]["content"] = [
                    choice["logprobs"]["content"][position]
                ]
        result["choices"].reverse()
        result["usage"]["prompt_tokens_details"] = {
            "cached_tokens": 32,
            "created_cache_tokens": 32,
        }
        return httpx.Response(200, json=result)

    async with api_client(respond, trace=trace) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "shared",
                "samples": 2,
                "questions": {
                    "binary": {"type": "noul"},
                    "category": {
                        "type": "choice",
                        "criteria": {"a": None, "b": None, "c": None},
                    },
                    "rating": {"type": "score", "criteria": ["none", "some", "all"]},
                },
            },
        )
    assert response.status_code == 200, response.text
    assert len(requests) == 1 and requests[0]["n"] == 6
    assert response.json()["usage"] == {
        "input_tokens": 240,
        "output_tokens": 6 * TEST_CANVAS_LENGTH,
    }
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    parent, *samples, result = events
    assert parent["event"] == "parent" and parent["status"] == "completed"
    assert parent["upstream_usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 6 * TEST_CANVAS_LENGTH,
    }
    assert parent["children"] == [
        {"choice_index": index, "question_id": name, "sample": draw}
        for index, (name, draw) in enumerate(
            (name, draw)
            for name in ("binary", "category", "rating")
            for draw in range(2)
        )
    ]
    assert result["event"] == "response" and result["parent_id"] == "chatcmpl-test"
    for index, sample in enumerate(samples):
        assert sample["event"] == "sample"
        assert sample["parent_id"] == "chatcmpl-test"
        assert sample["choice_index"] == index and sample["child_count"] == 6
        assert sample["cached_prompt_tokens"] is None
        assert sample["created_cache_tokens"] is None
        assert sample["shared_prompt_tokens"] == 40
        assert sample["canvas_text"] == [
            Tokenizer().decode([token]) for token in sample["canvas_tokens"]
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "range",
        "bool",
        "failed",
        "truncated",
        "illegal",
        "duplicate_candidate",
        "usage",
    ],
)
async def test_invalid_group_fails_without_partial_sample_trace(tmp_path, failure):
    """A valid first child must not leak a successful sample when its sibling fails."""
    trace = tmp_path / "invalid-group.jsonl"
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        result = completion(body, [0.25, 0.75])
        choice = result["choices"][-1]
        if failure == "missing":
            result["choices"].pop()
        elif failure in {"duplicate", "range", "bool"}:
            choice["index"] = {"duplicate": 0, "range": 2, "bool": True}[failure]
        elif failure == "failed":
            choice["finish_reason"] = "error"
        elif failure == "truncated":
            choice["logprobs"]["content"].pop()
        elif failure == "usage":
            result["usage"]["completion_tokens"] -= 1
        else:
            position = body["diffusion_reads"][-1]["answer_positions"][0]
            row = choice["logprobs"]["content"][position]
            if failure == "illegal":
                row["token"] = "token_id:123"
            else:
                row["top_logprobs"].append(row["top_logprobs"][0])
        return httpx.Response(200, json=result)

    async with api_client(respond, trace=trace) as client:
        response = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul"}}, "samples": 2},
        )
    assert response.status_code == 502, response.text
    assert len(calls) == 1
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event["event"] for event in events] == ["parent", "error"]
    assert events[0]["status"] == "error"
    assert events[0]["parent_id"] == "chatcmpl-test"


@pytest.mark.asyncio
async def test_cancelling_group_cancels_its_only_upstream_request(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with api_client(respond, trace=tmp_path / "cancel.jsonl") as client:
        task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                json={"state": "x", "questions": {"q": {"type": "noul"}}, "samples": 2},
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


def inline_image(format="PNG", color="red", *, animated=False):
    """Small generated input images exercise parsing without model inference."""
    output = BytesIO()
    image = Image.new("RGB", (2, 3), color)
    kwargs = (
        {"save_all": True, "append_images": [Image.new("RGB", (2, 3), "blue")]}
        if animated
        else {}
    )
    image.save(output, format=format, **kwargs)
    data = output.getvalue()
    mime_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[format]
    return f"data:{mime_type};base64," + base64.b64encode(data).decode(), data


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_supported_images_keep_original_bytes_and_safe_metadata(format):
    url, data = inline_image(format)
    blocks, metadata = prepare_images([ImageInput(url=url)])
    assert blocks == [{"type": "image_url", "image_url": {"url": url}}]
    assert metadata == [
        {
            "sha256": hashlib.sha256(data).hexdigest(),
            "mime_type": url.split(";")[0].removeprefix("data:"),
            "width": 2,
            "height": 3,
            "byte_size": len(data),
        }
    ]


@pytest.mark.asyncio
async def test_images_share_multimodal_prefill_and_never_enter_canvas_or_trace(
    tmp_path,
):
    """Image identity belongs to shared prefill, not per-question JSON or logs."""
    url, data = inline_image()
    changed_url, _ = inline_image(color="green")
    trace_path = tmp_path / "images.jsonl"
    upstream_requests = []

    def respond(request):
        body = json.loads(request.content)
        upstream_requests.append(body)
        return httpx.Response(200, json=completion(body, [0.25, 0.75]))

    body = {
        "state": {"task": "Inspect the supplied image."},
        "questions": {
            "red": {"type": "noul", "instructions": "Is the image red?"},
            "green": {"type": "noul", "instructions": "Is the image green?"},
        },
        "images": [{"url": url}],
        "samples": 2,
        "seed": 17,
    }
    async with api_client(respond, trace=trace_path) as client:
        for images in ([{"url": url}], [{"url": changed_url}]):
            response = await client.post(
                "/v1/systemone", json={**body, "images": images}
            )
            assert response.status_code == 200, response.text
    assert len(upstream_requests) == 2
    for first, batch in enumerate(upstream_requests):
        assert batch["n"] == 4
        assert len(batch["diffusion_reads"]) == 4
        content = batch["messages"][1]["content"]
        assert content[0] == {
            "type": "text",
            "text": 'Shared state:\n{"task":"Inspect the supplied image."}',
        }
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": url if first == 0 else changed_url},
        }
    text = trace_path.read_text()
    assert url not in text and changed_url not in text
    assert base64.b64encode(data).decode() not in text
    samples = [
        item
        for line in text.splitlines()
        if (item := json.loads(line))["event"] == "sample"
    ]
    assert all(item["image_count"] == 1 for item in samples)
    assert len({item["shared_prompt_hash"] for item in samples[:4]}) == 1
    assert len({item["shared_prompt_hash"] for item in samples[4:]}) == 1
    assert samples[0]["shared_prompt_hash"] != samples[4]["shared_prompt_hash"]
    assert samples[0]["shared_images"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    for sample in samples:
        assert "data:" not in "".join(sample["canvas_text"])
        assert "image_url" not in sample["fixed_question_json"]


@pytest.mark.asyncio
async def test_default_inputs_preserve_text_messages_and_seed_canvas():
    upstream_requests = []

    def respond(request):
        body = json.loads(request.content)
        upstream_requests.append(body)
        return httpx.Response(200, json=completion(body, [0.25, 0.75]))

    body = {"state": {"item": "x"}, "questions": {"q": {"type": "noul"}}, "seed": 17}
    extra: dict[str, Any]
    async with api_client(respond) as client:
        for extra in (
            {},
            {"images": []},
            {"system_prompt": None},
            {"system_prompt": server.COMMON_INSTRUCTIONS},
            {"canvas_format": "json"},
            {"reason_initial_answer_noise": None},
            {"reason_schedule_steps": None, "reason_entropy_bound": None},
            {"reason_schedule_steps": 1, "reason_entropy_bound": 0},
        ):
            response = await client.post("/v1/systemone", json={**body, **extra})
            assert response.status_code == 200, response.text
    assert all(body == upstream_requests[0] for body in upstream_requests[1:])
    assert upstream_requests[0]["messages"] == [
        {"role": "system", "content": server.COMMON_INSTRUCTIONS},
        {"role": "user", "content": 'Shared state:\n{"item":"x"}'},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("canvas_format", ["json", "chat", "chat_single_user"])
@pytest.mark.parametrize("reason_mode", [False, True])
async def test_system_prompt_replacement_is_request_local_and_shared(
    tmp_path, with_image, canvas_format, reason_mode
):
    """Prompt overrides reach every child without changing later requests."""
    upstream_requests = []

    def respond(request):
        body = json.loads(request.content)
        upstream_requests.append(body)
        if reason_mode:
            result = reason_completion(body, [0.25, 0.75])
        else:
            result = completion(body, [0.25, 0.75])
            if canvas_format != "json":
                for choice in result["choices"]:
                    choice["token_ids"] = [
                        int(row["token"].removeprefix("token_id:"))
                        for row in choice["logprobs"]["content"]
                    ]
        return httpx.Response(200, json=result)

    trace_path = tmp_path / "system-prompts.jsonl"
    body = {
        "state": {"item": "x"},
        "questions": {"q": {"type": "noul"}, "r": {"type": "noul"}},
        "samples": 2,
        "seed": 17,
        "canvas_format": canvas_format,
        "reason_mode": reason_mode,
        "reason_tokens": 10,
    }
    if with_image:
        body["images"] = [{"url": inline_image()[0]}]
    prompts = ["Apply policy A.", "Apply policy B.", "Apply policy A.", ""]
    async with api_client(respond, trace=trace_path) as client:
        for prompt in prompts:
            response = await client.post(
                "/v1/systemone", json={**body, "system_prompt": prompt}
            )
            assert response.status_code == 200, response.text
            assert response.headers["x-typesafe-constraint-backend"] == (
                "fixed-token-candidates" if canvas_format != "json" else "xgrammar"
            )

    for prompt, request in zip(prompts, upstream_requests):
        expected = (
            "<|think|>\n" + prompt
            if canvas_format != "json" and reason_mode
            else prompt
        )
        assert request["messages"][0] == {"role": "system", "content": expected}
        assert request["messages"][1] == upstream_requests[0]["messages"][1]
        assert request["n"] == 4
        if with_image and canvas_format == "chat_single_user":
            content = request["messages"][1]["content"]
            assert content[-1] == {"type": "text", "text": ""}
            assert content[-2]["type"] == "image_url"
    assert upstream_requests[0] == upstream_requests[2]
    samples = [
        event
        for line in trace_path.read_text().splitlines()
        if (event := json.loads(line))["event"] == "sample"
    ]
    hashes = [samples[index]["shared_prompt_hash"] for index in range(0, 16, 4)]
    assert hashes[0] == hashes[2]
    assert len(set(hashes)) == 3
    if canvas_format != "json":
        assert all(
            request["diffusion_reads"] == upstream_requests[0]["diffusion_reads"]
            for request in upstream_requests
        )
        assert len({sample["seed_prompt_hash"] for sample in samples}) == 1
    for index, prompt_hash in enumerate(hashes):
        assert all(
            sample["shared_prompt_hash"] == prompt_hash
            for sample in samples[index * 4 : (index + 1) * 4]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "remote",
        "file",
        "unsupported",
        "base64",
        "not_image",
        "mime",
        "truncated",
        "animated",
        "count",
        "schema",
        "schema_type",
        "top_level_list",
        "question_tag",
        "field_name",
    ],
)
async def test_invalid_images_reject_whole_call_without_outbound_or_payload_echo(
    invalid,
):
    url, data = inline_image()
    images: list[dict[str, Any]] = [{"url": url}]
    if invalid in {"remote", "file"}:
        bad_url = (
            "https://example.invalid/image.png"
            if invalid == "remote"
            else "file:///tmp/image.png"
        )
    elif invalid == "unsupported":
        bad_url = url.replace("image/png", "image/gif")
    elif invalid == "base64":
        bad_url = "data:image/png;base64,SECRET_NOT_BASE64!"
    elif invalid == "not_image":
        bad_url = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
    elif invalid == "mime":
        bad_url = url.replace("image/png", "image/jpeg")
    elif invalid == "truncated":
        bad_url = "data:image/png;base64," + base64.b64encode(data[:40]).decode()
    elif invalid == "animated":
        bad_url, _ = inline_image(animated=True)
    else:
        bad_url = url
    images.append({"url": bad_url})
    if invalid == "count":
        images = [{"url": url}] * 5
    elif invalid == "schema":
        images = [{"url": url, "unexpected": url}]
    elif invalid == "schema_type":
        images = [{"url": [url]}]
    elif invalid == "field_name":
        images = [{"url": url, url: url}]
    upstream_requests = []

    def respond(request):
        upstream_requests.append(request)
        return httpx.Response(500)

    body: Any = {
        "state": "x",
        "questions": {"q": {"type": "noul"}},
        "images": images,
    }
    if invalid == "question_tag":
        body["questions"]["q"]["type"] = url
    elif invalid == "top_level_list":
        body = [body]
    async with api_client(respond) as client:
        response = await client.post("/v1/systemone", json=body)
    assert response.status_code == 422, response.text
    assert url not in response.text and bad_url not in response.text
    assert upstream_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit", ["MAX_IMAGE_BYTES", "MAX_TOTAL_IMAGE_BYTES", "MAX_IMAGE_PIXELS"]
)
async def test_image_resource_limits_reject_before_upstream(monkeypatch, limit):
    url, data = inline_image()
    bound = 5 if limit == "MAX_IMAGE_PIXELS" else len(data) - 1
    images = [{"url": url}]
    if limit == "MAX_TOTAL_IMAGE_BYTES":
        bound = len(data) + 1
        images *= 2
    monkeypatch.setattr(server, limit, bound)
    upstream_requests = []

    def respond(request):
        upstream_requests.append(request)
        return httpx.Response(500)

    async with api_client(respond) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "images": images,
            },
        )
    assert response.status_code == 422, response.text
    assert upstream_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["status", "token", "transport"])
async def test_image_upstream_errors_do_not_reflect_payloads(tmp_path, failure):
    url, _ = inline_image()
    trace_path = tmp_path / "errors.jsonl"

    def respond(request):
        if failure == "status":
            return httpx.Response(400, text=url)
        if failure == "transport":
            raise httpx.ReadError(url)
        result = completion(json.loads(request.content), [0.25, 0.75])
        result["choices"][0]["logprobs"]["content"][0]["token"] = url
        return httpx.Response(200, json=result)

    async with api_client(respond, trace=trace_path) as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {"q": {"type": "noul"}},
                "images": [{"url": url}],
            },
        )
    assert response.status_code == 502
    assert url not in response.text
    if trace_path.exists():
        assert url not in trace_path.read_text()


def render_canvas(record, **kwargs):
    output = StringIO()
    Console(file=output, width=120, color_system=None).print(
        canvas_panel(record, **kwargs)
    )
    return output.getvalue()


@pytest.mark.parametrize("status", ["completed", "error"])
def test_parent_child_mapping_uses_choice_indices_not_sample_arrival(tmp_path, status):
    """Out-of-order samples remain attached to the recorded parent child map."""
    parent_id = "chatcmpl-" + "a" * 32
    parent = {
        "event": "parent",
        "request_id": "sdk-request",
        "parent_id": parent_id,
        "child_count": 6,
        "status": status,
        "children": [
            {
                "choice_index": index,
                "question_id": f"q{index // 2}",
                "sample": index % 2,
            }
            for index in range(6)
        ],
    }
    samples = [
        {
            "event": "sample",
            "request_id": "sdk-request",
            "parent_id": parent_id,
            "child_count": 6,
            **parent["children"][index],
        }
        for index in (5, 1)
    ]
    response = {
        "event": "response",
        "request_id": "sdk-request",
        "parent_id": parent_id,
        "execution_evidence": {
            "model_forward_count": 2,
            "source": "audit/decoder-forwards.json",
            "verified": True,
        },
    }
    trace = Trace(tmp_path / "parallel.jsonl")
    trace.path.write_text(
        "\n".join(json.dumps(row) for row in [parent, *samples, response])
    )
    trace.read(final=True)
    output = StringIO()
    Console(file=output, width=155, color_system=None).print(
        render(trace, 0, False, False, 0, details=True, width=155)
    )
    text = output.getvalue()
    assert "child 6/6 (choice 5) · q2 · draw 1" in text
    assert f"Parent ID: {parent_id}" in text
    assert f"Status: {status} · children: 6" in text
    assert "Model forwards: 2 (verified)" in text
    assert "audit/decoder-forwards.json" in text
    for index in range(6):
        state = "recorded" if index in (5, 1) else "not recorded"
        assert re.search(rf"\b{index}\s+q{index // 2}\s+{index % 2}\s+{state}\b", text)


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {"verified": False, "model_forward_count": 2, "source": "audit.json"},
        {"verified": "true", "model_forward_count": 2, "source": "audit.json"},
        {"verified": True, "model_forward_count": 2},
    ],
)
def test_model_forwards_are_unverified_without_explicit_audit(tmp_path, evidence):
    """Configured denoising steps do not establish a measured forward count."""
    trace = Trace(tmp_path / "synthetic.jsonl")
    trace.samples.append(
        {
            "parent_id": "parent-a",
            "child_count": 6,
            "choice_index": 0,
            "denoise_steps": 1,
            "execution_evidence": evidence,
        }
    )
    output = StringIO()
    Console(file=output, width=155, color_system=None).print(
        render(trace, 0, False, False, 0, details=True, width=155)
    )
    text = output.getvalue()
    assert "denoise passes 1" in text
    assert "Model forwards: unverified" in text
    assert "Model forwards: 2" not in text


def test_parent_error_without_samples_does_not_survive_trace_replacement(tmp_path):
    """A failed parent is visible, but cannot leak into a replacement legacy trace."""
    trace = Trace(tmp_path / "parallel.jsonl")
    trace.path.write_text(
        json.dumps(
            {
                "event": "parent",
                "request_id": "sdk-request",
                "parent_id": "parent-error",
                "child_count": 6,
                "status": "error",
            }
        )
    )
    trace.read(final=True)
    output = StringIO()
    console = Console(file=output, width=155, color_system=None)
    console.print(render(trace, 0, False, False, 0, width=155))
    assert "Status: error · children: 6" in output.getvalue()
    assert "No sample events recorded" in output.getvalue()
    trace.path.write_text('{"event":"sample","question_id":"legacy"}')
    trace.read(final=True)
    output.seek(0)
    output.truncate()
    console.print(render(trace, 0, False, False, 0, width=155))
    assert "question legacy" in output.getvalue()
    assert "parent-error" not in output.getvalue()
    assert not trace.parents


@pytest.mark.parametrize("count,columns,rows", [(256, 16, 16), (32, 8, 4)])
def test_canvas_grid_renders_every_recorded_position_once(count, columns, rows):
    """Synthetic unique previews expose skipped, duplicated, or invented cells."""
    record = {
        "canvas_tokens": list(range(1000, 1000 + count)),
        "canvas_text": [f"t{index:03d}" for index in range(count)],
        "answer_position": 2,
    }
    rendered = render_canvas(record, selected_position=2)
    grid_lines = [
        line for line in rendered.splitlines() if re.search(r"\b\d{3}–\d{3}\b", line)
    ]
    assert canvas_grid_shape(count) == (columns, rows)
    assert len(grid_lines) == rows
    assert re.findall(r"\bt(\d{3})\b", "\n".join(grid_lines)) == [
        f"{index:03d}" for index in range(count)
    ]
    assert f"{count - columns:03d}–{count - 1:03d}" in grid_lines[-1]


def test_canvas_before_and_after_show_the_recorded_answer_slot():
    """The inspector changes from seeded noise to the actual legal output token."""
    record = {
        "canvas_tokens": [10, 99, 0],
        "canvas_text": ["answer", "noise", "<pad>"],
        "output_tokens": [10, 201, 0],
        "answer_position": 1,
        "candidate_token_ids": [201, 202],
        "candidate_codes": ["yes", "no"],
        "candidate_labels": ["true", "false"],
    }
    before = render_canvas(record, selected_position=1)
    after = render_canvas(record, output=True, selected_position=1)
    assert "Position 001 · token ID 99" in before
    assert "noise" in before
    assert "Position 001 · token ID 201" in after
    assert "yes" in after


def test_canvas_padding_is_recorded_but_missing_output_is_not_fabricated():
    """Token ID zero is real padding, not a placeholder for unavailable output."""
    record = {
        "canvas_tokens": [10, 99, 0],
        "canvas_text": ["answer", "noise", "<pad>"],
        "answer_position": 1,
    }
    before = render_canvas(record, selected_position=2)
    missing_output = render_canvas(record, output=True, selected_position=2)
    assert "PAD" in before
    assert "Position 002 · token ID 0" in before
    assert "<pad>" in before
    assert "Output canvas was not recorded" in missing_output
    assert "token ID 0" not in missing_output


@pytest.mark.parametrize("on_canvas", [False, True])
def test_context_placement_uses_recorded_metadata_only(tmp_path, on_canvas):
    """Legacy traces must not inherit the new shared-prompt placement claim."""
    trace = Trace(tmp_path / "synthetic.jsonl")
    record = {
        "state": {"shared": "SHARED_STATE_SENTINEL"},
        "question": {"instructions": "RAW_QUESTION_SENTINEL"},
        "fixed_question_json": json.dumps(
            {
                "instructions": "FIXED_CANVAS_QUESTION",
                "criteria": {"true": "YES_DESCRIPTION"},
            }
        ),
        "canvas_tokens": [10, 11, 12, 99] + [0] * 28,
        "canvas_text": ["{", "question", ":", "noise"] + ["<pad>"] * 28,
        "answer_position": 3,
        "shared_prompt_hash": "abc123def456" + "7" * 52,
        "shared_prompt_tokens": 140,
        "question_canvas_tokens": 4,
        "cached_prompt_tokens": None,
    }
    if on_canvas:
        record["question_placement"] = "canvas"
    trace.samples.append(record)
    output = StringIO()
    Console(file=output, width=155, color_system=None).print(
        render(trace, 0, False, False, 0, details=True, width=155)
    )
    text = output.getvalue()
    assert "SHARED_STATE_SENTINEL" in text
    assert "Shared images:" not in text
    assert "SHARED PROMPT IMAGES" not in text
    if on_canvas:
        assert "SHARED PROMPT STATE" in text
        assert "QUESTION JSON ON CANVAS" in text
        assert "FIXED_CANVAS_QUESTION" in text
        assert "YES_DESCRIPTION" in text
        assert "RAW_QUESTION_SENTINEL" not in text
        assert text.index("FIXED_CANVAS_QUESTION") < text.index("SHARED_STATE_SENTINEL")
        assert "shared prompt abc123def456" in text
        assert "prompt tokens: 140" in text
        assert "used canvas tokens: 4" in text
        assert "upstream cached: unreported" in text
        grid = render_canvas(record)
        assert "Seeded JSON canvas" in grid
        assert "Fixed JSON/scaffold: 3 tokens" in grid
        assert "fixed PAD: 28" in grid
    else:
        assert "SHARED PROMPT STATE" not in text
        assert "QUESTION JSON ON CANVAS" not in text
        assert "RAW_QUESTION_SENTINEL" in text
        assert "FIXED_CANVAS_QUESTION" not in text
        assert "shared prompt abc123def456" not in text


def test_shared_images_show_metadata_without_becoming_canvas_cells(tmp_path):
    """Image context is metadata only; arbitrary trace fields are not displayed."""
    trace = Trace(tmp_path / "synthetic.jsonl")
    record: dict[str, Any] = {
        "question_placement": "canvas",
        "canvas_tokens": [10, 99] + [0] * 254,
        "canvas_text": ["answer", "noise"] + ["<pad>"] * 254,
        "answer_position": 1,
    }
    original_canvas = render_canvas(record)
    record.update(
        image_count=2,
        shared_images=[
            {
                "sha256": "a" * 64,
                "mime_type": "image/png",
                "width": 640,
                "height": 480,
                "byte_size": 2048,
            },
            {
                "sha256": "b" * 64,
                "mime_type": "image/jpeg",
                "width": 320,
                "height": 240,
                "byte_size": 1024,
                "url": "https://example.invalid/NOT_IMAGE_METADATA",
            },
        ],
    )
    trace.samples.append(record)
    output = StringIO()
    Console(file=output, width=155, color_system=None).print(
        render(trace, 0, False, False, 0, details=True, width=155)
    )
    text = output.getvalue()
    assert "Shared images: 2" in text
    assert "SHARED PROMPT IMAGES: 2" in text
    assert "shared prefill, not the token canvas" in text
    assert "pixels are not rendered" in text
    for image in record["shared_images"]:
        for field in ("sha256", "mime_type", "width", "height", "byte_size"):
            assert str(image[field]) in text
    assert "NOT_IMAGE_METADATA" not in text
    assert render_canvas(record) == original_canvas


def test_keyboard_burst_preserves_every_ordinary_key():
    assert input_keys("ppp") == (["p", "p", "p"], "")
    assert input_keys("nv q") == (["n", "v", " ", "q"], "")


def test_keyboard_retains_split_arrow_sequence_without_losing_adjacent_keys():
    keys, pending = input_keys("p\x1b[")
    assert keys == ["p"]
    assert pending == "\x1b["
    assert input_keys(pending + "Cvn") == (["right", "v", "n"], "")


def test_keyboard_retains_split_page_key_and_adjacent_navigation():
    keys, pending = input_keys("d\x1b[6")
    assert keys == ["d"]
    assert pending == "\x1b[6"
    assert input_keys(pending + "~\x1b[5~\x1bOH\x1bOF][") == (
        ["page_down", "page_up", "home", "end", "]", "["],
        "",
    )


@pytest.mark.parametrize("width,height", [(155, 38), (188, 46), (240, 60)])
def test_details_scroll_preserves_full_state_rubrics_and_all_categories(
    tmp_path, width, height
):
    """Every detail remains reachable in a terminal-sized paginated view."""
    labels = [f"category{index:02d}" for index in range(26)]
    trace = Trace(tmp_path / "synthetic.jsonl")
    trace.samples.append(
        {
            "state": {"entries": [f"state-{index:02d}" for index in range(40)]},
            "question": {
                "type": "choice",
                "instructions": "Choose the category using the complete rubric.",
                "criteria": {
                    label: f"criterion-{index:02d}"
                    for index, label in enumerate(labels)
                },
            },
            "candidate_labels": labels,
            "candidate_codes": [chr(65 + index) for index in range(26)],
            "candidate_token_ids": list(range(65, 91)),
            "candidate_logprobs": [math.log(1 / 26)] * 26,
            "probabilities": [1 / 26] * 26,
        }
    )
    scroll = DetailScroll()
    pages = []
    while True:
        output = StringIO()
        Console(file=output, width=width, height=height, color_system=None).print(
            render(
                trace,
                0,
                False,
                False,
                0,
                details=True,
                width=width,
                height=height,
                detail_scroll=scroll,
            )
        )
        page = output.getvalue()
        assert len(page.splitlines()) <= height - 1
        assert "PgUp/PgDn" in page
        assert "q quit" in page.splitlines()[-1]
        pages.append(page)
        if scroll.offset == scroll.maximum:
            break
        scroll.move("page_down")
    assert len(pages) > 1
    complete = "\n".join(pages)
    assert "state-39" in complete
    assert "criterion-25" in complete
    assert all(label in complete for label in labels)
    scroll.move("home")
    assert scroll.offset == 0
    scroll.move("end")
    assert scroll.offset == scroll.maximum


@pytest.mark.parametrize(
    "width,height,preview,cell_height",
    [
        (96, 38, "ans…", 1),
        (155, 38, "answ…", 1),
        (188, 46, "answer…", 2),
        (240, 60, "answer_de…", 2),
        (320, 80, "answer_detail", 3),
    ],
)
@pytest.mark.parametrize("with_image", [False, True])
def test_full_view_keeps_last_canvas_row_in_sidebar_and_stacked_layouts(
    tmp_path, width, height, preview, cell_height, with_image
):
    """Resizing expands actual cells without losing rows, metadata, or controls."""
    trace = Trace(tmp_path / "synthetic.jsonl")
    trace.samples.append(
        {
            "canvas_tokens": [10, 99] + [0] * 254,
            "canvas_text": ["answer_detail", "noise"] + ["<pad>"] * 254,
            "output_tokens": [10, 201] + [0] * 254,
            "answer_position": 1,
            "candidate_token_ids": [201, 202],
            "candidate_codes": ["yes", "no"],
            "candidate_labels": ["true", "false"],
            "candidate_logprobs": [math.log(0.75), math.log(0.25)],
            "probabilities": [0.75, 0.25],
            "state": {"door": "open"},
            "question": {"type": "noul", "instructions": "Is the door open?"},
            "question_placement": "canvas",
            "fixed_question_json": {
                "type": "noul",
                "description": "Is the door open?",
            },
            "shared_prompt_hash": "a" * 64,
            "shared_prompt_tokens": 140,
            "question_canvas_tokens": 2,
            "cached_prompt_tokens": 128,
            "constraint_backend": "xgrammar",
            "constraint_version": "0.2.7",
        }
    )
    if with_image:
        trace.samples[0].update(
            parent_id="chatcmpl-" + "a" * 32,
            child_count=6,
            choice_index=3,
            image_count=1,
            shared_images=[
                {
                    "sha256": "a" * 64,
                    "mime_type": "image/png",
                    "width": 640,
                    "height": 480,
                    "byte_size": 2048,
                }
            ],
        )
    output = StringIO()
    Console(file=output, width=width, height=height, color_system=None).print(
        render(trace, 0, False, False, 12, width=width, height=height)
    )
    rendered = output.getvalue()
    assert "240–255" in rendered
    assert "16×16" in rendered
    assert "PAUSED" in rendered
    assert "upstream cached: 128" in rendered
    if with_image:
        assert "child 4/6 (choice 3)" in rendered
        assert "Shared images: 1" in rendered
        if width >= 150:
            assert "SHARED IMAGES: 1 (prefill)" in rendered
            assert "image/png 640×480" in rendered
    lines = rendered.splitlines()
    grid_rows = [
        index for index, line in enumerate(lines) if re.search(r"\b\d{3}–\d{3}\b", line)
    ]
    assert len(grid_rows) == 16
    assert preview in lines[grid_rows[0]]
    assert grid_rows[1] - grid_rows[0] == cell_height
    assert len(lines) <= height - 1
    assert "q quit" in lines[-1]


@pytest.mark.parametrize("with_image", [False, True])
def test_larger_sidebar_reveals_more_context_and_candidates(tmp_path, with_image):
    """Extra terminal space shows more real content, not just wider borders."""
    labels = [f"category{index:02d}" for index in range(26)]
    trace = Trace(tmp_path / "synthetic.jsonl")
    trace.samples.append(
        {
            "canvas_tokens": [10] * 256,
            "canvas_text": ["fixed"] * 256,
            "state": "context " * 20 + "MORE_CONTEXT",
            "candidate_labels": labels,
            "candidate_token_ids": list(range(26)),
            "candidate_logprobs": [math.log(1 / 26)] * 26,
            "probabilities": [1 / 26] * 26,
        }
    )
    trace.samples[0].update(
        image_count=int(with_image),
        shared_images=[
            {
                "sha256": "a" * 64,
                "mime_type": "image/png",
                "width": 640,
                "height": 480,
                "byte_size": 2048,
            }
        ]
        if with_image
        else [],
    )
    for width, height, candidates in [(155, 38, 6), (188, 46, 8), (240, 60, 11)]:
        output = StringIO()
        Console(file=output, width=width, height=height, color_system=None).print(
            render(trace, 0, False, False, 0, width=width, height=height)
        )
        text = output.getvalue()
        assert f"Shared images: {int(with_image)}" in text
        assert f"showing {candidates} of 26" in text
        assert labels[candidates - 1] in text
        assert ("log P" in text) == (width >= 240)
        assert ("MORE_CONTEXT" in text) == (width > 155)
        assert len(text.splitlines()) <= height - 1
        assert "q quit" in text.splitlines()[-1]
