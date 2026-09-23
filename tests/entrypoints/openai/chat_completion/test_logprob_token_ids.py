# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the `logprob_token_ids` field on the OpenAI-compat
chat-completion endpoint.

`logprob_token_ids` lets a caller pin the set of vocab ids whose logprobs
should appear in the response, independent of where those ids would rank in
the natural top-k distribution. This is the primitive that multilabel
scoring postprocessors use to gather logprobs at a fixed small label
vocabulary (e.g. PII detection where each label corresponds to a known
digit-token vocab id).
"""

import math
from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import _pooled_answer_distribution
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import FlatLogprobs, Logprob

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _diffusion_read_request(
    *, answer_only_logprobs: bool = False, **overrides: Any
) -> ChatCompletionRequest:
    body = {
        "messages": [{"role": "user", "content": "Shared state"}],
        "n": 2,
        "logprobs": True,
        "max_tokens": 3,
        "ignore_eos": True,
        "diffusion_reads": [
            {
                "seed_canvas": [9, 100, 0],
                "answer_positions": [1],
                "logprob_token_ids": [10, 11],
                "answer_only_logprobs": answer_only_logprobs,
            },
            {
                "seed_canvas": [8, 200, 0],
                "answer_positions": [1],
                "logprob_token_ids": [20, 21, 22],
                "answer_only_logprobs": answer_only_logprobs,
            },
        ],
    }
    return ChatCompletionRequest.model_validate({**body, **overrides})


def _reasoning_mean_request():
    request = _diffusion_read_request(answer_only_logprobs=True)
    reads = [read.model_dump() for read in request.diffusion_reads]
    for read in reads:
        read.update(reasoning_positions=[0], reasoning_forbidden_token_ids=[], steps=4)
    return _diffusion_read_request(diffusion_reads=reads)


def test_answer_mean_changes_only_its_child_and_preserves_explicit_last():
    baseline = _reasoning_mean_request()
    original = baseline.to_child_sampling_params(baseline.to_sampling_params(3, {}))
    reads = [read.model_dump() for read in baseline.diffusion_reads]
    reads[1]["answer_pass_aggregation"] = "mean"
    request = _diffusion_read_request(diffusion_reads=reads)
    children = request.to_child_sampling_params(request.to_sampling_params(3, {}))
    assert children[0].extra_args == original[0].extra_args
    assert children[1].extra_args == {
        **original[1].extra_args,
        "diffusion_answer_pass_aggregation": "mean",
    }
    assert [child.logprob_token_ids for child in children] == [[10, 11], [20, 21, 22]]
    assert all(
        "diffusion_answer_pass_aggregation" not in child.extra_args
        for child in original
    )
    assert baseline.diffusion_reads[1].answer_pass_aggregation == "last"
    for read in reads:
        read.pop("answer_pass_aggregation")
    omitted = _diffusion_read_request(diffusion_reads=reads)
    omitted_children = omitted.to_child_sampling_params(
        omitted.to_sampling_params(3, {})
    )
    assert [child.extra_args for child in omitted_children] == [
        child.extra_args for child in original
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_convergence_stop_is_child_local_and_false_matches_omission(enabled):
    reads = [read.model_dump() for read in _diffusion_read_request().diffusion_reads]
    for read in reads:
        read.pop("reasoning_convergence_stop", None)
    reads[1].update(reasoning_positions=[2], reasoning_forbidden_token_ids=[])
    control = _diffusion_read_request(diffusion_reads=reads)
    expected = control.to_child_sampling_params(control.to_sampling_params(3, {}))
    changed = deepcopy(reads)
    changed[0]["reasoning_convergence_stop"] = False
    changed[1]["reasoning_convergence_stop"] = enabled
    before = deepcopy(changed)
    request = _diffusion_read_request(diffusion_reads=changed)
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    assert children[0] == expected[0]
    extra = dict(expected[1].extra_args)
    if enabled:
        extra["diffusion_reasoning_convergence_stop"] = True
    assert children[1].extra_args == extra
    assert children[1].logprob_token_ids == expected[1].logprob_token_ids
    assert parent.extra_args is None
    assert changed == before
    assert "diffusion_reasoning_convergence_stop" not in expected[1].extra_args


@pytest.mark.parametrize(
    "override",
    [
        {"answer_pass_aggregation": None},
        {"answer_pass_aggregation": True},
        {"answer_pass_aggregation": "sum"},
        {"reasoning_positions": None},
        {"answer_only_logprobs": False},
        {"reasoning_preserve_feedback": True},
    ],
)
def test_answer_mean_rejects_invalid_enum_or_phase_context(override):
    descriptor = _reasoning_mean_request().diffusion_reads[1].model_dump()
    descriptor.update(answer_pass_aggregation="mean")
    descriptor.update(override)
    with pytest.raises(ValidationError):
        _diffusion_read_request(n=1, diffusion_reads=[descriptor])


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
def test_answer_mean_cannot_be_mislabeled_through_scalar_endpoints(request_type):
    body: dict[str, Any] = (
        {"messages": []} if request_type is ChatCompletionRequest else {"prompt": "x"}
    )
    with pytest.raises(VLLMValidationError, match="diffusion_reads"):
        request_type(**body, vllm_xargs={"diffusion_answer_pass_aggregation": "mean"})


@pytest.mark.parametrize("flat", [False, True])
def test_pooled_answer_response_has_explicit_semantics_and_keeps_default_schema(flat):
    read = _reasoning_mean_request().diffusion_reads[1]
    read.answer_pass_aggregation = "mean"
    probabilities = [0.6, 0.3, 0.1]
    row = {
        token: Logprob(math.log(p))
        for token, p in zip(read.logprob_token_ids, probabilities)
    }
    tokens = [8, 20, 0]
    scores = FlatLogprobs() if flat else [None, row, None]
    if flat:
        scores.append(row)
    distribution = _pooled_answer_distribution(tokens, scores, read)
    assert distribution.probabilities == pytest.approx(probabilities)
    assert distribution.token_ids == read.logprob_token_ids
    choice = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(role="assistant"),
        token_ids=tokens,
        diffusion_answer_distribution=distribution,
    )
    serialized = choice.model_dump()
    assert serialized["logprobs"] is None
    assert serialized["diffusion_answer_distribution"]["aggregation"] == "mean"
    assert (
        serialized["diffusion_answer_distribution"]["normalization"]
        == "allowed_token_ids"
    )
    assert serialized["diffusion_answer_distribution"]["answer_passes"] == 4
    default = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant"),
        logprobs=ChatCompletionLogProbs(content=[]),
    ).model_dump()
    assert "diffusion_answer_distribution" not in default
    assert default["logprobs"] == {"content": []}


@pytest.mark.parametrize("corruption", ["missing", "unnormalized", "wrong_token"])
def test_pooled_answer_response_rejects_invalid_distribution(corruption):
    read = _reasoning_mean_request().diffusion_reads[1]
    row = {
        token: Logprob(math.log(p))
        for token, p in zip(read.logprob_token_ids, [0.6, 0.3, 0.1])
    }
    tokens = [8, 20, 0]
    if corruption == "missing":
        row.pop(21)
    elif corruption == "unnormalized":
        row[20] = Logprob(math.log(0.5))
    else:
        tokens[1] = 22
    with pytest.raises(ValueError, match="mean answer"):
        _pooled_answer_distribution(tokens, [row], read)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 1},
        {"diffusion_reads": []},
        {"logprobs": False},
        {"stream": True},
        {"use_beam_search": True},
        {"logprob_token_ids": [10]},
        {"logprob_token_ids": []},
        {"top_logprobs": 1},
        {"top_logprobs": -1},
        {"kv_transfer_params": {}},
        {"ec_transfer_params": {}},
        {"session_id": "session"},
        {"truncate_prompt_tokens": -1},
        {"truncate_prompt_tokens": 100},
        {"reasoning_effort": "high"},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": "false"}},
        {"chat_template_kwargs": {"enable_thinking": None}},
        {"vllm_xargs": {"diffusion_read_only": True}},
        {"vllm_xargs": {"diffusion_seed_canvas": [1, 2, 3]}},
        {"vllm_xargs": {"diffusion_answer_positions": [1]}},
        {"vllm_xargs": {"diffusion_answer_only_logprobs": True}},
        {"vllm_xargs": {"diffusion_read_steps": 2}},
        {"vllm_xargs": {"diffusion_read_self_conditioning": False}},
        {"vllm_xargs": {"diffusion_read_self_conditioning_scope": "canvas"}},
        {"vllm_xargs": {"diffusion_reasoning_positions": [0]}},
        {"vllm_xargs": {"diffusion_reasoning_forbidden_token_ids": []}},
        {"vllm_xargs": {"diffusion_reasoning_steps": 8}},
        {"vllm_xargs": {"diffusion_reasoning_seed": 0}},
        {"vllm_xargs": {"diffusion_reasoning_known_positions": [0]}},
        pytest.param(
            {"vllm_xargs": {"diffusion_reasoning_delayed_suffix_positions": [2]}},
            id="delayed_suffix_scalar_conflict",
        ),
        {"vllm_xargs": {"diffusion_reasoning_answer_noise": "vocab"}},
        {"vllm_xargs": {"diffusion_reasoning_token_noise": "vocab"}},
        {"vllm_xargs": {"diffusion_reasoning_token_proposals": "vocab"}},
        {"vllm_xargs": {"diffusion_reasoning_schedule_steps": 48}},
        {"vllm_xargs": {"diffusion_reasoning_entropy_bound": 0.1}},
        {"vllm_xargs": {"diffusion_reasoning_round_feedback_logits": True}},
        {"vllm_xargs": {"diffusion_reasoning_preserve_feedback": False}},
        {"vllm_xargs": {"diffusion_reasoning_answer_feedback": True}},
        {"vllm_xargs": {"diffusion_reasoning_unmasked_feedback": True}},
        pytest.param(
            {"vllm_xargs": {"diffusion_reasoning_convergence_stop": True}},
            id="convergence_stop_true_scalar_conflict",
        ),
        pytest.param(
            {"vllm_xargs": {"diffusion_reasoning_convergence_stop": False}},
            id="convergence_stop_false_scalar_conflict",
        ),
        {"vllm_xargs": {"diffusion_canvas_length": 3}},
        {"vllm_xargs": {"diffusion_max_steps": 1}},
        {"vllm_xargs": {"session_id": "session"}},
        {"vllm_xargs": {"kv_transfer_params": "transfer"}},
        {"vllm_xargs": {"ec_transfer_params": "transfer"}},
    ],
)
def test_diffusion_reads_reject_ambiguous_or_unsupported_requests(overrides):
    with pytest.raises((ValidationError, VLLMValidationError)):
        _diffusion_read_request(**overrides)


@pytest.mark.parametrize(
    "override",
    [
        {"seed_canvas": []},
        {"seed_canvas": [1, True, 0]},
        {"seed_canvas": [1, "2", 0]},
        {"seed_canvas": [1, 2.0, 0]},
        {"seed_canvas": [1, -1, 0]},
        {"answer_positions": []},
        {"answer_positions": [1, 2]},
        {"answer_positions": [3]},
        {"answer_positions": [True]},
        {"logprob_token_ids": []},
        {"logprob_token_ids": [10, 10]},
        {"logprob_token_ids": [False]},
        {"logprob_token_ids": list(range(129))},
        {"steps": 0},
        {"steps": 33},
        {"steps": True},
        {"steps": "2"},
        {"self_conditioning": 1},
        {"self_conditioning": "false"},
        {"self_conditioning_scope": "all"},
        {"self_conditioning_scope": True},
        {"self_conditioning_scope": None},
        {"reasoning_forbidden_token_ids": []},
        {"reasoning_steps": 8},
        {"reasoning_seed": 0},
        {"reasoning_known_positions": [0]},
        pytest.param(
            {"reasoning_delayed_suffix_positions": [2]},
            id="delayed_suffix_requires_reasoning",
        ),
        {"reasoning_answer_noise": "candidates"},
        {"reasoning_answer_noise": "vocab"},
        {"reasoning_token_noise": "allowed"},
        {"reasoning_token_proposals": "allowed"},
        {"reasoning_token_noise": "vocab"},
        {"reasoning_token_proposals": "vocab"},
        {"reasoning_schedule_steps": 8},
        {"reasoning_entropy_bound": 0},
        {"reasoning_round_feedback_logits": True},
        {"reasoning_preserve_feedback": True},
        {"reasoning_answer_feedback": True},
        {"reasoning_unmasked_feedback": True},
        pytest.param(
            {"reasoning_convergence_stop": True},
            id="convergence_stop_requires_reasoning",
        ),
        {"unexpected": "ignored?"},
    ],
)
def test_diffusion_reads_reject_malformed_single_answer_descriptors(override):
    descriptor = {
        "seed_canvas": [9, 100, 0],
        "answer_positions": [1],
        "logprob_token_ids": [10, 11],
    }
    with pytest.raises(ValidationError):
        _diffusion_read_request(n=1, diffusion_reads=[{**descriptor, **override}])


def test_diffusion_reads_build_independent_child_parameters():
    request = _diffusion_read_request(vllm_xargs={"custom": [7]})
    request.diffusion_reads[1].steps = 3
    request.diffusion_reads[1].self_conditioning = False
    request.diffusion_reads[1].self_conditioning_scope = "canvas"
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    assert children is not None
    assert parent.n == 2
    assert parent.logprob_token_ids is None
    assert parent.extra_args == {"custom": [7]}
    assert [child.n for child in children] == [1, 1]
    assert all(child.detokenize for child in children)
    assert all(not child.flat_logprobs for child in children)
    assert [child.logprob_token_ids for child in children] == [[10, 11], [20, 21, 22]]
    assert all(child.logprobs is None for child in children)
    assert all(child.extra_args["diffusion_read_only"] for child in children)
    assert [child.extra_args["diffusion_read_steps"] for child in children] == [1, 3]
    assert [
        child.extra_args["diffusion_read_self_conditioning"] for child in children
    ] == [True, False]
    assert [
        child.extra_args["diffusion_read_self_conditioning_scope"] for child in children
    ] == ["answer", "canvas"]
    assert all(
        not child.extra_args["diffusion_answer_only_logprobs"] for child in children
    )

    children[0].extra_args["custom"][0] = 99
    children[0].extra_args["diffusion_seed_canvas"][1] = 300
    children[0].logprob_token_ids.append(12)
    assert children[1].extra_args["custom"] == [7]
    assert parent.extra_args == {"custom": [7]}
    assert request.diffusion_reads[0].seed_canvas == [9, 100, 0]
    assert request.diffusion_reads[0].logprob_token_ids == [10, 11]

    token_id_request = _diffusion_read_request(return_tokens_as_token_ids=True)
    token_id_children = token_id_request.to_child_sampling_params(
        token_id_request.to_sampling_params(3, {})
    )
    assert token_id_children is not None
    assert all(child.detokenize for child in token_id_children)
    assert all(not child.flat_logprobs for child in token_id_children)

    optimized_request = _diffusion_read_request(
        answer_only_logprobs=True, return_tokens_as_token_ids=True
    )
    optimized_children = optimized_request.to_child_sampling_params(
        optimized_request.to_sampling_params(3, {})
    )
    assert optimized_children is not None
    assert all(not child.detokenize for child in optimized_children)
    assert all(child.flat_logprobs for child in optimized_children)
    assert all(
        child.extra_args["diffusion_answer_only_logprobs"]
        for child in optimized_children
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning_positions": None},
        {"reasoning_positions": []},
        {"reasoning_positions": [0, 0]},
        {"reasoning_positions": [1, 0]},
        {"reasoning_positions": [2, 3]},
        {"reasoning_positions": [4]},
        {"reasoning_positions": [True]},
        {"reasoning_positions": [0.0]},
        {"reasoning_known_positions": []},
        pytest.param(
            {"reasoning_delayed_suffix_positions": []},
            id="delayed_suffix_empty",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [2, 2]},
            id="delayed_suffix_duplicate",
        ),
        pytest.param(
            {"reasoning_positions": [0], "reasoning_delayed_suffix_positions": [2, 1]},
            id="delayed_suffix_unsorted",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [0]},
            id="delayed_suffix_reason_overlap",
        ),
        pytest.param(
            {"reasoning_positions": [0, 2], "reasoning_delayed_suffix_positions": [1]},
            id="delayed_suffix_fixed_gap_before_last_reason",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [3]},
            id="delayed_suffix_answer_overlap",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [4]},
            id="delayed_suffix_out_of_range",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [-1]},
            id="delayed_suffix_negative",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [True]},
            id="delayed_suffix_bool",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": [2.0]},
            id="delayed_suffix_float",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": ["2"]},
            id="delayed_suffix_string",
        ),
        pytest.param(
            {"reasoning_delayed_suffix_positions": (2,)},
            id="delayed_suffix_tuple",
        ),
        {"reasoning_known_positions": [2, 2]},
        {"reasoning_known_positions": [2, 0]},
        {"reasoning_known_positions": [0]},
        {"reasoning_known_positions": [3]},
        {"reasoning_known_positions": [4]},
        {"reasoning_known_positions": [-1]},
        {"reasoning_known_positions": [True]},
        {"reasoning_known_positions": [2.0]},
        {"reasoning_known_positions": (2,)},
        {"reasoning_forbidden_token_ids": None},
        {"reasoning_forbidden_token_ids": [0, 0]},
        {"reasoning_forbidden_token_ids": [-1]},
        {"reasoning_forbidden_token_ids": [True]},
        {"reasoning_steps": 0},
        {"reasoning_steps": 65},
        {"reasoning_steps": True},
        {"reasoning_steps": "8"},
        {"reasoning_seed": -1},
        {"reasoning_seed": 2**64},
        {"reasoning_seed": True},
        {"reasoning_seed": False},
        {"reasoning_seed": 1.0},
        {"reasoning_seed": "1"},
        {"reasoning_answer_noise": "all"},
        {"reasoning_answer_noise": True},
        {"reasoning_answer_noise": 0},
        {"reasoning_answer_noise": []},
        {"reasoning_token_noise": "candidates"},
        {"reasoning_token_proposals": "candidates"},
        {"reasoning_token_noise": True},
        {"reasoning_token_proposals": True},
        {"reasoning_token_noise": 0},
        {"reasoning_token_proposals": 0},
        {"reasoning_token_noise": []},
        {"reasoning_token_proposals": []},
        {"reasoning_token_noise": b"vocab"},
        {"reasoning_token_proposals": b"vocab"},
        {"reasoning_schedule_steps": 0},
        {"reasoning_schedule_steps": 7},
        {"reasoning_schedule_steps": True},
        {"reasoning_schedule_steps": "8"},
        {"reasoning_schedule_steps": 8.0},
        {"reasoning_steps": 32, "reasoning_schedule_steps": 31},
        {"reasoning_steps": 64, "reasoning_schedule_steps": 63},
        {"reasoning_entropy_bound": -0.1},
        {"reasoning_entropy_bound": True},
        {"reasoning_entropy_bound": "0.1"},
        {"reasoning_entropy_bound": float("nan")},
        {"reasoning_entropy_bound": float("inf")},
        {"reasoning_entropy_bound": float("-inf")},
    ],
)
def test_reasoning_descriptors_reject_invalid_span_or_constraints(overrides):
    descriptor = {
        "seed_canvas": [9, 8, 7, 100],
        "answer_positions": [3],
        "logprob_token_ids": [10, 11],
        "reasoning_positions": [0, 1],
        "reasoning_forbidden_token_ids": [0, 2],
    }
    with pytest.raises(ValidationError):
        _diffusion_read_request(n=1, diffusion_reads=[{**descriptor, **overrides}])


@pytest.mark.parametrize("steps", [None, 1, 32, 48, 64])
@pytest.mark.parametrize("answer_noise", [None, "candidates", "vocab"])
@pytest.mark.parametrize("positions", [[2], [0, 2]])
def test_reasoning_children_keep_independent_constraints_and_default_omission(
    steps, answer_noise, positions
):
    request = _diffusion_read_request()
    descriptor = request.diffusion_reads[1].model_dump()
    descriptor.update(
        reasoning_positions=positions,
        reasoning_forbidden_token_ids=[],
        reasoning_steps=steps,
        reasoning_answer_noise=answer_noise,
    )
    request = _diffusion_read_request(
        diffusion_reads=[request.diffusion_reads[0].model_dump(), descriptor]
    )
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    assert children is not None
    assert not any(
        key.startswith("diffusion_reasoning_") for key in children[0].extra_args
    )
    extra = children[1].extra_args
    assert extra["diffusion_reasoning_steps"] == (8 if steps is None else steps)
    assert extra["diffusion_reasoning_answer_noise"] == (answer_noise or "candidates")
    assert "diffusion_reasoning_schedule_steps" not in extra
    assert "diffusion_reasoning_seed" not in extra
    assert "diffusion_reasoning_entropy_bound" not in extra
    assert "diffusion_reasoning_round_feedback_logits" not in extra
    assert "diffusion_reasoning_preserve_feedback" not in extra
    assert "diffusion_reasoning_answer_feedback" not in extra
    assert "diffusion_reasoning_unmasked_feedback" not in extra
    assert "diffusion_reasoning_token_noise" not in extra
    assert "diffusion_reasoning_token_proposals" not in extra
    assert extra["diffusion_read_steps"] == 1
    assert children[1].logprob_token_ids == [20, 21, 22]
    assert extra["diffusion_reasoning_positions"] == positions
    assert extra["diffusion_reasoning_forbidden_token_ids"] == []
    extra["diffusion_reasoning_positions"][0] = 1
    extra["diffusion_reasoning_forbidden_token_ids"].append(99)
    assert request.diffusion_reads[1].reasoning_positions == positions
    assert request.diffusion_reads[1].reasoning_forbidden_token_ids == []
    assert parent.extra_args is None


@pytest.mark.parametrize("seed", [None, 0, 2**64 - 1])
def test_reasoning_seed_is_local_to_child_and_null_keeps_default(seed):
    request = _diffusion_read_request()
    ordinary, reasoning = [read.model_dump() for read in request.diffusion_reads]
    ordinary["reasoning_seed"] = None
    reasoning.update(reasoning_positions=[2], reasoning_forbidden_token_ids=[])
    baseline = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    baseline_children = baseline.to_child_sampling_params(
        baseline.to_sampling_params(3, {})
    )
    changed = _diffusion_read_request(
        diffusion_reads=[ordinary, {**reasoning, "reasoning_seed": seed}]
    )
    parent = changed.to_sampling_params(3, {})
    children = changed.to_child_sampling_params(parent)
    assert parent.extra_args is None
    assert children[0].extra_args == baseline_children[0].extra_args
    expected = dict(baseline_children[1].extra_args)
    if seed is not None:
        expected["diffusion_reasoning_seed"] = seed
    assert children[1].extra_args == expected
    assert changed.diffusion_reads[1].reasoning_seed == seed
    assert "reasoning_seed" not in reasoning or reasoning["reasoning_seed"] is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("seed", None),
        ("seed", 0),
        ("seed", 2**64 - 1),
        ("known_positions", None),
        ("known_positions", [0, 2]),
        ("delayed_suffix_positions", None),
        ("delayed_suffix_positions", [2]),
    ],
)
@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
def test_scalar_reasoning_control_preserves_type_and_does_not_mutate_xargs(
    name, value, request_type
):
    name = f"diffusion_reasoning_{name}"
    xargs = {name: value, "unrelated": [1, "text"]}
    original = deepcopy(xargs)
    body = (
        {"messages": [{"role": "user", "content": "State"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "State"}
    )
    request = request_type(**body, vllm_xargs=xargs)
    assert xargs == original
    expected = {"unrelated": [1, "text"]} if value is None else xargs
    assert request.vllm_xargs == expected
    assert request.to_sampling_params(3, {}).extra_args == expected
    assert xargs == original
    if value is not None:
        assert type(request.vllm_xargs[name]) is type(value)


@pytest.mark.parametrize("positions", [None, [0]])
def test_reasoning_known_positions_remain_local_to_the_selected_child(positions):
    request = _diffusion_read_request()
    ordinary, reasoning = [read.model_dump() for read in request.diffusion_reads]
    reasoning.update(
        reasoning_positions=[2],
        reasoning_forbidden_token_ids=[],
        reasoning_known_positions=positions,
    )
    request = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    name = "diffusion_reasoning_known_positions"
    assert name not in children[0].extra_args
    assert parent.extra_args is None
    if positions is None:
        assert name not in children[1].extra_args
    else:
        assert children[1].extra_args[name] == positions
        children[1].extra_args[name].append(1)
        assert request.diffusion_reads[1].reasoning_known_positions == positions


@pytest.mark.parametrize("positions", [True, 2, "2", (2,), [True], [2.0], ["2"]])
@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
@pytest.mark.parametrize("name", ["known_positions", "delayed_suffix_positions"])
def test_scalar_known_positions_reject_type_coercion(positions, request_type, name):
    body = (
        {"messages": [{"role": "user", "content": "State"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "State"}
    )
    name = f"diffusion_reasoning_{name}"
    with pytest.raises(VLLMValidationError, match=name):
        request_type(**body, vllm_xargs={name: positions})


@pytest.mark.parametrize("positions", [None, [2]])
def test_reasoning_delayed_suffix_positions_remain_local_to_one_child(positions):
    ordinary, reasoning = [
        read.model_dump() for read in _diffusion_read_request().diffusion_reads
    ]
    ordinary.pop("reasoning_delayed_suffix_positions", None)
    reasoning.pop("reasoning_delayed_suffix_positions", None)
    reasoning.update(reasoning_positions=[0], reasoning_forbidden_token_ids=[])
    baseline = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    baseline_children = baseline.to_child_sampling_params(
        baseline.to_sampling_params(3, {})
    )
    changed = {**reasoning, "reasoning_delayed_suffix_positions": positions}
    descriptors = [ordinary, changed]
    original = deepcopy(descriptors)
    request = _diffusion_read_request(diffusion_reads=descriptors)
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    assert parent.extra_args is None
    assert children[0] == baseline_children[0]
    expected = deepcopy(baseline_children[1])
    name = "diffusion_reasoning_delayed_suffix_positions"
    if positions is not None:
        expected.extra_args[name] = positions
    assert children[1] == expected
    assert descriptors == original
    if positions is not None:
        children[1].extra_args[name].append(1)
        assert request.diffusion_reads[1].reasoning_delayed_suffix_positions == [2]
        assert descriptors == original


@pytest.mark.parametrize("seed", [-1, 2**64, False, True, 1.0, "1"])
@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
def test_scalar_reasoning_seed_rejects_coercion_and_overflow(seed, request_type):
    body = (
        {"messages": [{"role": "user", "content": "State"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "State"}
    )
    with pytest.raises(VLLMValidationError, match="diffusion_reasoning_seed"):
        request_type(**body, vllm_xargs={"diffusion_reasoning_seed": seed})


@pytest.mark.parametrize(
    ("steps", "schedule", "entropy"),
    [
        (None, None, None),
        (None, 8, 0),
        (1, 1, 0.1),
        (32, 64, 2),
        (48, 48, 0.1),
        (64, 64, 1),
        (8, None, 1),
        (8, 48, None),
    ],
)
def test_reasoning_children_forward_only_explicit_schedule_and_entropy(
    steps, schedule, entropy
):
    """Optional controls affect only their child and leave omitted defaults intact."""
    request = _diffusion_read_request()
    ordinary, reasoning = [read.model_dump() for read in request.diffusion_reads]
    reasoning.update(
        reasoning_positions=[2],
        reasoning_forbidden_token_ids=[],
        reasoning_steps=steps,
    )
    baseline = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    baseline_children = baseline.to_child_sampling_params(
        baseline.to_sampling_params(3, {})
    )
    changed = _diffusion_read_request(
        diffusion_reads=[
            ordinary,
            {
                **reasoning,
                "reasoning_schedule_steps": schedule,
                "reasoning_entropy_bound": entropy,
            },
        ]
    )
    parent = changed.to_sampling_params(3, {})
    children = changed.to_child_sampling_params(parent)
    assert children is not None and baseline_children is not None
    assert parent.extra_args is None
    assert children[0].extra_args == baseline_children[0].extra_args
    expected = dict(baseline_children[1].extra_args)
    if schedule is not None:
        expected["diffusion_reasoning_schedule_steps"] = schedule
    if entropy is not None:
        expected["diffusion_reasoning_entropy_bound"] = entropy
    assert children[1].extra_args == expected
    if schedule is entropy is None:
        assert changed.model_dump(
            exclude_none=True, exclude={"request_id"}
        ) == baseline.model_dump(exclude_none=True, exclude={"request_id"})


@pytest.mark.parametrize(
    "name", ["diffusion_reasoning_schedule_steps", "diffusion_reasoning_entropy_bound"]
)
@pytest.mark.parametrize("value", [False, True])
def test_reasoning_scalar_controls_reject_booleans_before_xargs_coercion(name, value):
    with pytest.raises(VLLMValidationError, match="not boolean"):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Shared state"}],
            vllm_xargs={name: value},
        )


@pytest.mark.parametrize("round_logits", [False, True])
@pytest.mark.parametrize("preserve_feedback", [False, True])
@pytest.mark.parametrize("answer_feedback", [False, True])
@pytest.mark.parametrize("unmasked_feedback", [False, True])
def test_reasoning_children_forward_only_enabled_feedback_controls(
    round_logits, preserve_feedback, answer_feedback, unmasked_feedback
):
    """Feedback experiments change only enabled child extras, never default seeds."""
    request = _diffusion_read_request()
    ordinary, reasoning = [read.model_dump() for read in request.diffusion_reads]
    reasoning.update(reasoning_positions=[2], reasoning_forbidden_token_ids=[])
    baseline = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    expected = baseline.to_child_sampling_params(baseline.to_sampling_params(3, {}))
    reasoning.update(
        reasoning_round_feedback_logits=round_logits,
        reasoning_preserve_feedback=preserve_feedback,
        reasoning_answer_feedback=answer_feedback,
        reasoning_unmasked_feedback=unmasked_feedback,
    )
    changed = _diffusion_read_request(diffusion_reads=[ordinary, reasoning])
    parent = changed.to_sampling_params(3, {})
    actual = changed.to_child_sampling_params(parent)
    assert actual is not None and expected is not None
    assert parent.extra_args is None
    assert actual[0].extra_args == expected[0].extra_args
    extras = dict(expected[1].extra_args)
    for name, enabled in (
        ("diffusion_reasoning_round_feedback_logits", round_logits),
        ("diffusion_reasoning_preserve_feedback", preserve_feedback),
        ("diffusion_reasoning_answer_feedback", answer_feedback),
        ("diffusion_reasoning_unmasked_feedback", unmasked_feedback),
    ):
        assert name not in extras
        if enabled:
            extras[name] = True
    assert actual[1].extra_args == extras


@pytest.mark.parametrize(
    "name",
    [
        "reasoning_round_feedback_logits",
        "reasoning_preserve_feedback",
        "reasoning_answer_feedback",
        "reasoning_unmasked_feedback",
        "reasoning_convergence_stop",
    ],
)
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_reasoning_feedback_descriptors_require_strict_booleans(name, value):
    descriptor = {
        "seed_canvas": [1, 2, 3],
        "answer_positions": [1],
        "logprob_token_ids": [10, 11],
        "reasoning_positions": [2],
        "reasoning_forbidden_token_ids": [],
        name: value,
    }
    with pytest.raises(ValidationError):
        _diffusion_read_request(n=1, diffusion_reads=[descriptor])


@pytest.mark.parametrize("name", ["token_noise", "token_proposals"])
@pytest.mark.parametrize("mode", [None, "allowed", "vocab"])
def test_reasoning_token_modes_are_child_local_and_defaults_stay_omitted(name, mode):
    request = _diffusion_read_request()
    descriptor = request.diffusion_reads[1].model_dump()
    descriptor.update(reasoning_positions=[2], reasoning_forbidden_token_ids=[])
    control = _diffusion_read_request(diffusion_reads=[descriptor, descriptor])
    request = _diffusion_read_request(
        diffusion_reads=[descriptor, {**descriptor, f"reasoning_{name}": mode}]
    )
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    baseline = control.to_child_sampling_params(control.to_sampling_params(3, {}))
    assert children[0] == baseline[0]
    expected = dict(baseline[1].extra_args)
    if mode == "vocab":
        expected[f"diffusion_reasoning_{name}"] = mode
    assert children[1].extra_args == expected
    assert parent.extra_args is None
    assert (
        f"reasoning_{name}" not in descriptor or descriptor[f"reasoning_{name}"] is None
    )


@pytest.mark.parametrize("name", ["token_noise", "token_proposals"])
@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
@pytest.mark.parametrize(
    "mode", ["allowed", "vocab", None, True, 0, "all", b"vocab", []]
)
def test_scalar_reasoning_token_modes_accept_only_exact_enum(request_type, name, mode):
    body = (
        {"messages": [{"role": "user", "content": "Shared state"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "Shared state"}
    )
    extra = {f"diffusion_reasoning_{name}": mode, "custom": 7}
    if type(mode) is not str or mode not in ("allowed", "vocab"):
        with pytest.raises((ValidationError, VLLMValidationError)):
            request_type(**body, vllm_xargs=extra)
    else:
        request = request_type(**body, vllm_xargs=extra)
        assert request.to_sampling_params(3, {}).extra_args == extra
    assert extra == {f"diffusion_reasoning_{name}": mode, "custom": 7}


@pytest.mark.parametrize(
    "request_type,name",
    [
        (ChatCompletionRequest, name)
        for name in (
            "diffusion_reasoning_round_feedback_logits",
            "diffusion_reasoning_preserve_feedback",
            "diffusion_reasoning_answer_feedback",
            "diffusion_reasoning_unmasked_feedback",
            "diffusion_reasoning_convergence_stop",
        )
    ]
    + [
        (CompletionRequest, name)
        for name in (
            "diffusion_reasoning_unmasked_feedback",
            "diffusion_reasoning_convergence_stop",
        )
    ],
)
@pytest.mark.parametrize("value", [False, True])
def test_scalar_reasoning_feedback_preserves_booleans_through_sampling_params(
    request_type, name, value
):
    """Scoped feedback booleans survive the numeric custom-extension value schema."""
    body = (
        {"messages": [{"role": "user", "content": "Shared state"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "Shared state"}
    )
    original = {name: value, "custom": 7}
    request = request_type(**body, vllm_xargs=original)
    params = request.to_sampling_params(3, {})
    assert request.vllm_xargs[name] is value
    assert params.extra_args[name] is value
    assert request.model_dump()["vllm_xargs"][name] is value
    assert params.extra_args["custom"] == 7
    assert original == {name: value, "custom": 7}
    assert original[name] is value


@pytest.mark.parametrize(
    "request_type,name",
    [
        (ChatCompletionRequest, name)
        for name in (
            "diffusion_reasoning_round_feedback_logits",
            "diffusion_reasoning_preserve_feedback",
            "diffusion_reasoning_answer_feedback",
            "diffusion_reasoning_unmasked_feedback",
            "diffusion_reasoning_convergence_stop",
        )
    ]
    + [
        (CompletionRequest, name)
        for name in (
            "diffusion_reasoning_unmasked_feedback",
            "diffusion_reasoning_convergence_stop",
        )
    ],
)
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_scalar_reasoning_feedback_rejects_boolean_coercion(request_type, name, value):
    body = (
        {"messages": [{"role": "user", "content": "Shared state"}]}
        if request_type is ChatCompletionRequest
        else {"prompt": "Shared state"}
    )
    with pytest.raises(VLLMValidationError, match="must be a boolean"):
        request_type(**body, vllm_xargs={name: value})


@pytest.mark.parametrize("answer_only", [False, True])
def test_diffusion_reads_give_each_child_its_physical_canvas_budget(answer_only):
    request = _diffusion_read_request(
        answer_only_logprobs=answer_only, return_tokens_as_token_ids=True
    )
    request.diffusion_reads[1].seed_canvas = [8, 200]
    parent = request.to_sampling_params(3, {})
    children = request.to_child_sampling_params(parent)
    assert children is not None
    assert parent.max_tokens == 3
    assert [child.max_tokens for child in children] == [3, 2]
    assert all(child.flat_logprobs == answer_only for child in children)
    assert all(child.detokenize != answer_only for child in children)
    assert all(
        child.extra_args["diffusion_answer_only_logprobs"] == answer_only
        for child in children
    )


@pytest.mark.parametrize(
    "kwargs", [None, {}, {"custom": "keep"}, {"enable_thinking": False}]
)
def test_diffusion_reads_disable_thinking_without_mutating_template_kwargs(kwargs):
    request = _diffusion_read_request(chat_template_kwargs=kwargs)
    params = request.build_chat_params(None, "auto").with_defaults(
        {"enable_thinking": True}
    )
    assert params.chat_template_kwargs["enable_thinking"] is False
    assert request.chat_template_kwargs == kwargs
    if kwargs:
        assert all(
            params.chat_template_kwargs[key] == value for key, value in kwargs.items()
        )


def test_diffusion_reads_reject_incomplete_canvas_budget():
    request = _diffusion_read_request()
    with pytest.raises(VLLMValidationError, match="complete canvas"):
        request.to_child_sampling_params(request.to_sampling_params(2, {}))


def test_legacy_requests_do_not_build_explicit_children():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "x"}])
    assert request.to_child_sampling_params(request.to_sampling_params(3, {})) is None


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _logprob_entries(resp) -> list:
    """Return the top_logprobs list from the first generated position."""
    return resp.choices[0].logprobs.content[0].top_logprobs


def _token_id(token: str) -> int:
    assert token.startswith("token_id:"), (
        "expected return_tokens_as_token_ids=True to yield the "
        f"`token_id:<int>` form, got {token!r}"
    )
    return int(token.removeprefix("token_id:"))


def _top_logprob_token_ids(resp) -> list[int]:
    return [_token_id(entry.token) for entry in _logprob_entries(resp)]


def test_chat_request_decouples_top_k_from_explicit_token_ids():
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello"}],
        logprobs=True,
        top_logprobs=5,
        logprob_token_ids=[5000],
    )

    sampling_params = request.to_sampling_params(
        max_tokens=1, default_sampling_params={}
    )

    assert sampling_params.logprobs is None
    assert sampling_params.logprob_token_ids == [5000]
    assert sampling_params.num_logprobs == 1


def test_completion_request_decouples_top_k_from_explicit_token_ids():
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Hello",
        logprobs=5,
        logprob_token_ids=[5000],
    )

    sampling_params = request.to_sampling_params(max_tokens=1)

    assert sampling_params.logprobs is None
    assert sampling_params.logprob_token_ids == [5000]
    assert sampling_params.num_logprobs == 1


def test_completion_rejects_explicit_token_ids_without_generated_tokens():
    with pytest.raises(VLLMValidationError, match="no output tokens are generated"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Hello",
            echo=True,
            max_tokens=0,
            logprobs=5,
            logprob_token_ids=[5000],
        )


def test_requests_reject_explicit_token_ids_with_beam_search():
    with pytest.raises(VLLMValidationError, match="not supported with beam search"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            logprobs=True,
            logprob_token_ids=[5000],
            use_beam_search=True,
        )

    with pytest.raises(VLLMValidationError, match="not supported with beam search"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Hello",
            logprobs=5,
            logprob_token_ids=[5000],
            use_beam_search=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_ids", "top_logprobs", "sampled_id"),
    [
        pytest.param([5000], 5, 42, id="original-issue-shape"),
        pytest.param([100, 1000, 5000], 3, 42, id="sampled-outside-set"),
        pytest.param([100, 1000, 5000], 3, 5000, id="sampled-inside-set"),
    ],
)
async def test_logprob_token_ids_returns_requested_ids(
    server, requested_ids, top_logprobs, sampled_id
):
    async with server.get_async_client() as client:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body={
                "logprob_token_ids": requested_ids,
                "allowed_token_ids": [sampled_id],
                "return_tokens_as_token_ids": True,
            },
        )
        entries = _logprob_entries(resp)
        returned_token_ids = _top_logprob_token_ids(resp)

        assert _token_id(resp.choices[0].logprobs.content[0].token) == sampled_id
        assert set(returned_token_ids) == {*requested_ids, sampled_id}
        assert len(returned_token_ids) == len(set(returned_token_ids))

        for e in entries:
            assert isinstance(e.logprob, float)
            assert not math.isnan(e.logprob)
            assert not math.isinf(e.logprob)


@pytest.mark.asyncio
async def test_logprob_token_ids_stream_with_no_top_k(server):
    async with server.get_async_client() as client:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=None,
            stream=True,
            extra_body={
                "logprob_token_ids": [5000],
                "allowed_token_ids": [42],
                "return_tokens_as_token_ids": True,
            },
        )

        returned_token_ids: list[int] = []
        async for chunk in stream:
            if not chunk.choices or chunk.choices[0].logprobs is None:
                continue
            for content in chunk.choices[0].logprobs.content:
                returned_token_ids.extend(
                    _token_id(entry.token) for entry in content.top_logprobs
                )

    assert set(returned_token_ids) == {42, 5000}


@pytest.mark.asyncio
async def test_logprob_token_ids_default_behavior_unchanged(server):
    """Without `logprob_token_ids`, the response carries the natural top-k
    most-likely tokens. This guards against the new field accidentally
    changing the default-path output."""
    async with server.get_async_client() as client:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        entries = _logprob_entries(resp)
        assert len(entries) == 5
        # Default mode emits log_softmax, so all values are <= 0.
        for e in entries:
            assert e.logprob <= 0.0
