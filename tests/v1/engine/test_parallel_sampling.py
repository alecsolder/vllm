# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest


def test_parent_request_to_output_stream() -> None:
    parent_request = ParentRequest(make_request(SamplingParams(n=2)))
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)

    # output_1 finished
    output_1.finish_reason = "ended"
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    # Finished output_1 had already returned, DO NOT returned again
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)

    # output_0 finished
    output_0.finish_reason = "ended"
    assert ([output_0], True) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)
    # Finished output_0 had already returned, DO NOT returned again
    assert parent_request.get_outputs("child_id_0", output_0) == ([], True)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)


def test_parent_request_to_output_final_only() -> None:
    parent_request = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY))
    )
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished, return nothing
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_1 finished, but outputs won't be returned until all child requests finished
    output_1.finish_reason = "ended"
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_0 finished, as all child requests finished, the output would be returned
    output_0.finish_reason = "ended"
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_1", output_1
    )


def test_parallel_sampling_child_requests_preserve_session_id() -> None:
    request = make_request(SamplingParams(n=2))
    request.session_id = "session-1"
    parent_request = ParentRequest(request)

    for idx in range(parent_request.n):
        request_id, child_params = parent_request.get_child_info(idx)
        child_request = request if idx == parent_request.n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = child_params

        assert child_request.session_id == "session-1"


def test_prepared_children_do_not_alias_nested_parameters() -> None:
    child = SamplingParams(
        extra_args={"canvas": [1, 2]}, logprob_token_ids=[3, 4], skip_clone=True
    )
    parent = ParentRequest(make_request(SamplingParams(n=2)), [child, child])
    first = parent.get_child_info(0)[1]
    second = parent.get_child_info(1)[1]
    first.extra_args["canvas"][0] = 9
    first.logprob_token_ids.append(5)
    assert second.extra_args == child.extra_args == {"canvas": [1, 2]}
    assert second.logprob_token_ids == child.logprob_token_ids == [3, 4]
    assert first.seed is second.seed is None


def test_prepared_parent_waits_for_children_not_yet_looked_up() -> None:
    parent = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY)),
        [SamplingParams(), SamplingParams()],
    )
    first_id, _ = parent.get_child_info(0)
    first = CompletionOutput(0, "a", [1], None, None, finish_reason="length")
    second = CompletionOutput(1, "b", [2], None, None, finish_reason="length")
    assert parent.get_outputs(first_id, first) == ([], False)
    assert parent.get_outputs(parent.get_child_info(1)[0], second) == (
        [first, second],
        True,
    )
    # Looking up a completed child must not re-add it to the pending set.
    parent.get_child_info(0)
    assert not parent.child_requests


@pytest.mark.parametrize("children", [[], [SamplingParams(n=2)]])
def test_prepared_parent_rejects_invalid_child_count(children) -> None:
    with pytest.raises(ValueError, match="n=1"):
        ParentRequest(make_request(SamplingParams(n=2)), children)


def _group_fixture():
    from vllm.v1.engine import EngineCoreRequest

    children = [
        SamplingParams(detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
        for _ in range(3)
    ]
    request = EngineCoreRequest(
        request_id="parent",
        external_req_id="external",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(
            n=3, detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    engine = SimpleNamespace(
        output_processor=OutputProcessor(None, log_stats=False),
        check_admission=MagicMock(),
        engine_core=SimpleNamespace(add_requests_async=AsyncMock()),
    )
    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request.request_id)
    return engine, request, children, queue


@pytest.mark.asyncio
async def test_group_registers_every_child_before_fast_out_of_order_completion():
    from vllm.v1.engine import EngineCoreOutput, FinishReason

    engine, request, children, queue = _group_fixture()

    async def finish(requests):
        assert set(engine.output_processor.request_states) == {
            "0_parent",
            "1_parent",
            "2_parent",
        }
        assert all(
            child.prompt_token_ids is request.prompt_token_ids for child in requests
        )
        for index in (2, 0, 1):
            engine.output_processor.process_outputs(
                [
                    EngineCoreOutput(
                        request_id=f"{index}_parent",
                        new_token_ids=[index],
                        finish_reason=FinishReason.LENGTH,
                    )
                ]
            )
            if index != 1:
                assert queue.get_nowait() is None

    engine.engine_core.add_requests_async.side_effect = finish
    await AsyncLLM._add_request_group(engine, request, None, children, queue)
    output = queue.get_nowait()
    assert output.finished
    assert [child.index for child in output.outputs] == [0, 1, 2]
    assert not engine.output_processor.request_states
    assert not engine.output_processor.parent_requests
    engine.check_admission.assert_called_once_with(3, "parent")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("rejected"), asyncio.CancelledError()])
async def test_group_submission_failure_removes_all_local_registrations(error):
    engine, request, children, queue = _group_fixture()
    engine.engine_core.add_requests_async.side_effect = error
    with pytest.raises(type(error)):
        await AsyncLLM._add_request_group(engine, request, None, children, queue)
    assert not engine.output_processor.request_states
    assert not engine.output_processor.parent_requests
    assert not engine.output_processor.external_req_ids


@pytest.mark.asyncio
async def test_group_registration_failure_rolls_back_without_submission(monkeypatch):
    engine, request, children, queue = _group_fixture()
    register = engine.output_processor.add_request

    def fail_second(child, *args):
        if child.request_id == "1_parent":
            raise ValueError("bad child")
        register(child, *args)

    monkeypatch.setattr(engine.output_processor, "add_request", fail_second)
    with pytest.raises(ValueError, match="bad child"):
        await AsyncLLM._add_request_group(engine, request, None, children, queue)
    engine.engine_core.add_requests_async.assert_not_awaited()
    assert not engine.output_processor.request_states
    assert not engine.output_processor.parent_requests
    assert not engine.output_processor.external_req_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_group_abort_by_parent_or_external_id_cleans_all_children(internal):
    engine, request, children, queue = _group_fixture()
    await AsyncLLM._add_request_group(engine, request, None, children, queue)
    ids = engine.output_processor.abort_requests(
        [request.request_id if internal else request.external_req_id], internal=internal
    )
    assert set(ids) == {"0_parent", "1_parent", "2_parent"}
    assert not engine.output_processor.request_states
    assert not engine.output_processor.parent_requests
    assert not engine.output_processor.external_req_ids


@pytest.mark.asyncio
async def test_group_collision_preserves_existing_request():
    from copy import copy

    engine, request, children, queue = _group_fixture()
    existing = copy(request)
    existing.request_id = "1_parent"
    existing.sampling_params = children[0]
    engine.output_processor.add_request(existing, None)
    state = engine.output_processor.request_states[existing.request_id]
    with pytest.raises(VLLMValidationError, match="already registered"):
        await AsyncLLM._add_request_group(engine, request, None, children, queue)
    assert engine.output_processor.request_states == {"1_parent": state}
    engine.engine_core.add_requests_async.assert_not_awaited()


def make_request(sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent_id",
        external_req_id="ext_parent_id",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
