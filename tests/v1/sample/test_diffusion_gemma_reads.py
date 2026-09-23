# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constrained decision reads and their interaction with ordinary diffusion."""

import hashlib
import json
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models import diffusion_gemma
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionGemmaRequestStates,
    DiffusionSampler,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DiffusionGemma sampler requires CUDA"
)

CL = 4
VOCAB = 16
DEVICE = torch.device("cuda")


@pytest.fixture(params=[False, True], ids=["baseline_init", "fast_read_init"])
def sampler(request, monkeypatch):
    monkeypatch.setenv("VLLM_DIFFUSION_READ_FAST_ADMISSION", str(int(request.param)))
    req_states = RequestState(4, 32, 16, CL, VOCAB, DEVICE)
    base = Sampler(4, VOCAB, DEVICE, req_states, logprobs_mode="raw_logprobs")
    states = DiffusionGemmaRequestStates(4, CL, VOCAB, 1, DEVICE, 4, 2)
    return DiffusionSampler(
        base,
        SimpleNamespace(canvas_length=CL),
        VOCAB,
        states,
        confidence_threshold=0.1,
        t_min=0.2,
        t_max=0.8,
        entropy_bound=0.1,
        embed_weight=torch.zeros(VOCAB, 4, device=DEVICE),
        normalizer=torch.tensor(1.0, device=DEVICE),
    )


def _add(sampler, slot, params):
    sampler.diffusion_states.add_request(
        slot, read_only=bool((params.extra_args or {}).get("diffusion_read_only"))
    )
    sampler.add_request(slot, 4, params)
    sampler.apply_staged_writes()
    sampler.diffusion_states.is_encoder_phase[slot] = False


def _read_params(
    candidates, positions=(1,), *, answer_only=False, scope=None, **kwargs
):
    kwargs.setdefault("detokenize", not answer_only)
    scope_args = (
        {} if scope is None else {"diffusion_read_self_conditioning_scope": scope}
    )
    return SamplingParams(
        logprob_token_ids=candidates,
        extra_args={
            "diffusion_read_only": True,
            "diffusion_seed_canvas": [10, 11, 12, 13],
            "diffusion_answer_positions": list(positions),
            "diffusion_answer_only_logprobs": answer_only,
            **scope_args,
        },
        **kwargs,
    )


def _batch(slots, nlogits=None, computed=4):
    n = len(slots)
    nlogits = nlogits if nlogits is not None else [CL] * n
    cu_logits = np.cumsum([0, *nlogits], dtype=np.int32)
    query_lens = np.maximum(nlogits, 2).astype(np.int32)
    query_start = np.cumsum([0, *query_lens], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=n,
        num_draft_tokens=sum(nlogits),
        idx_mapping_np=np.array(slots, dtype=np.int32),
        idx_mapping=torch.tensor(slots, device=DEVICE),
        logits_indices=torch.arange(sum(nlogits), device=DEVICE),
        cu_num_logits_np=cu_logits,
        query_start_loc_np=query_start,
        query_start_loc=torch.tensor(query_start, device=DEVICE),
        num_scheduled_tokens=query_lens,
        num_computed_prefill_tokens_np=np.full(n, computed, dtype=np.int32),
        prefill_len_np=np.full(n, 4, dtype=np.int32),
    )


def test_fused_read_seeds_drafts_before_prefill(sampler, monkeypatch):
    """The first forward must see the supplied canvas, including its noise slot."""
    monkeypatch.setenv("VLLM_DIFFUSION_FUSE_READ_PREFILL", "1")
    params = _read_params([1, 3])
    sampler.diffusion_states.add_request(2, read_only=True)
    sampler.add_request(2, 4, params)
    assert sampler.req_states.draft_tokens[2].tolist() == [10, 11, 12, 13]
    assert sampler.diffusion_states.is_encoder_phase[2]


@pytest.mark.parametrize("scope", [None, "canvas"])
def test_read_admission_and_removal_preserve_unused_state_before_ordinary_reuse(
    sampler, scope
):
    """Read slots can retain stale SC, but ordinary reuse must reset it."""
    states = sampler.diffusion_states
    model_state = SimpleNamespace(_req_id_to_index={}, diffusion_states=states)
    request = SimpleNamespace(
        req_id="reused",
        prompt_token_ids=[1] * 4,
        sampling_params=_read_params([1, 3], scope=scope),
    )
    states.canvas[2].fill_(-1)
    states.self_conditioning_embeds[2].fill_(7)
    states.step[2] = 9
    states.accepted_canvas_history_len[2] = 2
    states.read_progress[2] = (2, True)
    states.read_self_conditioning.add(2)
    states.read_canvas_self_conditioning.add(2)
    rng_before = torch.cuda.get_rng_state(DEVICE)
    DiffusionGemmaModelState.add_request(model_state, 2, request)
    sampler.add_request(2, 4, request.sampling_params)

    assert states.is_encoder_phase[2]
    assert states.step[2] == 0
    assert states.accepted_canvas_history_len[2] == 0
    assert states.prompt_len[2] == 4
    assert states.read_progress[2] == (1, True)
    assert 2 not in states.read_self_conditioning
    assert (2 in states.read_canvas_self_conditioning) == (scope == "canvas")
    if states._fast_read_admission:
        assert torch.equal(torch.cuda.get_rng_state(DEVICE), rng_before)
        assert (states.canvas[2] == -1).all()
        assert (states.self_conditioning_embeds[2] == 7).all()
    else:
        assert not torch.equal(torch.cuda.get_rng_state(DEVICE), rng_before)
        assert (states.canvas[2] >= 0).all()
        assert (states.self_conditioning_embeds[2] == 0).all()

    states.self_conditioning_embeds[2].fill_(7)
    DiffusionGemmaModelState.remove_request(model_state, request.req_id)
    assert not states.is_encoder_phase[2]
    assert 2 not in states.read_requests
    assert 2 not in states.read_progress
    assert 2 not in states.read_self_conditioning
    assert 2 not in states.read_canvas_self_conditioning
    expected_sc = 7 if states._fast_read_admission else 0
    assert (states.self_conditioning_embeds[2] == expected_sc).all()

    request.sampling_params = SamplingParams()
    DiffusionGemmaModelState.add_request(model_state, 2, request)
    assert states.is_encoder_phase[2]
    assert (states.canvas[2] >= 0).all()
    assert (states.self_conditioning_embeds[2] == 0).all()
    assert 2 not in states.read_canvas_self_conditioning


def test_fused_read_self_conditioning_preserves_prompt_embeddings():
    """Only the trailing canvas receives decoder self-conditioning."""
    embeds = torch.arange(12 * 4, device=DEVICE).reshape(12, 4).float()
    original = embeds.clone()
    model_state = SimpleNamespace(
        model=SimpleNamespace(self_conditioning=lambda inputs, soft: inputs + 1),
        diffusion_states=SimpleNamespace(read_requests={}),
    )
    DiffusionGemmaModelState._apply_self_conditioning(
        model_state,
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 7, 12]),
        embeds,
        torch.zeros((2, CL, 4), device=DEVICE),
        num_logits_np=np.array([CL, CL]),
    )
    expected = original.clone()
    expected[3:7] += 1
    expected[8:12] += 1
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("refining", [False, True])
@pytest.mark.usefixtures("default_vllm_config")
def test_read_self_conditioning_in_mixed_batch(dtype, refining):
    """Only later read passes consume their soft signal; ordinary rows retain it."""
    torch.manual_seed(0)
    sc = diffusion_gemma.DiffusionGemmaSelfConditioning(8, 16).to(
        device=DEVICE, dtype=dtype
    )
    inputs = torch.randn(6, 8, device=DEVICE, dtype=dtype)
    soft = torch.randn(3, 4, 8, device=DEVICE, dtype=dtype)
    read_soft = soft[2] if refining else torch.zeros_like(soft[2])
    expected = torch.cat((sc(inputs[:4], read_soft), sc(inputs[4:], soft[0, :2])))
    state = SimpleNamespace(
        model=SimpleNamespace(self_conditioning=sc),
        diffusion_states=SimpleNamespace(
            read_requests={2: None},
            read_reasoning={},
            read_self_conditioning={2} if refining else set(),
        ),
    )
    DiffusionGemmaModelState._apply_self_conditioning(
        state,
        np.array([2, 0]),
        np.array([0, 1]),
        np.array([0, 4, 6]),
        inputs,
        soft,
    )
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "phase,enabled,known",
    [
        ("first", True, True),
        ("intermediate", True, True),
        ("terminal", True, True),
        ("preserved", True, True),
        ("first", False, True),
        ("intermediate", True, False),
    ],
)
def test_known_content_feedback_preserves_mutable_rows_and_storage(
    dtype, device, phase, enabled, known
):
    """Known evidence uses one-hot embeddings without changing stored soft state."""
    weight = torch.arange(16 * 3, device=device).reshape(16, 3).sin().to(dtype)
    normalizer = torch.tensor(3**0.5, dtype=dtype, device=device)
    ids = torch.tensor([7, 8, 9, 10, 11, 12, 0], device=device)
    inputs = weight[ids] * normalizer
    original = inputs.clone()
    stored = torch.arange(15, device=device).reshape(1, 5, 3).float() / 9
    stored[:, -1].zero_()
    stored_before = stored.clone()
    has_previous = phase in ("intermediate", "preserved")
    previous = stored[0].to(dtype) if has_previous else torch.zeros_like(inputs[2:])
    mask = torch.tensor([True, False, True, False, False], device=device)
    captured = []

    class CaptureSC:
        @staticmethod
        def post_norm(values):
            return values * 3

        def __call__(self, values, soft):
            captured.append(soft.clone())
            return values + soft

    states = SimpleNamespace(
        read_requests={0: None},
        read_reasoning={
            0: SimpleNamespace(
                known_positions_mask=mask if known else None,
                delayed_suffix_positions=None,
            )
        },
        read_progress={0: (1, enabled)},
        read_self_conditioning={0} if has_previous else set(),
    )
    state = SimpleNamespace(
        model=SimpleNamespace(self_conditioning=CaptureSC()), diffusion_states=states
    )
    rng = torch.random.get_rng_state()
    DiffusionGemmaModelState._apply_self_conditioning(
        state,
        np.array([0]),
        np.array([0]),
        np.array([0, 7]),
        inputs,
        stored,
        num_logits_np=np.array([5]),
    )
    expected_soft = previous.clone()
    if known and enabled:
        one_hot = torch.nn.functional.one_hot(ids[2:][mask], len(weight)).to(dtype)
        expected_soft[mask] = (one_hot @ weight) * normalizer
    if has_previous or (known and enabled):
        torch.testing.assert_close(captured[0], expected_soft, rtol=0, atol=0)
        torch.testing.assert_close(inputs[2:], original[2:] + expected_soft)
    else:
        assert not captured
        torch.testing.assert_close(inputs[2:], original[2:] * 3)
    torch.testing.assert_close(inputs[:2], original[:2], rtol=0, atol=0)
    torch.testing.assert_close(stored, stored_before, rtol=0, atol=0)
    assert states.read_self_conditioning == ({0} if has_previous else set())
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("answer_only", [False, True])
@pytest.mark.parametrize("scope", [None, "answer", "canvas"])
def test_read_emits_only_after_requested_refinements(
    sampler, monkeypatch, self_conditioning, answer_only, scope
):
    """Intermediate passes feed back legal answers and preserve the scaffold."""
    params = _read_params([1, 3], answer_only=answer_only, scope=scope)
    params.extra_args.update(
        diffusion_read_steps=3,
        diffusion_read_self_conditioning=self_conditioning,
    )
    _add(sampler, 2, params)
    states = sampler.diffusion_states
    states.self_conditioning_embeds[2].fill_(7)
    sampler.embed_weight = torch.arange(
        VOCAB * 4, device=DEVICE, dtype=torch.bfloat16
    ).reshape(VOCAB, 4)
    batch = _batch([2])

    def no_ordinary_sampling(*args, **kwargs):
        pytest.fail("constrained refinement entered ordinary diffusion")

    monkeypatch.setattr(diffusion_gemma, "_compiled_sample_step", no_ordinary_sampling)
    for step, selected_token in enumerate([3, 1, 3]):
        logits = torch.zeros(CL, VOCAB, device=DEVICE)
        logits[1, selected_token] = 4
        logits[1, 15] = 5  # A higher-scoring illegal token must never be fed back.
        selected = sampler.prepare_sample_hidden_states(logits, batch)
        result = sampler(selected, batch)
        if step < 2:
            assert result.num_sampled.tolist() == [0]
            assert result.num_rejected.tolist() == [CL]
            assert result.logprobs_tensors is None
            assert states.read_progress[2] == (2 - step, self_conditioning)
            assert states.canvas[2].tolist() == [10, selected_token, 12, 13]
            assert sampler.req_states.draft_tokens[2].tolist() == [
                10,
                selected_token,
                12,
                13,
            ]
            if self_conditioning:
                assert 2 in states.read_self_conditioning
                rows = slice(None) if scope == "canvas" else [1]
                soft = (
                    logits[rows].softmax(-1).to(torch.bfloat16) @ sampler.embed_weight
                )
                torch.testing.assert_close(
                    states.self_conditioning_embeds[2, rows], soft.float()
                )
                if scope != "canvas":
                    assert (states.self_conditioning_embeds[2, [0, 2, 3]] == 0).all()
            else:
                assert 2 not in states.read_self_conditioning
        else:
            assert result.num_sampled.tolist() == [CL]
            assert result.num_rejected.tolist() == [0]
            assert result.sampled_token_ids.tolist() == [[10, 3, 12, 13]]
            lp = result.logprobs_tensors
            assert lp is not None
            row = 0 if answer_only else 1
            ids = lp.logprob_token_ids[row].tolist()
            for token in [1, 3]:
                torch.testing.assert_close(
                    lp.logprobs[row, ids.index(token)], logits[1].log_softmax(-1)[token]
                )
        assert not sampler._pending_logprobs


@pytest.mark.parametrize("scope", [None, "answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_refinement_feedback_is_scoped_and_preserves_other_slots(scope, enabled, dtype):
    """Tiled feedback updates valid rows without leaking into another request."""
    length = 65
    states = DiffusionGemmaRequestStates(2, length + 3, VOCAB, 1, "cpu", 4, 2)
    states.self_conditioning_embeds.fill_(7)
    seed = torch.arange(length) % VOCAB
    positions = torch.tensor([1])
    states.read_requests[0] = (seed, positions, (1,), True, True)
    states.read_progress[0] = (2, enabled)
    if scope == "canvas":
        states.read_canvas_self_conditioning.add(0)
    reader = DiffusionSampler.__new__(DiffusionSampler)
    reader.diffusion_states = states
    reader.req_states = SimpleNamespace(draft_tokens=torch.zeros_like(states.canvas))
    reader.embed_weight = torch.arange(VOCAB * 4).reshape(VOCAB, 4).sin().to(dtype)
    reader.normalizer = torch.tensor(2.0)
    reader.sc_vocab_start, reader.sc_vocab_end = 0, VOCAB
    reader.tp_size = 1
    reader._pending_logprobs = {}
    logits = torch.arange(length * VOCAB).reshape(length, VOCAB).cos().to(dtype)

    reader._advance_read(0, seed, logits)

    torch.testing.assert_close(states.canvas[0, :length], seed)
    torch.testing.assert_close(reader.req_states.draft_tokens[0, :length], seed)
    assert states.read_progress[0] == (1, enabled)
    assert (0 in states.read_self_conditioning) == enabled
    assert (states.self_conditioning_embeds[1] == 7).all()
    if enabled:
        rows = slice(0, length) if scope == "canvas" else positions
        expected = logits[rows].float().softmax(-1).to(dtype) @ reader.embed_weight
        expected = expected * reader.normalizer
        torch.testing.assert_close(
            states.self_conditioning_embeds[0, rows], expected.float()
        )
        untouched = torch.ones(length + 3, dtype=torch.bool)
        untouched[rows] = False
        assert not states.self_conditioning_embeds[0, untouched].count_nonzero()
    else:
        assert (states.self_conditioning_embeds[0] == 7).all()


def test_mixed_refinement_scopes_keep_separate_feedback(sampler):
    """Answer-only and full-canvas requests can refine in the same batch."""
    for slot, scope in [(0, "answer"), (2, "canvas")]:
        params = _read_params([1, 3], scope=scope)
        params.extra_args["diffusion_read_steps"] = 2
        _add(sampler, slot, params)
    sampler.embed_weight.fill_(1)
    logits = torch.zeros(2 * CL, VOCAB, device=DEVICE)

    result = sampler(logits, _batch([2, 0]))

    assert result.num_sampled.tolist() == [0, 0]
    feedback = sampler.diffusion_states.self_conditioning_embeds
    torch.testing.assert_close(feedback[2], torch.ones_like(feedback[2]))
    torch.testing.assert_close(feedback[0, 1], torch.ones_like(feedback[0, 1]))
    assert not feedback[0, [0, 2, 3]].count_nonzero()


def test_read_refinement_does_not_delay_other_read_results(sampler):
    """A mixed batch emits completed siblings while retaining unfinished reads."""
    _add(sampler, 2, _read_params([1, 3], answer_only=True))
    params = _read_params([2, 4], answer_only=True)
    params.extra_args["diffusion_read_steps"] = 2
    _add(sampler, 0, params)
    batch = _batch([2, 0])
    logits = torch.arange(VOCAB, device=DEVICE).float().repeat(2 * CL, 1)
    selected = sampler.prepare_sample_hidden_states(logits, batch)
    assert selected.shape == logits.shape
    result = sampler(selected, batch)
    assert result.num_sampled.tolist() == [CL, 0]
    assert result.num_rejected.tolist() == [0, CL]
    assert result.logprobs_tensors.cu_num_generated_tokens == [0, 1]
    assert result.logprobs_tensors.logprobs.shape[0] == 1
    assert sampler.diffusion_states.read_progress[0][0] == 1
    assert not sampler._pending_logprobs


@pytest.mark.parametrize("reason_steps", [1, 3, 64])
@pytest.mark.parametrize("answer_steps", [1, 2])
@pytest.mark.parametrize("answer_only", [False, True])
def test_reasoning_finishes_before_answer_reads_and_preserves_its_canvas(
    sampler, reason_steps, answer_steps, answer_only
):
    """Both compact and full final reads must attend and emit the frozen reason."""
    params = _read_params([1, 3], answer_only=answer_only)
    params.extra_args["diffusion_seed_canvas"][1] = 1
    params.extra_args.update(
        diffusion_read_steps=answer_steps,
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[
            token for token in range(VOCAB) if token not in (8, 9, 12, 13)
        ],
        diffusion_reasoning_steps=reason_steps,
    )
    _add(sampler, 2, params)
    states = sampler.diffusion_states
    batch = _batch([2])
    logits = torch.zeros(CL, VOCAB, device=DEVICE)
    logits[:, 15] = 100  # An illegal token beats every legal reasoning choice.
    logits[2, 9] = 4
    logits[3, 8] = 4
    for step in range(reason_steps):
        predicted = 3 if step % 2 == 0 else 1
        logits[1, [1, 3]] = 0
        logits[1, predicted] = 4
        selected = sampler.prepare_sample_hidden_states(logits, batch)
        assert selected.shape == (3, VOCAB)
        result = sampler(selected, batch)
        assert result.num_sampled.tolist() == [0]
        assert result.num_rejected.tolist() == [CL]
        assert result.logprobs_tensors is None
        assert states.canvas[2, :2].tolist() == [10, predicted]
        draft = sampler.req_states.draft_tokens[2]
        assert draft[1].item() in (1, 3)
        assert torch.equal(draft[[0, 2, 3]], states.canvas[2, [0, 2, 3]])
        assert set(states.canvas[2, 2:].tolist()) <= {8, 9, 12, 13}
        assert states.read_progress[2][0] == answer_steps
    assert states.canvas[2].tolist() == [10, predicted, 9, 8]
    assert 2 not in states.read_self_conditioning
    assert not states.self_conditioning_embeds[2].count_nonzero()

    for step in range(answer_steps):
        # A changed prediction must replace the actual noisy model input.
        noise = sampler.req_states.draft_tokens[2, 1].item()
        predicted = 1 if noise == 3 else 3
        logits[1, [1, 3]] = 0
        logits[1, predicted] = 4
        selected = sampler.prepare_sample_hidden_states(logits, batch)
        compact = step < answer_steps - 1 or answer_only
        assert selected.shape == ((1 if compact else CL), VOCAB)
        result = sampler(selected, batch)
        if step < answer_steps - 1:
            assert result.num_sampled.tolist() == [0]
            assert result.logprobs_tensors is None
            assert states.canvas[2, 1].item() == predicted
            assert sampler.req_states.draft_tokens[2, 1].item() in (1, 3)
        else:
            assert result.num_sampled.tolist() == [CL]
            assert result.sampled_token_ids.tolist() == [[10, predicted, 9, 8]]
            assert result.logprobs_tensors is not None

    states.remove_request(2)
    assert 2 not in states.read_reasoning
    _add(sampler, 2, _read_params([1, 3], answer_only=answer_only))
    logits[1, 1] = 0
    logits[1, 3] = 4
    result = sampler(sampler.prepare_sample_hidden_states(logits, batch), batch)
    assert result.sampled_token_ids.tolist() == [[10, 3, 12, 13]]


def _cpu_reasoning_sampler(
    *, scope="answer", enabled=True, answer_steps=1, answer_noise="candidates"
):
    states = DiffusionGemmaRequestStates(2, CL, VOCAB, 1, "cpu", 4, 2)
    reader = DiffusionSampler.__new__(DiffusionSampler)
    reader.diffusion_states = states
    reader.vocab_size = VOCAB
    reader.req_states = SimpleNamespace(draft_tokens=torch.zeros_like(states.canvas))
    reader.embed_weight = torch.arange(VOCAB * 4).reshape(VOCAB, 4).sin()
    reader.normalizer = torch.tensor(2.0)
    reader.sc_vocab_start, reader.sc_vocab_end = 0, VOCAB
    reader.tp_size = 1
    reader.t_min, reader.t_max = 0.2, 0.8
    reader.entropy_bound = 0.1
    reader._pending_logprobs = {}
    reader.logprob_token_ids_state = SimpleNamespace(
        num_token_ids=SimpleNamespace(np=np.array([2, 2])),
        token_ids=SimpleNamespace(gpu=torch.tensor([[1, 3], [1, 3]])),
    )
    for slot in range(2):
        states.read_requests[slot] = (
            torch.tensor([10, 1, 12, 13]),
            torch.tensor([1]),
            (1,),
            True,
            True,
        )
        states.canvas[slot] = states.read_requests[slot][0]
        reader.req_states.draft_tokens[slot] = states.read_requests[slot][0]
        states.read_progress[slot] = (answer_steps, enabled)
        states.read_reasoning[slot] = diffusion_gemma._ReadReasoning(
            positions=torch.tensor([2, 3]),
            positions_cpu=(2, 3),
            forbidden_token_ids=torch.tensor([15]),
            allowed_token_ids=torch.arange(15),
            steps=3,
            generator=torch.Generator().manual_seed(42),
            answer_noise=answer_noise,
        )
        if scope == "canvas":
            states.read_canvas_self_conditioning.add(slot)
    return reader


def _cpu_admit_reasoning(reader, slot, params, monkeypatch):
    reader.sampling_states = SimpleNamespace(add_request=lambda *args: None)
    reader.logprob_token_ids_state.add_request = lambda *args: None
    monkeypatch.setattr(
        diffusion_gemma,
        "async_tensor_h2d",
        lambda data, *, dtype, device: torch.tensor(data, dtype=dtype, device=device),
    )
    reader.add_request(slot, 4, params)


def _cpu_convergence_reads(
    monkeypatch, *, flags=(False, True), enabled=True, steps=4, counter=None
):
    reader = _cpu_reasoning_sampler(enabled=enabled, answer_noise="vocab")
    reader.diffusion_states.reasoning_counter_trace = counter
    reader.req_states.index_to_req_id = {0: "control", 1: "adaptive"}
    for slot, flag in enumerate(flags):
        params = _read_params([1, 3], answer_only=True)
        params.extra_args["diffusion_seed_canvas"][1] = 1
        params.extra_args.update(
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=steps,
            diffusion_reasoning_seed=42,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_read_self_conditioning=enabled,
        )
        if flag:
            params.extra_args["diffusion_reasoning_convergence_stop"] = True
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
    reader._compute_logprobs = lambda logits, *args, **kwargs: logits.log_softmax(-1)
    return reader


@pytest.mark.parametrize("enabled", [False, True])
def test_convergence_stop_cpu_commits_legal_argmax_then_runs_fresh_answer(
    monkeypatch, enabled
):
    """Only reason rows determine stopping; legacy draws precede the A1 boundary."""
    reader = _cpu_convergence_reads(monkeypatch, flags=(True, True), enabled=enabled)
    states = reader.diffusion_states
    logits = torch.zeros(CL, VOCAB)
    logits[2:, 15] = 100  # Forbidden raw winners cannot enter the final canvas.
    logits[2, 9] = logits[3, 8] = 30
    batch = SimpleNamespace(
        num_draft_tokens=CL,
        num_reqs=1,
        cu_num_logits_np=np.array([0, CL]),
        idx_mapping_np=np.array([1]),
    )
    expected_rng = torch.Generator().manual_seed(42)
    staged = []
    stage = reader._stage_read_canvas

    def record_stage(slot, canvas):
        staged.append(slot)
        stage(slot, canvas)

    monkeypatch.setattr(reader, "_stage_read_canvas", record_stage)
    for step in range(2):
        torch.rand((2, VOCAB), generator=expected_rng)
        torch.randint(15, (2,), generator=expected_rng)
        next_answer = torch.randint(VOCAB, (1,), generator=expected_rng)
        selected = reader.prepare_sample_hidden_states(logits, batch)
        assert selected.shape == (3, VOCAB)
        reader._advance_reasoning(0, logits)
        reader._advance_reasoning(1, selected, projected=True)
        for slot in (0, 1):
            reason = states.read_reasoning[slot]
            assert reason.steps == 4 and reason.step == step + 1
            assert reader._is_reasoning(slot) == (step == 0)
            assert torch.equal(reason.generator.get_state(), expected_rng.get_state())
            assert torch.equal(reader.req_states.draft_tokens[slot, [1]], next_answer)
            assert states.read_progress[slot] == (1, enabled)
            assert slot not in reader._pending_logprobs
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    assert staged == [0, 1, 0, 1]
    assert states.canvas.tolist() == [[10, 1, 9, 8], [10, 1, 9, 8]]
    assert not states.self_conditioning_embeds.count_nonzero()
    assert not states.read_self_conditioning
    assert all(
        reason.previous_argmax is None for reason in states.read_reasoning.values()
    )
    logits[1, 3] = 5
    selected = reader.prepare_sample_hidden_states(logits, batch)
    assert selected.shape == (1, VOCAB)
    for slot, rows, projected in ((0, logits, False), (1, selected, True)):
        canvas = reader._read_canvas(rows, slot, 0, 2, CL, projected=projected)
        assert canvas.tolist() == [10, 3, 9, 8]
        torch.testing.assert_close(
            reader._pending_logprobs[slot], logits[[1]].log_softmax(-1)
        )
        assert torch.equal(
            states.read_reasoning[slot].generator.get_state(), expected_rng.get_state()
        )


@pytest.mark.parametrize(
    "case", ["stable_high_entropy", "changing_confident", "cap_one"]
)
def test_convergence_stop_cpu_falls_back_to_exact_cap_path(monkeypatch, case):
    """Neither stability alone nor confidence alone can shorten the fixed path."""
    cap = 1 if case == "cap_one" else 4
    reader = _cpu_convergence_reads(monkeypatch, steps=cap)
    states = reader.diffusion_states
    for step in range(cap):
        logits = torch.zeros(CL, VOCAB)
        logits[:, 15] = 100
        if case != "stable_high_entropy":
            logits[2, 8 + step % 2] = logits[3, 9 - step % 2] = 30
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
            assert reader._is_reasoning(slot) == (step + 1 < cap)
            assert states.read_reasoning[slot].step == step + 1
            assert states.read_reasoning[slot].steps == cap
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
    output = [reader._read_canvas(logits, slot, 0, 2, CL) for slot in (0, 1)]
    torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)
    torch.testing.assert_close(
        reader._pending_logprobs[0], reader._pending_logprobs[1], rtol=0, atol=0
    )
    assert not states.read_self_conditioning
    assert not states.self_conditioning_embeds.count_nonzero()


def test_convergence_stop_cpu_mixed_reads_keep_independent_phase_and_rng(monkeypatch):
    """The early child's answer can finish while the unflagged sibling reasons."""
    reader = _cpu_convergence_reads(monkeypatch, flags=(True, False))
    control = _cpu_convergence_reads(monkeypatch, flags=(False, False))
    logits = torch.zeros(CL, VOCAB)
    logits[2, 9] = logits[3, 8] = 30
    batch = SimpleNamespace(
        num_draft_tokens=2 * CL,
        num_reqs=2,
        cu_num_logits_np=np.array([0, CL, 2 * CL]),
        idx_mapping_np=np.array([0, 1]),
    )
    for _ in range(2):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
        control._advance_reasoning(1, logits)
    assert not reader._is_reasoning(0) and reader._is_reasoning(1)
    selected = reader.prepare_sample_hidden_states(logits.repeat(2, 1), batch)
    torch.testing.assert_close(selected, logits[[1, 2, 3, 1]])
    completed_rng = reader.diffusion_states.read_reasoning[0].generator.get_state()
    canvas = reader._read_canvas(selected[:1], 0, 0, 2, CL, projected=True)
    assert canvas.tolist() == [10, 1, 9, 8]
    for _ in range(2):
        reader._advance_reasoning(1, logits)
        control._advance_reasoning(1, logits)
        for values, expected in (
            (reader.diffusion_states.canvas, control.diffusion_states.canvas),
            (
                reader.diffusion_states.self_conditioning_embeds,
                control.diffusion_states.self_conditioning_embeds,
            ),
            (reader.req_states.draft_tokens, control.req_states.draft_tokens),
        ):
            torch.testing.assert_close(values[1], expected[1], rtol=0, atol=0)
        assert torch.equal(
            reader.diffusion_states.read_reasoning[1].generator.get_state(),
            control.diffusion_states.read_reasoning[1].generator.get_state(),
        )
    assert torch.equal(
        reader.diffusion_states.read_reasoning[0].generator.get_state(), completed_rng
    )
    assert reader.diffusion_states.read_reasoning[0].step == 2
    assert reader.diffusion_states.read_reasoning[1].step == 4


def test_convergence_stop_cpu_counters_distinguish_cap_completion_and_abort(
    monkeypatch, tmp_path
):
    """Completed and aborted reads retain actual counts without token traces."""
    counter = diffusion_gemma.ReasoningCounterTrace(
        str(tmp_path), "convergence-stop-test", diffusion_gemma.__file__, 0
    )
    reader = _cpu_convergence_reads(monkeypatch, counter=counter)
    states = reader.diffusion_states
    logits = torch.zeros(CL, VOCAB)
    logits[2, 9] = logits[3, 8] = 30
    for _ in range(2):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
    record = counter.pending[1]
    assert record["actual_reasoning_passes"] == 2
    assert record["requested_reasoning_passes"] == 4
    assert record["terminal_temperature"] == pytest.approx(0.65)
    assert record["terminal_stable"] is True
    assert 0 <= record["terminal_mean_entropy"] < 0.005
    with pytest.raises(AssertionError):
        counter.finish(1)
    reader._read_canvas(logits, 1, 0, 2, CL)
    counter.answer(1)
    counter.finish(1)
    assert 1 not in counter.pending and 0 in counter.pending
    for _ in range(2):
        reader._advance_reasoning(0, logits)
    reader._read_canvas(logits, 0, 0, 2, CL)
    counter.answer(0)
    counter.finish(0)
    records = {
        record["request_id"]: record
        for record in (json.loads(path.read_text()) for path in tmp_path.iterdir())
    }
    assert set(records) == {"control", "adaptive"}
    for name, passes, reason in (
        ("control", 4, "cap"),
        ("adaptive", 2, "stable_confident"),
    ):
        record = records[name]
        assert record["status"] == "completed"
        assert record["requested_reasoning_passes"] == 4
        assert record["actual_reasoning_passes"] == passes
        assert record["actual_answer_passes"] == 1
        assert record["total_child_forwards"] == passes + 1
        assert record["terminal_reason"] == reason
        assert record["early_stop"] == (name == "adaptive")
        assert not {"tokens", "logits", "probabilities"}.intersection(record)
        assert record["run_id"] == "convergence-stop-test"
    params = _read_params([1, 3], answer_only=True)
    params.extra_args.update(
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_reasoning_steps=4,
        diffusion_reasoning_convergence_stop=True,
    )
    reader.req_states.index_to_req_id[0] = "cancelled"
    _cpu_admit_reasoning(reader, 0, params, monkeypatch)
    reader._advance_reasoning(0, logits)
    assert states.read_reasoning[0].previous_argmax is not None
    states.remove_request(0)
    assert 0 not in states.read_reasoning and 0 not in counter.pending
    aborted = next(
        record
        for record in (json.loads(path.read_text()) for path in tmp_path.iterdir())
        if record["request_id"] == "cancelled"
    )
    assert aborted["status"] == "aborted"
    assert aborted["actual_reasoning_passes"] == aborted["total_child_forwards"] == 1
    assert aborted["actual_answer_passes"] == 0
    reader.req_states.index_to_req_id[0] = "replacement"
    _cpu_admit_reasoning(reader, 0, params, monkeypatch)
    reason = states.read_reasoning[0]
    assert reason.step == 0 and reason.previous_argmax is None and not reason.terminal
    reader._advance_reasoning(0, logits)
    assert reader._is_reasoning(0) and reason.step == 1
    states.remove_request(0)
    assert not counter.pending


@pytest.mark.parametrize("sparse", [False, True])
def test_answer_mean_cpu_averages_probabilities_with_mixed_candidate_counts(sparse):
    """The four-read mean wins over the last read, including padded transport."""
    reader = _cpu_reasoning_sampler(answer_steps=4, answer_noise="vocab")
    states = reader.diffusion_states
    reader.logprob_token_ids_state.num_token_ids.np = np.array([2, 3])
    reader.logprob_token_ids_state.token_ids.gpu = torch.tensor([[1, 3, 0], [1, 3, 5]])
    matrices = (
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.01, 0.99]],
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.7, 0.1, 0.2], [0.01, 0.98, 0.01]],
    )
    for slot, rows in enumerate(matrices):
        states.read_reasoning[slot].step = 3
        states.read_answer_means[slot] = diffusion_gemma._ReadAnswerMean(4)
        states.read_requests[slot] = (*states.read_requests[slot][:3], sparse, True)
        candidates = [1, 3, 5][: len(rows[0])]
        expected = torch.tensor(rows, dtype=torch.float64).mean(0)
        for step, probabilities in enumerate(rows):
            logits = torch.zeros(CL, VOCAB)
            logits[:, 15] = 10 * step  # Outside-candidate mass must not weight a read.
            logits[1, candidates] = torch.tensor(probabilities).log()
            selected = logits[[1]] if slot else logits
            canvas = reader._read_canvas(selected, slot, 0, 3, CL, projected=bool(slot))
            if step < 3:
                reader._advance_read(slot, canvas, selected, projected=bool(slot))
                assert slot not in reader._pending_logprobs
        assert canvas.tolist() == [10, 1, 12, 13]
        assert candidates[rows[-1].index(max(rows[-1]))] == 3
        part = reader._pending_logprobs.pop(slot)
        assert part.logprobs.shape == (1 if sparse else CL, 4)
        params = SamplingParams(logprob_token_ids=candidates, flat_logprobs=True)
        processor = LogprobsProcessor.from_new_request(
            None, SimpleNamespace(sampling_params=params)
        )
        processor._update_sample_logprobs(part.tolists())
        assert processor.logprobs is not None
        row = processor.logprobs[0 if sparse else 1]
        assert row is not None
        assert set(row) == set(candidates)  # The two-candidate child excludes padding.
        actual = torch.tensor([row[token].logprob for token in candidates]).exp()
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
        assert slot not in states.read_answer_means


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("noise", ["candidates", "vocab"])
def test_answer_mean_cpu_preserves_reasoning_rng_and_clears_aborted_state(
    monkeypatch, enabled, noise
):
    """Only post-reason feedback changes; every answer noise draw stays private."""
    reader = _cpu_reasoning_sampler(enabled=enabled, answer_noise=noise)
    states = reader.diffusion_states
    params = _read_params([1, 3], answer_only=True)
    params.extra_args.update(
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_reasoning_steps=3,
        diffusion_reasoning_answer_noise=noise,
        diffusion_read_self_conditioning=enabled,
    )
    _cpu_admit_reasoning(reader, 0, params, monkeypatch)
    changed = deepcopy(params)
    changed.extra_args.update(
        diffusion_read_steps=4, diffusion_answer_pass_aggregation="mean"
    )
    _cpu_admit_reasoning(reader, 1, changed, monkeypatch)
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for step in range(3):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert states.read_self_conditioning == (
            {0, 1} if enabled and step < 2 else set()
        )
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
    expected_rng = torch.Generator()
    expected_rng.set_state(states.read_reasoning[0].generator.get_state())
    for _ in range(3):
        expected = torch.randint(
            VOCAB if noise == "vocab" else 2, (1,), generator=expected_rng
        )
        if noise == "candidates":
            expected = torch.tensor([1, 3])[expected]
        canvas = reader._read_canvas(logits, 1, 0, 2, CL)
        reader._advance_read(1, canvas, logits)
        assert torch.equal(reader.req_states.draft_tokens[1, [1]], expected)
        assert 1 not in states.read_self_conditioning
        assert not states.self_conditioning_embeds[1].count_nonzero()
    states.remove_request(1)
    assert 1 not in states.read_answer_means
    states.read_answer_means[1] = diffusion_gemma._ReadAnswerMean(4)
    _cpu_admit_reasoning(reader, 1, params, monkeypatch)
    assert 1 not in states.read_answer_means


@pytest.mark.parametrize("seed", ["omitted", None, 0, 2**64 - 1])
def test_reasoning_cpu_admission_keeps_canvas_hash_default_and_explicit_zero(
    monkeypatch, seed
):
    """The seed controls only request-local RNG, including the unsigned endpoints."""
    reader = _cpu_reasoning_sampler()
    params = _read_params([1, 3])
    params.extra_args.update(
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[15],
    )
    if seed != "omitted":
        params.extra_args["diffusion_reasoning_seed"] = seed
    original = deepcopy(params.extra_args)
    _cpu_admit_reasoning(reader, 0, params, monkeypatch)
    expected = (
        int.from_bytes(
            hashlib.sha256(
                json.dumps(original["diffusion_seed_canvas"]).encode()
            ).digest()[:8],
            "little",
        )
        if seed in ("omitted", None)
        else seed
    )
    actual = reader.diffusion_states.read_reasoning[0].generator
    reference = torch.Generator().manual_seed(expected)
    assert actual.initial_seed() == expected
    assert torch.equal(actual.get_state(), reference.get_state())
    assert params.extra_args == original


@pytest.mark.parametrize("preserve", [False, True])
def test_reasoning_cpu_known_content_leaves_sampling_and_terminal_feedback_unchanged(
    monkeypatch, preserve
):
    """Known rows affect input feedback only, including after terminal clearing."""
    reader = _cpu_reasoning_sampler(answer_noise="vocab")
    states = reader.diffusion_states
    for slot, known in enumerate((None, [0])):
        params = _read_params([1, 3])
        params.extra_args.update(
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_reasoning_preserve_feedback=preserve,
            diffusion_reasoning_known_positions=known,
        )
        original = deepcopy(params.extra_args)
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        assert params.extra_args == original
    assert states.read_reasoning[0].known_positions_mask is None
    mask = states.read_reasoning[1].known_positions_mask
    assert mask.tolist() == [True, False, False, False]
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for _ in range(3):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
        for values in (
            states.canvas,
            reader.req_states.draft_tokens,
            states.self_conditioning_embeds,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
    assert states.read_reasoning[1].known_positions_mask is mask
    assert states.read_self_conditioning == ({0, 1} if preserve else set())
    assert not states.self_conditioning_embeds[:, :2].count_nonzero()


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_delayed_suffix_inputs_respect_phase_and_mixed_query_boundaries(steps, enabled):
    """Suffix input changes must not escape a child or survive into answer reads."""
    weight = torch.arange(VOCAB * 3).reshape(VOCAB, 3).float() / 13 + 1
    normalizer = torch.tensor(2.0)
    stored = torch.arange(3 * 6 * 3).reshape(3, 6, 3).float() / 9
    stored_before = stored.clone()
    rng = torch.random.get_rng_state()

    class CaptureSC:
        @staticmethod
        def post_norm(values):
            return values * 2

        def __call__(self, values, soft):
            return values + soft

    for span in (0, 1, 3, 6):
        prefix = 2
        end = prefix + span
        count = end + CL + 2
        ids = torch.arange(count) % VOCAB
        batch = SimpleNamespace(
            num_tokens=count,
            num_tokens_after_padding=count + 2,
            num_reqs=3,
            input_ids=ids,
            num_draft_tokens=span + CL,
            idx_mapping_np=np.array([0, 1, 2]),
            cu_num_logits_np=np.array([0, span, span + CL, span + CL]),
            query_start_loc_np=np.array([0, end, end + CL, count]),
        )
        for phase in range(steps + 1):
            lookups: list[torch.Tensor] = []

            def embed(token_ids, captured=lookups):
                captured.append(token_ids.clone())
                return weight[token_ids] * normalizer

            previous = enabled and 0 < phase < steps
            states = SimpleNamespace(
                read_requests={0: None, 1: None},
                read_reasoning={
                    0: SimpleNamespace(
                        delayed_suffix_positions=torch.tensor([1, 3, 5]),
                        delayed_suffix_positions_cpu=(1, 3, 5),
                        step=phase,
                        steps=steps,
                        terminal=phase == steps,
                        known_positions_mask=None,
                    )
                },
                read_progress={0: (1, enabled), 1: (1, False)},
                read_self_conditioning={0} if previous else set(),
                self_conditioning_embeds=stored,
            )
            state = SimpleNamespace(
                model=SimpleNamespace(
                    embed_input_ids=embed, self_conditioning=CaptureSC()
                ),
                diffusion_states=states,
                _req_id_to_index={"delayed": 0, "control": 1, "prefill": 2},
                _pending_mm_embeds=None,
                _inputs_embeds_buf=torch.full((count + 2, 3), -123.0),
            )
            state._apply_self_conditioning = MethodType(
                DiffusionGemmaModelState._apply_self_conditioning, state
            )
            actual = DiffusionGemmaModelState.prepare_inputs(state, batch, None)[
                "inputs_embeds"
            ]
            expected = torch.full_like(actual, -123.0)
            expected[:count] = weight[ids] * normalizer
            selected = [p for p in (1, 3, 5) if p < span and phase < steps]
            for position in selected:
                expected[prefix + position] = weight[0] * normalizer
            if previous:
                expected[prefix:end] += stored[0, :span]
            else:
                expected[prefix:end] *= 2
            expected[end : end + CL] *= 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert len(lookups) == 1 + bool(selected)
            if selected:
                assert lookups[1].tolist() == [0] * len(selected)
            torch.testing.assert_close(stored, stored_before, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("delayed", ["omitted", None, [3]])
def test_delayed_suffix_admission_preserves_sampling_and_private_rng(
    monkeypatch, delayed
):
    """The input-only option cannot alter canonical staging or sampler draws."""
    reader = _cpu_reasoning_sampler(answer_steps=2, answer_noise="vocab")
    states = reader.diffusion_states
    for slot in (0, 1):
        params = _read_params([1, 3], answer_only=True)
        params.extra_args.update(
            diffusion_read_steps=2,
            diffusion_reasoning_positions=[0, 2],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
            diffusion_reasoning_answer_noise="vocab",
        )
        if slot == 1 and delayed != "omitted":
            params.extra_args["diffusion_reasoning_delayed_suffix_positions"] = delayed
        original = deepcopy(params.extra_args)
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        assert params.extra_args == original
    reason = states.read_reasoning[1]
    expected_positions = () if delayed in ("omitted", None) else (3,)
    assert reason.delayed_suffix_positions_cpu == expected_positions
    if delayed in ("omitted", None):
        assert reason.delayed_suffix_positions is None
    else:
        assert reason.delayed_suffix_positions.tolist() == [3]
    batch = SimpleNamespace(
        num_draft_tokens=CL,
        num_reqs=1,
        cu_num_logits_np=np.array([0, CL]),
        idx_mapping_np=np.array([1]),
    )
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for step in range(4):
        selected = reader.prepare_sample_hidden_states(logits, batch)
        assert selected.shape == (3 if step < 3 else 1, VOCAB)
        if step < 3:
            reader._advance_reasoning(0, logits)
            reader._advance_reasoning(1, selected, projected=True)
        else:
            for slot, rows, compact in ((0, logits, False), (1, selected, True)):
                canvas = reader._read_canvas(rows, slot, 0, 2, CL, projected=compact)
                reader._advance_read(slot, canvas, rows, projected=compact)
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert states.canvas[1, 3].item() == 13
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            reason.generator.get_state(),
        )


def test_reasoning_cpu_seed_override_preserves_draws_after_prefix_crop(monkeypatch):
    """Moving fixed scaffold into a prefix leaves equal mutable-row draws aligned."""
    reader = _cpu_reasoning_sampler(enabled=False, answer_noise="vocab")
    states = reader.diffusion_states
    original_seed = [10, 1, 12, 13]
    seed = int.from_bytes(
        hashlib.sha256(json.dumps(original_seed).encode()).digest()[:8], "little"
    )
    for slot in (0, 1):
        params = _read_params([1, 3], positions=(1 - slot,))
        params.extra_args.update(
            diffusion_seed_canvas=original_seed[slot:],
            diffusion_read_self_conditioning=False,
            diffusion_reasoning_positions=[2 - slot, 3 - slot],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
            diffusion_reasoning_answer_noise="vocab",
        )
        if slot:
            params.extra_args["diffusion_reasoning_seed"] = seed
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        states.canvas[slot].zero_()
        states.canvas[slot, : CL - slot] = torch.tensor(original_seed[slot:])
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for _ in range(3):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits[slot:])
        assert torch.equal(states.canvas[0, 1:], states.canvas[1, :3])
        assert torch.equal(
            reader.req_states.draft_tokens[0, 1:], reader.req_states.draft_tokens[1, :3]
        )
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )


def test_reasoning_cpu_noncontiguous_sections_keep_joint_updates_and_rng(monkeypatch):
    """A pinned heading between free rows must not split their denoising passes."""
    reader = _cpu_reasoning_sampler(answer_noise="vocab")
    states = reader.diffusion_states
    order = [0, 2, 1, 3]
    for slot, (seed, positions) in enumerate(
        (([12, 13, 10, 1], [0, 1]), ([12, 10, 13, 1], [0, 2]))
    ):
        params = _read_params([1, 3], positions=(3,), answer_only=True)
        params.extra_args.update(
            diffusion_seed_canvas=seed,
            diffusion_reasoning_positions=positions,
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_reasoning_seed=42,
        )
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        states.canvas[slot] = torch.tensor(seed)
        reader.req_states.draft_tokens[slot] = torch.tensor(seed)
    batch = SimpleNamespace(
        num_draft_tokens=CL,
        num_reqs=1,
        cu_num_logits_np=np.array([0, CL]),
        idx_mapping_np=np.array([1]),
    )
    reader._compute_logprobs = lambda logits, *args, **kwargs: logits.log_softmax(-1)
    noisy_answers = []
    for step in range(4):
        noise = reader.req_states.draft_tokens[0, 3].item()
        prediction = 3 if noise == 1 else 1
        logits = torch.zeros(CL, VOCAB)
        logits[0, 9] = logits[1, 8] = logits[3, prediction] = 4
        logits[:, 15] = 100
        gapped_logits = logits[order]
        selected = reader.prepare_sample_hidden_states(gapped_logits, batch)
        expected_rows = [0, 2, 3] if step < 3 else [3]
        torch.testing.assert_close(selected, gapped_logits[expected_rows])
        if step == 3:
            packed = reader._read_canvas(logits, 0, 0, 2, CL)
            gapped = reader._read_canvas(selected, 1, 0, 2, CL, projected=True)
            assert packed.tolist() == [9, 8, 10, prediction]
            assert gapped.tolist() == [9, 10, 8, prediction]
            assert prediction != noise
            torch.testing.assert_close(
                reader._pending_logprobs[0], reader._pending_logprobs[1]
            )
            break
        previous_rng = states.read_reasoning[0].generator.get_state()
        reader._advance_reasoning(0, logits)
        reader._advance_reasoning(1, selected, projected=True)
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1, order], rtol=0, atol=0)
        assert states.canvas[1, 1].item() == 10
        assert not states.self_conditioning_embeds[1, 1].count_nonzero()
        assert states.canvas[1, 3].item() == prediction
        assert all(reason.step == step + 1 for reason in states.read_reasoning.values())
        assert not torch.equal(
            previous_rng, states.read_reasoning[0].generator.get_state()
        )
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
        noisy_answers.append(reader.req_states.draft_tokens[1, 3].item())
    assert all(0 <= token < VOCAB for token in noisy_answers)
    assert set(noisy_answers) - {1, 3}


def test_reasoning_does_not_delay_completed_sibling_reads(sampler):
    """An unfinished reasoning request must not hold another request's answer."""
    params = _read_params([1, 3], answer_only=True)
    params.extra_args.update(
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_reasoning_steps=3,
    )
    _add(sampler, 2, params)
    _add(sampler, 0, _read_params([1, 3], answer_only=True))
    batch = _batch([2, 0])
    logits = torch.arange(VOCAB, device=DEVICE).float().repeat(2 * CL, 1)
    result = sampler(sampler.prepare_sample_hidden_states(logits, batch), batch)
    assert result.num_sampled.tolist() == [0, CL]
    assert result.sampled_token_ids[1].tolist() == [10, 3, 12, 13]
    assert result.logprobs_tensors is not None
    assert result.logprobs_tensors.cu_num_generated_tokens == [0, 0]
    assert result.logprobs_tensors.logprobs.shape[0] == 1
    assert not sampler._pending_logprobs


@pytest.mark.parametrize("scope", ["answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_cpu_feedback_is_constrained_and_cleared_before_answer(
    scope, enabled
):
    """Feedback cannot reintroduce forbidden tokens or leak into answer reads."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled)
    states = reader.diffusion_states
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float() / 8
    logits[:, 15] = 100
    reader._advance_reasoning(0, logits)
    assert 0 not in reader._pending_logprobs
    assert states.read_progress[0] == (1, enabled)
    assert states.canvas[0, :2].tolist() == [10, 3]
    assert (states.canvas[0, 2:] < 15).all()
    feedback = states.self_conditioning_embeds[0]
    if enabled:
        constrained = logits[2:] / reader.t_max
        constrained[:, 15] = -float("inf")
        expected = reader._read_soft_embeddings(constrained)
        torch.testing.assert_close(feedback[2:], expected)
        if scope == "canvas":
            torch.testing.assert_close(
                feedback[:2], reader._read_soft_embeddings(logits[:2])
            )
        else:
            assert not feedback[:2].count_nonzero()
    else:
        assert not feedback.count_nonzero()
    assert not states.self_conditioning_embeds[1].count_nonzero()

    reader._advance_reasoning(0, logits)
    reader._advance_reasoning(0, logits)
    assert states.canvas[0].tolist() == [10, 3, 14, 14]
    assert not feedback.count_nonzero()
    assert 0 not in states.read_self_conditioning
    assert not reader._is_reasoning(0)


def test_reasoning_cpu_randomness_is_independent_of_request_interleaving():
    """Equivalent requests keep the same path regardless of other random draws."""
    reader = _cpu_reasoning_sampler(enabled=False)
    states = reader.diffusion_states
    logits = torch.zeros(CL, VOCAB)
    for _ in range(2):
        reader._advance_reasoning(0, logits)
        torch.rand(1234)
        reader._advance_reasoning(1, logits)
        torch.testing.assert_close(states.canvas[0], states.canvas[1])
        torch.testing.assert_close(
            reader.req_states.draft_tokens[0], reader.req_states.draft_tokens[1]
        )


def test_reasoning_cpu_explicit_defaults_preserve_outputs_and_randomness():
    """Omitting the controls is identical to the original schedule and budget."""
    reader = _cpu_reasoning_sampler()
    states = reader.diffusion_states
    states.read_reasoning[1].schedule_steps = 3
    states.read_reasoning[1].entropy_bound = 0.1
    states.read_reasoning[1].round_feedback_logits = False
    states.read_reasoning[1].preserve_feedback = False
    states.read_reasoning[1].answer_feedback = False
    states.read_reasoning[1].unmasked_feedback = False
    states.read_reasoning[1].token_noise = "allowed"
    states.read_reasoning[1].token_proposals = "allowed"
    states.read_reasoning[1].convergence_stop = False
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for _ in range(3):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_cpu_vocab_noise_changes_only_rejected_transient_tokens(
    monkeypatch, steps, enabled
):
    """Rejected rows can contain forbidden noise; committed rows and RNG stay exact."""
    reader = _cpu_reasoning_sampler(enabled=enabled)
    states = reader.diffusion_states
    for slot in (0, 1):
        params = _read_params([1, 3])
        params.extra_args.update(
            diffusion_read_self_conditioning=enabled,
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=steps,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_reasoning_entropy_bound=0.0,
        )
        if slot:
            params.extra_args["diffusion_reasoning_token_noise"] = "vocab"
        original = deepcopy(params.extra_args)
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        assert params.extra_args == original
        assert states.read_reasoning[slot].token_noise == (
            "vocab" if slot else "allowed"
        )
    calls: list[list[tuple[int, tuple[int, ...]]]] = [[], []]
    randint = torch.randint

    def draw(high, size, **kwargs):
        result = randint(high, size, **kwargs)
        slot = next(
            i
            for i in (0, 1)
            if kwargs["generator"] is states.read_reasoning[i].generator
        )
        calls[slot].append((high, tuple(size)))
        if tuple(size) == (2,):
            result.fill_(high - 1)
        return result

    monkeypatch.setattr(torch, "randint", draw)
    logits = torch.zeros(CL, VOCAB)
    logits[2] = -4
    logits[2, 2] = 2
    logits[:, 15] = 100
    for step in range(steps):
        reader._advance_reasoning(0, logits)
        reader._advance_reasoning(1, logits[[2, 3, 1]], projected=True)
        if step < steps - 1:
            assert states.canvas[:, 3].tolist() == [14, 15]
            assert states.canvas[0, 2] == states.canvas[1, 2] < 15
        else:
            torch.testing.assert_close(
                states.canvas[0], states.canvas[1], rtol=0, atol=0
            )
            assert (states.canvas[:, 2:] < 15).all()
        assert set(states.canvas[:, 1].tolist()) <= {1, 3}
        assert states.canvas[:, 0].tolist() == [10, 10]
        assert (
            reader.req_states.draft_tokens[0, 1] == reader.req_states.draft_tokens[1, 1]
        )
        torch.testing.assert_close(
            states.self_conditioning_embeds[0],
            states.self_conditioning_embeds[1],
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
    assert calls[0] == [(15, (2,)), (16, (1,))] * (steps - 1) + [(16, (1,))]
    assert calls[1] == [(16, (2,)), (16, (1,))] * (steps - 1) + [(16, (1,))]


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_cpu_vocab_proposals_change_acceptance_but_keep_final_mask(
    monkeypatch, steps, enabled
):
    """Unmasked proposals change accepted rows, not final legality, SC or RNG."""
    reader = _cpu_reasoning_sampler(enabled=enabled)
    states = reader.diffusion_states
    for slot in (0, 1):
        params = _read_params([1, 3])
        params.extra_args.update(
            diffusion_read_self_conditioning=enabled,
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=steps,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_reasoning_entropy_bound=0.0,
        )
        if slot:
            params.extra_args["diffusion_reasoning_token_proposals"] = "vocab"
        original = deepcopy(params.extra_args)
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        assert params.extra_args == original
        assert states.read_reasoning[slot].token_proposals == (
            "vocab" if slot else "allowed"
        )
    calls: list[list[tuple[str, int | None, tuple[int, ...]]]] = [[], []]
    rand, randint = torch.rand, torch.randint

    def record(kind, high, size, generator):
        slot = next(
            i for i in (0, 1) if generator is states.read_reasoning[i].generator
        )
        calls[slot].append((kind, high, tuple(size)))

    def uniform(size, **kwargs):
        value = rand(size, **kwargs)
        record("rand", None, size, kwargs["generator"])
        return value.fill_(0.5)

    def integer(high, size, **kwargs):
        value = randint(high, size, **kwargs)
        record("randint", high, size, kwargs["generator"])
        if tuple(size) == (2,):
            value.fill_(7)
        return value

    monkeypatch.setattr(torch, "rand", uniform)
    monkeypatch.setattr(torch, "randint", integer)
    logits = torch.zeros(CL, VOCAB)
    logits[2] = -6
    logits[2, 2], logits[2, 15], logits[3, 15] = 6, 5, 10
    original_logits = logits.clone()
    for step in range(steps):
        reader._advance_reasoning(0, logits)
        reader._advance_reasoning(1, logits[[2, 3, 1]], projected=True)
        if step < steps - 1:
            assert states.canvas[:, 2:].tolist() == [[2, 7], [7, 15]]
        else:
            assert states.canvas[:, 2:].tolist() == [[2, 0], [2, 0]]
        assert states.canvas[:, :2].tolist() == [[10, 1], [10, 1]]
        assert (
            reader.req_states.draft_tokens[0, 1] == reader.req_states.draft_tokens[1, 1]
        )
        first, second = states.self_conditioning_embeds
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        if enabled and step < steps - 1:
            temperature = reader.t_min + (reader.t_max - reader.t_min) * (
                (steps - step) / steps
            )
            masked = logits[2:].float() / temperature
            masked[:, 15] = -float("inf")
            probabilities = masked.softmax(-1).to(reader.embed_weight.dtype)
            expected = probabilities @ reader.embed_weight * reader.normalizer
            torch.testing.assert_close(
                second[2:], expected.to(second.dtype), rtol=0, atol=0
            )
            assert not second[:2].count_nonzero()
        else:
            assert not second.count_nonzero()
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
        torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
    expected_calls = [
        ("rand", None, (2, VOCAB)),
        ("randint", 15, (2,)),
        ("randint", 16, (1,)),
    ] * (steps - 1) + [("randint", 16, (1,))]
    assert calls[0] == calls[1] == expected_calls
    assert not states.read_self_conditioning


@pytest.mark.parametrize(
    "scope,enabled,round_logits,preserve,entropy_bound",
    [
        ("answer", True, False, False, 0.0),
        ("answer", True, True, True, 0.1),
        ("canvas", True, False, False, 0.1),
        ("answer", False, True, True, 0.1),
    ],
)
def test_reasoning_cpu_unmasked_feedback_keeps_sampling_constrained(
    monkeypatch, scope, enabled, round_logits, preserve, entropy_bound
):
    """Forbidden mass can inform feedback without altering hard choices or RNG."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled)
    reader.embed_weight = reader.embed_weight.bfloat16()
    states = reader.diffusion_states
    for slot in (0, 1):
        params = _read_params([1, 3], scope=scope)
        params.extra_args.update(
            diffusion_read_self_conditioning=enabled,
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
            diffusion_reasoning_answer_noise="vocab",
            diffusion_reasoning_entropy_bound=entropy_bound,
            diffusion_reasoning_round_feedback_logits=round_logits,
            diffusion_reasoning_preserve_feedback=preserve,
            diffusion_reasoning_answer_feedback=True,
        )
        if slot:
            params.extra_args["diffusion_reasoning_unmasked_feedback"] = True
        _cpu_admit_reasoning(reader, slot, params, monkeypatch)
        assert states.read_reasoning[slot].unmasked_feedback == bool(slot)
    batch = SimpleNamespace(
        num_draft_tokens=CL,
        num_reqs=1,
        cu_num_logits_np=np.array([0, CL]),
        idx_mapping_np=np.array([1]),
    )
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    logits[:, 15] = 7
    original = logits.clone()
    for step in range(3):
        selected = reader.prepare_sample_hidden_states(logits, batch)
        projected = not (scope == "canvas" and enabled)
        assert selected.shape == ((3 if projected else CL), VOCAB)
        reader._advance_reasoning(0, logits)
        reader._advance_reasoning(1, selected, projected=projected)
        first, second = states.self_conditioning_embeds
        keep = enabled and (step < 2 or preserve)
        assert (1 in states.read_self_conditioning) == keep
        if keep:
            temperature = reader.t_min + (reader.t_max - reader.t_min) * (
                (3 - step) / 3
            )
            scaled = logits[2:].float() / temperature
            if round_logits:
                scaled = scaled.to(reader.embed_weight.dtype)
            probabilities = scaled.float().softmax(-1).to(reader.embed_weight.dtype)
            expected = probabilities @ reader.embed_weight * reader.normalizer
            torch.testing.assert_close(
                second[2:], expected.to(second.dtype), rtol=0, atol=0
            )
            assert not torch.equal(first[2:], second[2:])
        else:
            assert not second.count_nonzero()
        torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)
        if step == 2:
            assert not second[:2].count_nonzero()
        for values in (states.canvas, reader.req_states.draft_tokens):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
            assert (values[:, 2:] < 15).all()
        assert set(states.canvas[:, 1].tolist()) <= {1, 3}
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
        torch.testing.assert_close(logits, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("control", "embedding_dtype"),
    [
        ("round_feedback_logits", torch.bfloat16),
        ("preserve_feedback", torch.bfloat16),
        ("both", torch.bfloat16),
        ("round_feedback_logits", torch.float32),
    ],
)
@pytest.mark.parametrize("scope", ["answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_cpu_feedback_controls_preserve_sampling(
    control, embedding_dtype, scope, enabled
):
    """Feedback changes only the requested rows, never hard outputs or RNG draws."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled)
    reader.embed_weight = reader.embed_weight.to(embedding_dtype)
    states = reader.diffusion_states
    states.read_reasoning[1].round_feedback_logits = control != "preserve_feedback"
    states.read_reasoning[1].preserve_feedback = control != "round_feedback_logits"
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float() / 100 + 64
    logits[:, 15] = 100
    for step in range(3):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
            feedback = states.self_conditioning_embeds[slot]
            keep = enabled and (
                step < 2 or (slot == 1 and control != "round_feedback_logits")
            )
            assert (slot in states.read_self_conditioning) == keep
            if keep:
                temperature = reader.t_min + (reader.t_max - reader.t_min) * (
                    (3 - step) / 3
                )
                scaled = logits[2:] / temperature
                scaled[:, 15] = -float("inf")
                if slot == 1 and control != "preserve_feedback":
                    scaled = scaled.to(reader.embed_weight.dtype)
                probabilities = scaled.float().softmax(dim=-1).to(embedding_dtype)
                expected = probabilities @ reader.embed_weight * reader.normalizer
                expected = expected.to(feedback.dtype)
                torch.testing.assert_close(feedback[2:], expected, rtol=0, atol=0)
            else:
                assert not feedback[2:].count_nonzero()
            if step == 2 or scope == "answer" or not enabled:
                assert not feedback[:2].count_nonzero()
        for values in (states.canvas, reader.req_states.draft_tokens):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        torch.testing.assert_close(
            states.self_conditioning_embeds[0, :2],
            states.self_conditioning_embeds[1, :2],
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )
        if step == 0 and enabled and control != "preserve_feedback":
            identical = torch.equal(
                states.self_conditioning_embeds[0, 2:],
                states.self_conditioning_embeds[1, 2:],
            )
            assert identical == (embedding_dtype == torch.float32)
    assert not reader._is_reasoning(0) and not reader._is_reasoning(1)


@pytest.mark.parametrize("scope", ["answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("reason_steps", [1, 3])
@pytest.mark.parametrize("preserve", [False, True])
def test_reasoning_cpu_answer_feedback_is_only_intermediate(
    scope, enabled, reason_steps, preserve
):
    """Only active answer rows gain feedback; final reasoning stays separate."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled)
    reader.embed_weight = reader.embed_weight.bfloat16()
    states = reader.diffusion_states
    states.read_reasoning[1].answer_feedback = True
    for reasoning in states.read_reasoning.values():
        reasoning.steps = reason_steps
        reasoning.preserve_feedback = preserve
        reasoning.round_feedback_logits = True
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    logits[:, 15] = 7
    probabilities = logits[1:2].float().softmax(dim=-1).to(reader.embed_weight.dtype)
    expected_answer = probabilities @ reader.embed_weight * reader.normalizer
    for step in range(reason_steps):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
        first, second = states.self_conditioning_embeds
        torch.testing.assert_close(first[[0, 2, 3]], second[[0, 2, 3]], rtol=0, atol=0)
        active = enabled and scope == "answer" and step < reason_steps - 1
        if active:
            assert not first[1].count_nonzero()
            torch.testing.assert_close(
                second[1:2], expected_answer.to(second.dtype), rtol=0, atol=0
            )
            assert second[1].count_nonzero()
        else:
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        if step == reason_steps - 1:
            assert not second[:2].count_nonzero()
            assert (1 in states.read_self_conditioning) == (enabled and preserve)
        for values in (states.canvas, reader.req_states.draft_tokens):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )


def test_reasoning_cpu_longer_schedule_changes_feedback_without_extra_passes():
    """The requested horizon softens later feedback only for that request."""
    reader = _cpu_reasoning_sampler()
    states = reader.diffusion_states
    states.read_reasoning[1].schedule_steps = 48
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float() / 8
    for _ in range(2):
        for slot in (0, 1):
            reader._advance_reasoning(slot, logits)
    for slot, temperature in ((0, 0.6), (1, 0.7875)):
        scaled = logits[2:] / temperature
        scaled[:, 15] = -float("inf")
        torch.testing.assert_close(
            states.self_conditioning_embeds[slot, 2:],
            reader._read_soft_embeddings(scaled),
        )
    assert not torch.equal(
        states.self_conditioning_embeds[0], states.self_conditioning_embeds[1]
    )
    for slot in (0, 1):
        reader._advance_reasoning(slot, logits)
        assert not reader._is_reasoning(slot)
        assert not states.self_conditioning_embeds[slot].count_nonzero()


def test_reasoning_cpu_entropy_budget_controls_renoising_per_request(monkeypatch):
    """Zero accepts one uncertain position; a larger budget accepts both."""
    reader = _cpu_reasoning_sampler(enabled=False)
    states = reader.diffusion_states
    states.read_reasoning[0].entropy_bound = 0.0
    states.read_reasoning[1].entropy_bound = 3.0
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.full(shape, 0.5))
    monkeypatch.setattr(
        torch, "randint", lambda high, shape, **kw: torch.full(shape, high - 1)
    )
    for slot in (0, 1):
        reader._advance_reasoning(slot, torch.zeros(CL, VOCAB))
        assert states.canvas[slot, :2].tolist() == [10, 1]
    assert sorted(states.canvas[0, 2:].tolist()) == [0, 14]
    assert states.canvas[1, 2:].tolist() == [0, 0]


@pytest.mark.parametrize("scope", ["answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "feedback",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_reasoning_cpu_projection_preserves_denoising_and_refinement(
    scope, enabled, feedback
):
    """Removing unused vocabulary rows preserves predictions, feedback and RNG."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled, answer_steps=2)
    states = reader.diffusion_states
    for reasoning in states.read_reasoning.values():
        (
            reasoning.round_feedback_logits,
            reasoning.preserve_feedback,
            reasoning.answer_feedback,
        ) = feedback
    batch = SimpleNamespace(
        num_draft_tokens=CL,
        num_reqs=1,
        cu_num_logits_np=np.array([0, CL]),
        idx_mapping_np=np.array([1]),
    )
    logits = torch.arange(CL * VOCAB).reshape(CL, VOCAB).float().sin()
    for step in range(4):
        selected = reader.prepare_sample_hidden_states(logits, batch)
        projected = not (scope == "canvas" and enabled)
        expected_rows = (3 if step < 3 else 1) if projected else CL
        assert selected.shape == (expected_rows, VOCAB)
        if step < 3:
            reader._advance_reasoning(0, logits)
            reader._advance_reasoning(1, selected, projected=projected)
        else:
            for slot, rows, compact in ((0, logits, False), (1, selected, projected)):
                canvas = reader._read_canvas(rows, slot, 0, 2, CL, projected=compact)
                reader._advance_read(slot, canvas, rows, projected=compact)
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            reader.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert torch.equal(
            states.read_reasoning[0].generator.get_state(),
            states.read_reasoning[1].generator.get_state(),
        )


@pytest.mark.parametrize(
    "answer_only,answer_noise,reason_positions,unmasked_feedback,token_noise,"
    "token_proposals",
    [
        (False, "candidates", (2, 3), False, "allowed", "allowed"),
        (False, "vocab", (2, 3), False, "allowed", "allowed"),
        (True, "candidates", (2, 3), False, "allowed", "allowed"),
        (True, "vocab", (2, 3), False, "allowed", "allowed"),
        pytest.param(
            True, "vocab", (0, 2), False, "allowed", "allowed", id="noncontiguous"
        ),
        pytest.param(True, "vocab", (2, 3), True, "allowed", "allowed", id="unmasked"),
        pytest.param(
            True, "vocab", (2, 3), False, "vocab", "allowed", id="token_noise"
        ),
        pytest.param(
            True, "vocab", (2, 3), False, "allowed", "vocab", id="token_proposals"
        ),
    ],
)
def test_reasoning_projected_logits_match_full_logits_through_final_output(
    sampler,
    answer_only,
    answer_noise,
    reason_positions,
    unmasked_feedback,
    token_noise,
    token_proposals,
    delayed_suffix=(),
    answer_aggregation="last",
):
    """Reduced projection must preserve the final full canvas and raw scores."""
    params = _read_params([1, 3], answer_only=answer_only)
    params.extra_args.update(
        diffusion_read_steps=2,
        diffusion_reasoning_positions=list(reason_positions),
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_reasoning_steps=3,
        diffusion_reasoning_answer_noise=answer_noise,
    )
    if unmasked_feedback:
        params.extra_args["diffusion_reasoning_unmasked_feedback"] = True
        sampler.embed_weight = (
            torch.arange(VOCAB * 4, device=DEVICE).reshape(VOCAB, 4).sin().bfloat16()
        )
    if token_noise == "vocab" or token_proposals == "vocab":
        _add(sampler, 3, deepcopy(params))
    if token_noise == "vocab":
        params.extra_args["diffusion_reasoning_token_noise"] = "vocab"
    if token_proposals == "vocab":
        params.extra_args["diffusion_reasoning_token_proposals"] = "vocab"
    if delayed_suffix:
        params.extra_args["diffusion_reasoning_delayed_suffix_positions"] = list(
            delayed_suffix
        )
    if answer_aggregation == "mean":
        params.extra_args["diffusion_answer_pass_aggregation"] = "mean"
    for slot in (0, 2):
        _add(sampler, slot, params)
    states = sampler.diffusion_states
    logits = torch.arange(CL * VOCAB, device=DEVICE).reshape(CL, VOCAB).float().sin()
    for step in range(5):
        full = sampler(logits, _batch([0]))
        expected_tokens = full.sampled_token_ids.clone()
        expected_count = full.num_sampled.clone()
        expected_scores = (
            full.logprobs_tensors.logprobs.clone()
            if full.logprobs_tensors is not None
            else None
        )
        selected = sampler.prepare_sample_hidden_states(logits, _batch([2]))
        expected_rows = 3 if step < 3 else (1 if step == 3 or answer_only else CL)
        assert selected.shape == (expected_rows, VOCAB)
        compact = sampler(selected, _batch([2]))
        if (token_noise == "vocab" or token_proposals == "vocab") and step < 3:
            sampler._advance_reasoning(3, logits)
            assert torch.equal(
                states.read_reasoning[0].generator.get_state(),
                states.read_reasoning[3].generator.get_state(),
            )
            if step == 2:
                torch.testing.assert_close(
                    states.canvas[0], states.canvas[3], rtol=0, atol=0
                )
        torch.testing.assert_close(compact.sampled_token_ids, expected_tokens)
        torch.testing.assert_close(compact.num_sampled, expected_count)
        if expected_scores is not None:
            assert compact.sampled_token_ids[0, 1].item() in (1, 3)
            torch.testing.assert_close(
                compact.logprobs_tensors.logprobs, expected_scores, rtol=0, atol=0
            )
        for values in (
            states.canvas,
            states.self_conditioning_embeds,
            sampler.req_states.draft_tokens,
        ):
            torch.testing.assert_close(values[0], values[2], rtol=0, atol=0)
        fixed = next(pos for pos in range(CL) if pos not in (*reason_positions, 1))
        assert (
            states.canvas[2, fixed].item()
            == params.extra_args["diffusion_seed_canvas"][fixed]
        )
        noise = sampler.req_states.draft_tokens[2, 1].item()
        assert 0 <= noise < VOCAB
        if answer_noise == "candidates":
            assert noise in (1, 3)


def test_answer_mean_gpu_projected_and_mixed_last_reads(sampler):
    """Mean uses the projected fallback while ordinary last reads retain scores."""
    test_reasoning_projected_logits_match_full_logits_through_final_output(
        sampler,
        True,
        "vocab",
        (2, 3),
        False,
        "allowed",
        "allowed",
        answer_aggregation="mean",
    )
    logits = torch.arange(CL * VOCAB, device=DEVICE).reshape(CL, VOCAB).float().sin()
    ordinary = _read_params([1, 3], answer_only=True)
    _add(sampler, 3, ordinary)
    selected = sampler.prepare_sample_hidden_states(logits, _batch([3]))
    assert sampler._read_logits_compacted
    baseline = sampler(selected, _batch([3])).logprobs_tensors.logprobs.clone()
    _add(sampler, 0, ordinary)
    mean = _read_params([1, 3, 5], answer_only=True)
    mean.extra_args.update(
        diffusion_reasoning_positions=[2, 3],
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_answer_pass_aggregation="mean",
    )
    _add(sampler, 2, mean)
    states = sampler.diffusion_states
    states.read_reasoning[2].step = states.read_reasoning[2].steps
    states.canvas[2] = states.read_requests[2][0]
    selected = sampler.prepare_sample_hidden_states(logits.repeat(2, 1), _batch([0, 2]))
    assert not sampler._read_logits_compacted
    result = sampler(selected, _batch([0, 2]))
    assert result.num_sampled.tolist() == [CL, CL]
    torch.testing.assert_close(
        result.logprobs_tensors.logprobs[0, :3], baseline[0], rtol=0, atol=0
    )
    assert not states.read_answer_means


def test_convergence_stop_gpu_counts_only_final_mixed_read_emissions(sampler, tmp_path):
    """Real projected and compact dispatch complete each child's counter once."""
    states = sampler.diffusion_states
    counter = diffusion_gemma.ReasoningCounterTrace(
        str(tmp_path), "convergence-stop-gpu", diffusion_gemma.__file__, 0
    )
    states.reasoning_counter_trace = counter
    sampler.req_states.index_to_req_id.update({0: "adaptive", 2: "control"})
    for slot in (0, 2):
        params = _read_params([1, 3], answer_only=True)
        params.extra_args["diffusion_seed_canvas"][1] = 1
        params.extra_args.update(
            diffusion_reasoning_positions=[2, 3],
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=4,
            diffusion_reasoning_seed=42,
            diffusion_reasoning_answer_noise="vocab",
        )
        if slot == 0:
            params.extra_args["diffusion_reasoning_convergence_stop"] = True
        _add(sampler, slot, params)
    logits = torch.zeros(CL, VOCAB, device=DEVICE)
    logits[1, 3] = 6
    logits[2, 9] = logits[3, 8] = 30
    logits[2:, 15] = 100
    for forward in range(5):
        slots = [0, 2] if forward < 3 else [2]
        batch = _batch(slots)
        selected = sampler.prepare_sample_hidden_states(
            logits.repeat(len(slots), 1), batch
        )
        assert sampler._read_logits_compacted == (forward == 4)
        expected_rows = (6, 6, 4, 3, 1)[forward]
        assert selected.shape == (expected_rows, VOCAB)
        result = sampler(selected, batch)
        expected_counts = ([0, 0], [0, 0], [CL, 0], [0], [CL])[forward]
        assert result.num_sampled.tolist() == expected_counts
        records = [json.loads(path.read_text()) for path in tmp_path.iterdir()]
        assert len(records) == (0 if forward < 2 else 1 if forward < 4 else 2)
        if forward in (2, 4):
            assert result.sampled_token_ids[0].tolist() == [10, 3, 9, 8]
            scores = result.logprobs_tensors
            assert scores is not None and scores.logprobs.shape[0] == 1
            ids = scores.logprob_token_ids[0].tolist()
            for token in (1, 3):
                torch.testing.assert_close(
                    scores.logprobs[0, ids.index(token)],
                    logits[1].log_softmax(-1)[token],
                )
        else:
            assert result.logprobs_tensors is None
        assert not sampler._pending_logprobs
        if forward == 2:
            assert set(counter.pending) == {2}
            assert counter.pending[2]["actual_reasoning_passes"] == 3
            assert counter.pending[2]["actual_answer_passes"] == 0
            assert records[0]["request_id"] == "adaptive"
            assert records[0]["actual_reasoning_passes"] == 2
            assert records[0]["actual_answer_passes"] == 1
            states.remove_request(0)
    assert not counter.pending
    by_id = {record["request_id"]: record for record in records}
    assert set(by_id) == {"adaptive", "control"}
    for name, passes in (("adaptive", 2), ("control", 4)):
        record = by_id[name]
        assert record["status"] == "completed"
        assert record["actual_reasoning_passes"] == passes
        assert record["requested_reasoning_passes"] == 4
        assert record["actual_answer_passes"] == record["requested_answer_passes"] == 1
        assert record["total_child_forwards"] == passes + 1
        assert record["terminal_reason"] == (
            "stable_confident" if passes == 2 else "cap"
        )
    states.remove_request(2)
    assert len(list(tmp_path.iterdir())) == 2  # Removal cannot add aborted duplicates.


def test_delayed_suffix_gpu_projection_mixed_spans_and_slot_reuse(sampler):
    """Real device indices stay in one child and disappear after cancellation."""
    test_reasoning_projected_logits_match_full_logits_through_final_output(
        sampler,
        True,
        "vocab",
        (0, 2),
        False,
        "allowed",
        "allowed",
        delayed_suffix=(3,),
    )
    params = _read_params([1, 3], answer_only=True)
    params.extra_args.update(
        diffusion_reasoning_positions=[0, 2],
        diffusion_reasoning_forbidden_token_ids=[15],
        diffusion_reasoning_steps=3,
    )
    _add(sampler, 1, params)
    delayed = deepcopy(params)
    delayed.extra_args["diffusion_reasoning_delayed_suffix_positions"] = [3]
    _add(sampler, 2, delayed)
    states = sampler.diffusion_states
    assert states.read_reasoning[1].delayed_suffix_positions is None
    assert states.read_reasoning[2].delayed_suffix_positions_cpu == (3,)
    weight = torch.arange(VOCAB * 4, device=DEVICE).reshape(VOCAB, 4).float()
    model_state = SimpleNamespace(
        model=SimpleNamespace(
            embed_input_ids=lambda ids: weight[ids] * 2,
            self_conditioning=SimpleNamespace(post_norm=lambda values: values),
        ),
        diffusion_states=states,
        _req_id_to_index={"delayed": 2},
    )
    for span in (2, CL):
        inputs = torch.full((span + CL, 4), 7.0, device=DEVICE)
        expected = inputs.clone()
        if span == CL:
            expected[3] = weight[0] * 2
        DiffusionGemmaModelState._apply_self_conditioning(
            model_state,
            np.array([2, 1]),
            np.array([0, 1]),
            np.array([0, span, span + CL]),
            inputs,
            states.self_conditioning_embeds,
            num_logits_np=np.array([span, CL]),
        )
        torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    DiffusionGemmaModelState.remove_request(model_state, "delayed")
    assert 2 not in states.read_reasoning
    _add(sampler, 2, params)
    assert states.read_reasoning[2].delayed_suffix_positions is None
    assert states.read_reasoning[2].delayed_suffix_positions_cpu == ()
    _add(sampler, 2, SamplingParams())
    assert 2 not in states.read_reasoning


@pytest.mark.parametrize("answer_noise", ["candidates", "vocab"])
def test_reasoning_cpu_answer_noise_domain_is_reproducible(answer_noise):
    """Noise can explore vocabulary IDs without changing retained predictions."""
    reader = _cpu_reasoning_sampler(answer_noise=answer_noise)
    prediction = torch.tensor([10, 3, 12, 13])
    drawn = []
    for _ in range(16):
        reader._stage_read_canvas(0, prediction)
        torch.rand(31)
        reader._stage_read_canvas(1, prediction)
        for slot in (0, 1):
            torch.testing.assert_close(reader.diffusion_states.canvas[slot], prediction)
        torch.testing.assert_close(
            reader.req_states.draft_tokens[0], reader.req_states.draft_tokens[1]
        )
        drawn.append(reader.req_states.draft_tokens[0, 1].item())
    assert all(0 <= token < VOCAB for token in drawn)
    if answer_noise == "candidates":
        assert set(drawn) == {1, 3}
    else:
        assert set(drawn) - {1, 3}


@pytest.mark.parametrize("ordinary_sibling", [False, True])
def test_reasoning_projection_keeps_heterogeneous_child_rows_independent(
    sampler, monkeypatch, ordinary_sibling
):
    """Sparse reasoning, canvas SC and a different sampler can share one batch."""
    for slot, answer, reason, scope in (
        (0, 1, [0, 2], "answer"),
        (2, 2, [1, 3], "canvas"),
    ):
        params = _read_params(
            [1, 3], positions=(answer,), answer_only=True, scope=scope
        )
        params.extra_args.update(
            diffusion_reasoning_positions=reason,
            diffusion_reasoning_forbidden_token_ids=[15],
            diffusion_reasoning_steps=3,
        )
        _add(sampler, slot, params)
    sibling = (
        SamplingParams()
        if ordinary_sibling
        else _read_params([1, 4], positions=(0,), answer_only=True)
    )
    _add(sampler, 1, sibling)
    stage = sampler._stage_read_canvas

    def stage_distinct_noise(slot, canvas):
        stage(slot, canvas)
        positions = sampler.diffusion_states.read_requests[slot][1]
        sampler.req_states.draft_tokens[slot, positions] = torch.where(
            canvas[positions] == 1, 3, 1
        )

    monkeypatch.setattr(sampler, "_stage_read_canvas", stage_distinct_noise)
    batch = _batch([0, 2, 1])
    logits = torch.arange(VOCAB, device=DEVICE).float().repeat(3 * CL, 1)
    original_offsets = batch.cu_num_logits_np.copy()
    selected = sampler.prepare_sample_hidden_states(logits, batch)
    torch.testing.assert_close(selected, logits[[0, 2, 1, *range(CL, 3 * CL)]])
    np.testing.assert_array_equal(batch.cu_num_logits_np, original_offsets)
    result = sampler(selected, batch)
    assert result.num_sampled.tolist() == [0, 0, 0 if ordinary_sibling else CL]
    if not ordinary_sibling:
        assert result.sampled_token_ids[2].tolist() == [4, 11, 12, 13]
        assert result.logprobs_tensors.logprobs.shape[0] == 1
    assert sampler.diffusion_states.canvas[0, 3].item() == 13
    assert sampler.diffusion_states.canvas[2, 0].item() == 10
    for slot, answer in ((0, 1), (2, 2)):
        assert sampler.req_states.draft_tokens[slot, answer].item() == 1
        assert sampler.diffusion_states.canvas[slot, answer].item() == 3


@pytest.mark.parametrize("scope", ["answer", "canvas"])
@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_cpu_regenerates_answers_from_legal_noise_each_pass(scope, enabled):
    """Reason and answer passes renew input noise, but emit the final prediction."""
    reader = _cpu_reasoning_sampler(scope=scope, enabled=enabled, answer_steps=3)
    states = reader.diffusion_states
    scored = []

    def score(logits, selected, *args, **kwargs):
        scored.append(selected.clone())
        return logits.log_softmax(-1)

    reader._compute_logprobs = score
    inputs, predictions = [], []
    for step in range(6):
        draft = reader.req_states.draft_tokens[0].clone()
        noise = draft[1].item()
        assert noise in (1, 3)
        inputs.append(noise)
        predicted = 3 if noise == 1 else 1
        predictions.append(predicted)
        logits = torch.zeros(CL, VOCAB)
        logits[1, 15] = 100
        logits[1, predicted] = 4
        logits[2, 9] = logits[3, 8] = 4
        if step < 3:
            reader._advance_reasoning(0, logits)
        else:
            canvas = reader._read_canvas(logits, 0, 0, 2, CL)
            assert canvas.tolist() == [10, predicted, 9, 8]
            if step < 5:
                reader._advance_read(0, canvas, logits)
            else:
                assert canvas[1].item() != noise
                assert scored[0].tolist() == [predicted]
                torch.testing.assert_close(
                    reader._pending_logprobs[0], logits[[1]].log_softmax(-1)
                )
                break
        assert not scored
        assert not reader._pending_logprobs
        assert states.canvas[0, 1].item() == predicted
        assert reader.req_states.draft_tokens[0, 1].item() in (1, 3)
        torch.testing.assert_close(
            reader.req_states.draft_tokens[0, [0, 2, 3]],
            states.canvas[0, [0, 2, 3]],
        )
    assert set(inputs) == {1, 3}
    assert set(predictions) == {1, 3}
    assert len(scored) == 1


@pytest.mark.parametrize("fused_prefill", [False, True])
@pytest.mark.parametrize("scope", [None, "canvas"])
def test_read_preserves_scaffold_and_reports_unfiltered_candidate_logprobs(
    sampler, monkeypatch, fused_prefill, scope
):
    """Candidate probabilities must not inherit annealing or top-k filtering."""
    _add(sampler, 2, _read_params([1, 3], positions=(1, 3), top_k=1, scope=scope))
    other = _read_params([2, 4, 5], positions=(2,))
    other.extra_args["diffusion_seed_canvas"] = [14, 11, 12, 9]
    _add(sampler, 0, other)
    logits = torch.arange(VOCAB, dtype=torch.float32, device=DEVICE).repeat(2 * CL, 1)

    def no_ordinary_sampling(*args, **kwargs):
        pytest.fail("read requests entered the ordinary denoising sampler")

    monkeypatch.setattr(diffusion_gemma, "_compiled_sample_step", no_ordinary_sampling)
    batch = _batch([2, 0])
    if fused_prefill:
        sampler.diffusion_states.is_encoder_phase[[2, 0]] = True
        batch.query_start_loc_np = np.array([0, 8, 16], dtype=np.int32)
        batch.query_start_loc = torch.tensor([0, 8, 16], device=DEVICE)
        batch.num_scheduled_tokens = np.array([8, 8], dtype=np.int32)
        batch.num_computed_prefill_tokens_np[:] = 0
    result = sampler(logits, batch)

    assert result.sampled_token_ids.tolist() == [[10, 3, 12, 3], [14, 11, 5, 9]]
    assert result.num_sampled.tolist() == [CL, CL]
    assert (
        bool(sampler.diffusion_states.is_encoder_phase[[2, 0]].all()) == fused_prefill
    )
    assert sampler.diffusion_states.step[[2, 0]].tolist() == [0, 0]
    lp = result.logprobs_tensors
    assert lp is not None
    assert lp.cu_num_generated_tokens == [0, CL]
    expected = logits.log_softmax(dim=-1)
    for row, ids in enumerate(lp.logprob_token_ids.tolist()):
        candidates = [1, 3] if row < CL else [2, 4, 5]
        for token in candidates:
            torch.testing.assert_close(
                lp.logprobs[row, ids.index(token)], expected[row, token]
            )
    assert not sampler._pending_logprobs


@pytest.mark.parametrize("fused_prefill", [False, True])
def test_read_projects_and_returns_only_answer_logprobs(
    sampler, fused_prefill, monkeypatch
):
    _add(sampler, 2, _read_params([1, 3], positions=(1, 3), answer_only=True))
    other = _read_params([2, 4, 5], positions=(2,), answer_only=True)
    other.extra_args["diffusion_seed_canvas"] = [14, 11, 12, 9]
    _add(sampler, 0, other)
    batch = _batch([2, 0])
    hidden_states = torch.arange(
        2 * CL * VOCAB, dtype=torch.float32, device=DEVICE
    ).reshape(2 * CL, VOCAB)
    if fused_prefill:
        batch.query_start_loc_np = np.array([0, 7, 12], dtype=np.int32)
        batch.query_start_loc = torch.tensor([0, 7, 12], device=DEVICE)
        batch.num_scheduled_tokens = np.array([7, 5], dtype=np.int32)
        batch.logits_indices = torch.tensor([3, 4, 5, 6, 8, 9, 10, 11], device=DEVICE)
        physical_hidden = torch.arange(
            12 * VOCAB, dtype=torch.float32, device=DEVICE
        ).reshape(12, VOCAB)
        hidden_states = physical_hidden[batch.logits_indices]

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.shape == (3, VOCAB)
    torch.testing.assert_close(selected, hidden_states[[1, 3, CL + 2]])

    def fail_collect(*args, **kwargs):
        raise AssertionError("compact reads must bypass per-request logprob collection")

    monkeypatch.setattr(sampler, "_collect_logprobs", fail_collect)

    result = sampler(selected, batch)
    assert result.sampled_token_ids.tolist() == [[10, 3, 12, 3], [14, 11, 5, 9]]
    logprobs = result.logprobs_tensors
    assert logprobs is not None
    assert logprobs.logprobs.shape[0] == 3
    assert logprobs.cu_num_generated_tokens == [0, 2]
    expected = hidden_states.log_softmax(dim=-1)
    source_rows_and_candidates = (
        (1, [1, 3]),
        (3, [1, 3]),
        (CL + 2, [2, 4, 5]),
    )
    for logprob_row, (source_row, candidates) in enumerate(source_rows_and_candidates):
        ids = logprobs.logprob_token_ids[logprob_row].tolist()
        for token in candidates:
            torch.testing.assert_close(
                logprobs.logprobs[logprob_row, ids.index(token)],
                expected[source_row, token],
            )
    assert not sampler._pending_logprobs


@pytest.mark.parametrize("answer_only", [None, False], ids=["omitted", "false"])
def test_native_read_keeps_all_logprob_rows_without_answer_only_opt_in(
    sampler, answer_only
):
    params = _read_params([1, 3])
    if answer_only is None:
        params.extra_args.pop("diffusion_answer_only_logprobs")
    _add(sampler, 0, params)
    batch = _batch([0])
    hidden_states = torch.randn(CL, VOCAB, device=DEVICE)

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.data_ptr() == hidden_states.data_ptr()
    result = sampler(selected, batch)
    assert result.logprobs_tensors is not None
    assert result.logprobs_tensors.logprobs.shape[0] == CL
    assert result.logprobs_tensors.cu_num_generated_tokens == [0]


def test_answer_only_read_keeps_neutral_full_rows_when_detokenizing(sampler):
    params = _read_params([1, 3], answer_only=True, detokenize=True)
    _add(sampler, 0, params)
    batch = _batch([0])
    hidden_states = torch.arange(
        CL * VOCAB, dtype=torch.float32, device=DEVICE
    ).reshape(CL, VOCAB)

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.data_ptr() == hidden_states.data_ptr()
    result = sampler(selected, batch)
    assert result.logprobs_tensors is not None
    logprobs = result.logprobs_tensors
    assert logprobs.logprobs.shape[0] == CL
    fixed_rows = [0, 2, 3]
    assert not logprobs.logprobs[fixed_rows].count_nonzero()
    expected_fixed_ids = result.sampled_token_ids[0, fixed_rows, None].expand(
        -1, logprobs.logprobs.shape[1]
    )
    torch.testing.assert_close(
        logprobs.logprob_token_ids[fixed_rows],
        expected_fixed_ids.to(logprobs.logprob_token_ids.dtype),
    )
    answer_ids = logprobs.logprob_token_ids[1].tolist()
    expected = hidden_states.log_softmax(dim=-1)
    for token in (1, 3):
        torch.testing.assert_close(
            logprobs.logprobs[1, answer_ids.index(token)], expected[1, token]
        )
    torch.testing.assert_close(logprobs.logprobs[:, 0].sum(), logprobs.logprobs[1, 0])


@pytest.mark.parametrize("other_read", [False, True])
def test_mixed_read_batch_keeps_all_logit_rows(sampler, other_read):
    _add(sampler, 0, _read_params([1, 3], answer_only=True))
    _add(sampler, 1, _read_params([2, 4]) if other_read else SamplingParams())
    batch = _batch([0, 1])
    hidden_states = torch.arange(
        2 * CL * VOCAB, dtype=torch.float32, device=DEVICE
    ).reshape(2 * CL, VOCAB)

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.data_ptr() == hidden_states.data_ptr()


def test_mixed_answer_only_and_native_reads_keep_per_request_row_counts(sampler):
    _add(sampler, 0, _read_params([1, 3], answer_only=True))
    _add(sampler, 1, _read_params([2, 4], answer_only=False))
    batch = _batch([0, 1])
    hidden_states = torch.arange(
        2 * CL * VOCAB, dtype=torch.float32, device=DEVICE
    ).reshape(2 * CL, VOCAB)

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.data_ptr() == hidden_states.data_ptr()
    result = sampler(selected, batch)

    assert result.sampled_token_ids.tolist() == [
        [10, 3, 12, 13],
        [10, 4, 12, 13],
    ]
    logprobs = result.logprobs_tensors
    assert logprobs is not None
    assert logprobs.logprobs.shape[0] == 1 + CL
    assert logprobs.cu_num_generated_tokens == [0, 1]
    expected = hidden_states.log_softmax(dim=-1)
    for logprob_row, (source_row, candidates) in enumerate(
        [(1, [1, 3]), *[(CL + row, [2, 4]) for row in range(CL)]]
    ):
        ids = logprobs.logprob_token_ids[logprob_row].tolist()
        for token in candidates:
            torch.testing.assert_close(
                logprobs.logprobs[logprob_row, ids.index(token)],
                expected[source_row, token],
            )


def test_compact_read_offsets_include_nonemitting_prefills(sampler, monkeypatch):
    _add(sampler, 2, _read_params([1, 3], positions=(1, 3), answer_only=True))
    _add(sampler, 1, SamplingParams())
    _add(sampler, 0, _read_params([2, 4], positions=(2,), answer_only=True))
    batch = _batch([2, 1, 0], nlogits=[CL, 0, CL])
    hidden_states = torch.arange(
        2 * CL * VOCAB, dtype=torch.float32, device=DEVICE
    ).reshape(2 * CL, VOCAB)

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.shape == (3, VOCAB)

    def fail_collect(*args, **kwargs):
        raise AssertionError("compact reads must bypass per-request logprob collection")

    monkeypatch.setattr(sampler, "_collect_logprobs", fail_collect)
    result = sampler(selected, batch)

    assert result.num_sampled.tolist() == [CL, 0, CL]
    assert result.logprobs_tensors is not None
    assert result.logprobs_tensors.logprobs.shape[0] == 3
    assert result.logprobs_tensors.cu_num_generated_tokens == [0, 2, 2]


def test_prefill_skips_output_projection(sampler):
    _add(sampler, 0, _read_params([1, 3]))
    batch = _batch([0], nlogits=[0], computed=0)
    batch.logits_indices = torch.tensor([1], device=DEVICE)
    hidden_states = torch.randn(2, VOCAB, device=DEVICE)

    selected = sampler.prepare_sample_hidden_states(
        hidden_states[batch.logits_indices], batch
    )
    assert selected.shape == (0, VOCAB)


def test_compact_read_omits_answers_beyond_a_truncated_canvas(sampler):
    _add(sampler, 2, _read_params([1, 3], positions=(1, 3), answer_only=True))
    batch = _batch([2], nlogits=[3])
    hidden_states = torch.arange(3 * VOCAB, dtype=torch.float32, device=DEVICE).reshape(
        3, VOCAB
    )

    selected = sampler.prepare_sample_hidden_states(hidden_states, batch)
    assert selected.shape == (1, VOCAB)
    torch.testing.assert_close(selected, hidden_states[[1]])

    result = sampler(selected, batch)
    assert result.sampled_token_ids[0, :3].tolist() == [10, 3, 12]
    assert result.num_sampled.tolist() == [3]
    assert result.logprobs_tensors is not None
    assert result.logprobs_tensors.logprobs.shape[0] == 1
    assert result.logprobs_tensors.cu_num_generated_tokens == [0]


@pytest.mark.parametrize("embedding_dtype", [torch.float32, torch.bfloat16])
def test_mixed_reads_only_consume_logprobs_for_emitting_requests(
    sampler, embedding_dtype
):
    """A newly converged row keeps its stash while another row commits."""
    sampler.embed_weight = sampler.embed_weight.to(embedding_dtype)
    _add(sampler, 0, SamplingParams(logprobs=1))
    first_logits = torch.arange(VOCAB, dtype=torch.float32, device=DEVICE).repeat(CL, 1)
    first = sampler(first_logits, _batch([0]))
    assert first.num_sampled.tolist() == [0]
    first_lp = sampler._pending_logprobs[0].logprobs.clone()

    _add(sampler, 1, SamplingParams(logprobs=2))
    _add(sampler, 2, _read_params([1, 3, 5]))
    logits = torch.zeros(3 * CL, VOCAB, device=DEVICE)
    logits[:, 3] = 4
    mixed = sampler(logits, _batch([2, 0, 1]))
    assert mixed.num_sampled.tolist() == [CL, CL, 0]
    assert mixed.sampled_token_ids[0].tolist() == [10, 3, 12, 13]
    assert mixed.sampled_token_ids[1].tolist() == [15] * CL
    assert set(sampler._pending_logprobs) == {1}
    lp = mixed.logprobs_tensors
    assert lp is not None
    assert lp.cu_num_generated_tokens == [0, CL, 2 * CL]
    torch.testing.assert_close(lp.logprobs[CL:, :2], first_lp)
    assert torch.isneginf(lp.logprobs[CL:, 2:]).all()

    second_lp = sampler._pending_logprobs[1].logprobs.clone()
    committed = sampler(torch.zeros(CL, VOCAB, device=DEVICE), _batch([1]))
    assert committed.num_sampled.tolist() == [CL]
    assert committed.logprobs_tensors is not None
    torch.testing.assert_close(committed.logprobs_tensors.logprobs, second_lp)
    assert not sampler._pending_logprobs


@pytest.mark.parametrize("mixed", [False, True])
def test_seed_is_installed_only_after_the_final_prompt_chunk(sampler, mixed):
    _add(sampler, 2, _read_params([1, 3]))
    slots = [2]
    if mixed:
        _add(sampler, 0, SamplingParams())
        slots.append(0)
    states = sampler.diffusion_states
    states.is_encoder_phase[slots] = True
    states.canvas[2].fill_(7)
    logits = torch.empty(0, VOCAB, device=DEVICE)
    rng_before = torch.cuda.get_rng_state(DEVICE)
    num_random = int(mixed) if states._fast_read_admission else len(slots)
    expected_random = torch.randint(0, VOCAB, (num_random, CL), device=DEVICE)
    torch.cuda.set_rng_state(rng_before, DEVICE)

    partial = sampler(logits, _batch(slots, [0] * len(slots), computed=0))
    assert partial.num_sampled.tolist() == [0] * len(slots)
    assert states.is_encoder_phase[2]
    assert states.canvas[2].tolist() == [7] * CL

    completed = sampler(logits, _batch(slots, [0] * len(slots), computed=2))
    assert completed.num_sampled.tolist() == [0] * len(slots)
    assert not states.is_encoder_phase[2]
    assert states.canvas[2].tolist() == [10, 11, 12, 13]
    assert sampler.req_states.draft_tokens[2].tolist() == [10, 11, 12, 13]
    if mixed:
        assert not states.is_encoder_phase[0]
        torch.testing.assert_close(states.canvas[0], expected_random[-1])
        torch.testing.assert_close(
            sampler.req_states.draft_tokens[0], expected_random[-1]
        )
    elif states._fast_read_admission:
        assert torch.equal(torch.cuda.get_rng_state(DEVICE), rng_before)


def test_short_read_uses_only_its_physical_canvas_prefix(sampler):
    params = _read_params([1, 3])
    params.max_tokens = 3
    params.extra_args["diffusion_seed_canvas"] = [10, 11, 12]
    _add(sampler, 2, params)
    states = sampler.diffusion_states
    states.is_encoder_phase[2] = True
    logits = torch.empty(0, VOCAB, device=DEVICE)

    sampler(logits, _batch([2], [0], computed=4))
    assert states.canvas[2, :3].tolist() == [10, 11, 12]
    assert sampler.req_states.draft_tokens[2, :3].tolist() == [10, 11, 12]

    logits = torch.zeros(3, VOCAB, device=DEVICE)
    logits[:, 3] = 4
    result = sampler(logits, _batch([2], [3]))
    assert result.num_sampled.tolist() == [3]
    assert result.sampled_token_ids[0, :3].tolist() == [10, 3, 12]


def test_reused_slot_does_not_inherit_read_constraints(sampler):
    _add(sampler, 2, _read_params([1, 3]))
    sampler.diffusion_states.remove_request(2)
    assert 2 not in sampler.diffusion_states.read_requests
    _add(sampler, 2, SamplingParams())
    assert 2 not in sampler.diffusion_states.read_requests
    assert sampler.logprob_token_ids_state.num_token_ids.np[2] == 0


def _split_metadata_state():
    staging = diffusion_gemma.UvaBackedTensor
    return SimpleNamespace(
        max_num_reqs=8,
        _split_query_staging=staging(17, dtype=torch.int32),
        _split_logits_staging=staging(8, dtype=torch.int32),
        _split_read_staging=staging(8, dtype=torch.bool),
        _split_query=torch.full((17,), -1, dtype=torch.int32, device=DEVICE),
        _split_seq_lens=torch.full((16,), -1, dtype=torch.int32, device=DEVICE),
        _split_causal=torch.ones(16, dtype=torch.bool, device=DEVICE),
        _split_block_tables={},
        _mm_prefix_range_copy_pending=False,
        _mm_prefix_range_copy_event=None,
        diffusion_states=SimpleNamespace(read_requests={}),
    )


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("pure_prefill", [False, True])
@pytest.mark.parametrize("read_slots", [(), (0, 2, 5), tuple(range(8))])
def test_fused_read_metadata_survives_slot_and_batch_shape_changes(
    padded, pure_prefill, read_slots
):
    """Reuse graph storage without mixing slots, cache groups or stale rows."""
    state = _split_metadata_state()
    state.diffusion_states.read_requests = dict.fromkeys(read_slots)
    widths = [512] * 5 + [256]
    tables = [
        torch.arange(8 * width * 2, dtype=torch.int32, device=DEVICE).reshape(
            8, width * 2
        )[:, ::2]
        for width in widths
    ]
    pointers: dict[int, int] = {}
    for step, count in enumerate((8, 3, 1, 8)):
        num_reqs = 8 if padded else count
        slots = np.roll(np.array([7, 2, 5, 0, 6, 1, 4, 3]), step)[:count]
        tails = np.array([31, 895, 0, 672, 31, 4, 1052, 16])[:count]
        logits = np.array([249, 112, 58, 0, 256, 1, 0, 32])[:count]
        if pure_prefill:
            logits = np.zeros_like(logits)
        starts = np.cumsum(np.r_[0, tails + logits], dtype=np.int32)
        seqs = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
        seqs[:count] = torch.as_tensor(1052 + logits, device=DEVICE)
        causal = torch.zeros(num_reqs, dtype=torch.bool, device=DEVICE)
        causal[:count] = torch.as_tensor(slots % 2 == 0, device=DEVICE)
        batch = SimpleNamespace(
            num_reqs=count,
            num_draft_tokens=int(logits.sum()),
            idx_mapping_np=slots,
            query_start_loc_np=starts,
            cu_num_logits_np=np.cumsum(np.r_[0, logits], dtype=np.int32),
            seq_lens=seqs,
        )
        old_seq = state._split_seq_lens.clone()
        old_causal = state._split_causal.clone()
        metadata, old_tables = {}, {}
        for group, table in enumerate(tables):
            table.add_(10000 + group)
            table[count:].zero_()
            key = table.data_ptr()
            if key in state._split_block_tables:
                old_tables[key] = state._split_block_tables[key].clone()
            value = SimpleNamespace(block_table=table, mm_prefix_range_tensor=None)
            metadata[f"group{group}"] = value
            metadata[f"alias{group}"] = value
        DiffusionGemmaModelState._split_read_attention(
            state, metadata, batch, causal, num_reqs
        )
        expected_query = [int(starts[-1])] * 17
        for i in range(count):
            expected_query[2 * i : 2 * i + 3] = [
                int(starts[i]),
                int(starts[i + 1] - logits[i]),
                int(starts[i + 1]),
            ]
        assert state._split_query.tolist() == expected_query
        expected_seq = old_seq
        expected_causal = old_causal
        for i in range(num_reqs):
            expected_seq[2 * i] = seqs[i] - (int(logits[i]) if i < count else 0)
            expected_seq[2 * i + 1] = seqs[i]
            expected_causal[2 * i] = causal[i]
            expected_causal[2 * i + 1] = causal[i] & (
                i >= count or int(slots[i]) not in read_slots
            )
        assert torch.equal(state._split_seq_lens, expected_seq)
        assert torch.equal(state._split_causal, expected_causal)
        for group, table in enumerate(tables):
            value = metadata[f"group{group}"]
            key = table.data_ptr()
            output = state._split_block_tables[key]
            pointers.setdefault(key, output.data_ptr())
            assert output.data_ptr() == pointers[key] == value.block_table.data_ptr()
            assert metadata[f"alias{group}"] is value
            assert torch.equal(value.block_table[::2], table[:num_reqs])
            assert torch.equal(value.block_table[1::2], table[:num_reqs])
            if key in old_tables:
                assert torch.equal(
                    output[2 * num_reqs :], old_tables[key][2 * num_reqs :]
                )
            assert value.query_start_loc.data_ptr() == state._split_query.data_ptr()
            assert value.seq_lens.data_ptr() == state._split_seq_lens.data_ptr()
            assert value.causal.data_ptr() == state._split_causal.data_ptr()
            assert value.query_start_loc.shape == (2 * num_reqs + 1,)
            assert value.seq_lens.shape == value.causal.shape == (2 * num_reqs,)
            assert value.max_query_len == max(
                1, max(np.diff(expected_query[: 2 * num_reqs + 1]))
            )
