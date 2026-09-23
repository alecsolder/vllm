# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.config import DiffusionConfig
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.fixture
def diffusion_read_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=4,
        ignore_eos=True,
        logprob_token_ids=[5, 6],
        extra_args={
            "diffusion_read_only": True,
            "diffusion_seed_canvas": [1, 2, 3, 4],
            "diffusion_answer_positions": [1, 3],
        },
    )


@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_read_accepts_complete_fixed_canvas(diffusion_read_params, flag):
    diffusion_read_params.extra_args["diffusion_read_only"] = flag
    diffusion_read_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("diffusion_read_only", "true"),
        ("diffusion_read_only", 1.0),
        ("diffusion_read_only", 2),
        ("diffusion_read_only", None),
        ("diffusion_answer_only_logprobs", "true"),
        ("diffusion_answer_only_logprobs", 2),
        ("diffusion_read_steps", True),
        ("diffusion_read_steps", 0),
        ("diffusion_read_steps", 33),
        ("diffusion_read_steps", "2"),
        ("diffusion_read_steps", 2.0),
        ("diffusion_answer_pass_aggregation", None),
        ("diffusion_answer_pass_aggregation", True),
        ("diffusion_answer_pass_aggregation", "sum"),
        ("diffusion_read_self_conditioning", "false"),
        ("diffusion_read_self_conditioning", 2),
        ("diffusion_read_self_conditioning_scope", "all"),
        ("diffusion_read_self_conditioning_scope", True),
        ("diffusion_read_self_conditioning_scope", None),
        ("diffusion_read_self_conditioning_scope", []),
        ("diffusion_seed_canvas", [1, 2, 3, 4, 5]),
        ("diffusion_seed_canvas", (1, 2, 3, 4)),
        ("diffusion_seed_canvas", [1, 2, 3, True]),
        ("diffusion_seed_canvas", [1, 2, 3, 4.0]),
        ("diffusion_seed_canvas", [1, 2, 3, -1]),
        ("diffusion_seed_canvas", [1, 2, 3, 1024]),
        ("diffusion_answer_positions", []),
        ("diffusion_answer_positions", [1, 1]),
        ("diffusion_answer_positions", [4]),
        ("diffusion_answer_positions", [-1]),
        ("diffusion_answer_positions", [True]),
        ("diffusion_answer_positions", [1.0]),
        ("diffusion_answer_positions", "1"),
    ],
)
def test_diffusion_read_rejects_malformed_canvas_options(
    diffusion_read_params, key, value
):
    diffusion_read_params.extra_args[key] = value
    with pytest.raises(VLLMValidationError, match=key):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize(
    "key",
    ["diffusion_read_only", "diffusion_seed_canvas", "diffusion_answer_positions"],
)
def test_diffusion_read_rejects_partial_request(diffusion_read_params, key):
    del diffusion_read_params.extra_args[key]
    with pytest.raises(VLLMValidationError, match="diffusion_"):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


def test_diffusion_read_accepts_canvas_shorter_than_served_capacity(
    diffusion_read_params,
):
    diffusion_read_params.max_tokens = 3
    diffusion_read_params.extra_args["diffusion_seed_canvas"] = [1, 2, 3]
    diffusion_read_params.extra_args["diffusion_answer_positions"] = [1, 2]
    diffusion_read_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("logprob_token_ids", None, "nonempty logprob_token_ids"),
        ("logprob_token_ids", [], "nonempty logprob_token_ids"),
        ("logprob_token_ids", [1024], "out-of-vocab"),
        ("max_tokens", 3, "max_tokens=4"),
        ("max_tokens", 5, "max_tokens=4"),
        ("ignore_eos", False, "ignore_eos=True"),
        ("stop", ["stop"], "does not support stop"),
        ("stop_token_ids", [1], "does not support stop"),
        ("top_k", 2, "top_k, top_p, or penalties"),
        ("top_p", 0.9, "top_k, top_p, or penalties"),
        ("presence_penalty", 1.0, "top_k, top_p, or penalties"),
        ("frequency_penalty", 1.0, "top_k, top_p, or penalties"),
        ("repetition_penalty", 1.1, "top_k, top_p, or penalties"),
    ],
)
def test_diffusion_read_preserves_all_decision_scores(
    diffusion_read_params, key, value, match
):
    setattr(diffusion_read_params, key, value)
    with pytest.raises(VLLMValidationError, match=match):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


def test_diffusion_read_requires_served_canvas_config(diffusion_read_params):
    with pytest.raises(VLLMValidationError, match="configured diffusion canvas length"):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True), None, None, None
        )


@pytest.mark.parametrize("key", ["diffusion_canvas_length", "diffusion_max_steps"])
def test_diffusion_read_rejects_width_and_step_overrides(diffusion_read_params, key):
    diffusion_read_params.extra_args[key] = 1
    with pytest.raises(VLLMValidationError, match="overrides are not supported"):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


def test_diffusion_read_rejected_for_autoregressive_model(diffusion_read_params):
    with pytest.raises(VLLMValidationError, match="require a diffusion model"):
        diffusion_read_params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize("steps", [1, 2, 32])
@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("scope", ["answer", "canvas"])
def test_diffusion_read_accepts_request_time_refinement(
    diffusion_read_params, steps, self_conditioning, scope
):
    diffusion_read_params.extra_args.update(
        diffusion_read_steps=steps,
        diffusion_read_self_conditioning=self_conditioning,
        diffusion_read_self_conditioning_scope=scope,
    )
    diffusion_read_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize("read_only", [False, None])
@pytest.mark.parametrize(
    "extra_args",
    [
        {"diffusion_answer_only_logprobs": True},
        {"diffusion_read_self_conditioning_scope": "canvas"},
        {"diffusion_reasoning_positions": [0]},
        {"diffusion_reasoning_forbidden_token_ids": []},
        {"diffusion_reasoning_steps": 8},
        {"diffusion_reasoning_seed": 0},
        {"diffusion_reasoning_known_positions": [0]},
        pytest.param(
            {"diffusion_reasoning_delayed_suffix_positions": [2]},
            id="delayed_suffix_requires_read",
        ),
        {"diffusion_reasoning_answer_noise": "vocab"},
        {"diffusion_reasoning_token_noise": "vocab"},
        {"diffusion_reasoning_token_proposals": "vocab"},
        {"diffusion_reasoning_schedule_steps": 48},
        {"diffusion_reasoning_entropy_bound": 0.1},
        {"diffusion_reasoning_round_feedback_logits": True},
        {"diffusion_reasoning_preserve_feedback": False},
        {"diffusion_reasoning_answer_feedback": True},
        {"diffusion_reasoning_unmasked_feedback": True},
    ],
)
def test_read_options_require_diffusion_read(read_only, extra_args):
    extra_args = extra_args.copy()
    if read_only is not None:
        extra_args["diffusion_read_only"] = read_only
    params = SamplingParams(extra_args=extra_args)
    with pytest.raises(VLLMValidationError, match="diffusion_read_only=True"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


@pytest.mark.parametrize("flag", [False, 0])
def test_diffusion_read_can_be_disabled_without_changing_generation(flag):
    params = SamplingParams(extra_args={"diffusion_read_only": flag})
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


@pytest.fixture
def diffusion_reason_params(diffusion_read_params):
    diffusion_read_params.extra_args.update(
        diffusion_answer_positions=[3],
        diffusion_reasoning_positions=[0, 1],
        diffusion_reasoning_forbidden_token_ids=[0, 2],
    )
    return diffusion_read_params


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"diffusion_answer_only_logprobs": False},
        {"diffusion_reasoning_preserve_feedback": True},
        {"diffusion_reasoning_positions": None},
    ],
)
def test_answer_mean_requires_independent_post_reason_reads(
    diffusion_reason_params, override
):
    extra = diffusion_reason_params.extra_args
    extra.update(
        diffusion_answer_pass_aggregation="mean",
        diffusion_answer_only_logprobs=True,
        diffusion_read_steps=4,
    )
    extra.update(override)
    original = deepcopy(extra)
    args = (
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    if override:
        with pytest.raises(VLLMValidationError, match="Mean answer reads"):
            diffusion_reason_params.verify(*args)
    else:
        diffusion_reason_params.verify(*args)
        assert extra == original


@pytest.mark.parametrize("steps", [None, 1, 32, 48, 64])
@pytest.mark.parametrize("forbidden", [[], [0, 1023]])
@pytest.mark.parametrize("answer_noise", [None, "candidates", "vocab"])
def test_reasoning_accepts_bounded_span_and_retains_answer_steps(
    diffusion_reason_params, steps, forbidden, answer_noise
):
    extra = diffusion_reason_params.extra_args
    extra["diffusion_reasoning_forbidden_token_ids"] = forbidden
    extra["diffusion_read_steps"] = 3
    if steps is not None:
        extra["diffusion_reasoning_steps"] = steps
    if answer_noise is not None:
        extra["diffusion_reasoning_answer_noise"] = answer_noise
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize("name", ["token_noise", "token_proposals"])
@pytest.mark.parametrize("mode", ["allowed", "vocab"])
def test_reasoning_token_modes_preserve_initial_canvas(
    diffusion_reason_params, name, mode
):
    extra = diffusion_reason_params.extra_args
    extra[f"diffusion_reasoning_{name}"] = mode
    original = deepcopy(extra)
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    assert extra == original


@pytest.mark.parametrize("seed", [None, 0, 2**64 - 1])
def test_reasoning_seed_accepts_unsigned64_without_mutating_input(
    diffusion_reason_params, seed
):
    diffusion_reason_params.extra_args["diffusion_reasoning_seed"] = seed
    original = diffusion_reason_params.extra_args.copy()
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    assert diffusion_reason_params.extra_args == original


def test_reasoning_accepts_fixed_tokens_between_mutable_sections(
    diffusion_reason_params,
):
    diffusion_reason_params.extra_args["diffusion_reasoning_positions"] = [0, 2]
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize("is_diffusion", [False, True])
@pytest.mark.parametrize(
    "name", ["seed", "known_positions", "delayed_suffix_positions"]
)
def test_null_reasoning_control_keeps_ordinary_generation_unchanged(is_diffusion, name):
    extra = {f"diffusion_reasoning_{name}": None}
    params = SamplingParams(extra_args=extra)
    params.verify(MockModelConfig(is_diffusion=is_diffusion), None, None, None)
    assert params.extra_args == extra


@pytest.mark.parametrize(
    "name,value",
    [
        ("seed", None),
        ("seed", 0),
        ("known_positions", None),
        ("known_positions", [2]),
        ("delayed_suffix_positions", None),
        ("delayed_suffix_positions", [2]),
    ],
)
def test_only_nonnull_reasoning_control_requires_reasoning_span(
    diffusion_read_params, name, value
):
    diffusion_read_params.extra_args[f"diffusion_reasoning_{name}"] = value
    args = (
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    if value is None:
        diffusion_read_params.verify(*args)
    else:
        with pytest.raises(VLLMValidationError, match="diffusion_reasoning_positions"):
            diffusion_read_params.verify(*args)


@pytest.mark.parametrize("positions", [None, [2]])
@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("name", ["known_positions", "delayed_suffix_positions"])
def test_known_positions_accept_fixed_rows_without_mutating_parameters(
    diffusion_reason_params, positions, self_conditioning, name
):
    extra = diffusion_reason_params.extra_args
    extra[f"diffusion_reasoning_{name}"] = positions
    extra["diffusion_read_self_conditioning"] = self_conditioning
    original = deepcopy(extra)
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    assert extra == original


@pytest.mark.parametrize(
    "reasoning,delayed",
    [
        pytest.param([0, 2], [1], id="fixed_gap_before_last_reason"),
        pytest.param([0], [2, 1], id="reversed_valid_suffix_slots"),
    ],
)
def test_delayed_suffix_requires_ordered_slots_after_all_reasoning(
    diffusion_reason_params, reasoning, delayed
):
    extra = diffusion_reason_params.extra_args
    extra["diffusion_reasoning_positions"] = reasoning
    extra["diffusion_reasoning_delayed_suffix_positions"] = delayed
    original = deepcopy(extra)
    with pytest.raises(
        VLLMValidationError, match="diffusion_reasoning_delayed_suffix_positions"
    ):
        diffusion_reason_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )
    assert extra == original


@pytest.mark.parametrize(
    ("reason_steps", "schedule_steps"), [(8, 8), (8, 48), (48, 48), (64, 64)]
)
@pytest.mark.parametrize("entropy_bound", [0, 0.1, 1.0])
def test_reasoning_accepts_independent_schedule_and_entropy_budget(
    diffusion_reason_params, reason_steps, schedule_steps, entropy_bound
):
    """A longer annealing horizon does not increase the actual pass budget."""
    diffusion_reason_params.extra_args.update(
        diffusion_reasoning_steps=reason_steps,
        diffusion_reasoning_schedule_steps=schedule_steps,
        diffusion_reasoning_entropy_bound=entropy_bound,
    )
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )


@pytest.mark.parametrize("round_logits", [False, True])
@pytest.mark.parametrize("preserve_feedback", [False, True])
@pytest.mark.parametrize("answer_feedback", [False, True])
@pytest.mark.parametrize("unmasked_feedback", [False, True])
def test_reasoning_accepts_independent_feedback_controls(
    diffusion_reason_params,
    round_logits,
    preserve_feedback,
    answer_feedback,
    unmasked_feedback,
):
    diffusion_reason_params.extra_args.update(
        diffusion_reasoning_round_feedback_logits=round_logits,
        diffusion_reasoning_preserve_feedback=preserve_feedback,
        diffusion_reasoning_answer_feedback=answer_feedback,
        diffusion_reasoning_unmasked_feedback=unmasked_feedback,
    )
    before = dict(diffusion_reason_params.extra_args)
    diffusion_reason_params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    assert diffusion_reason_params.extra_args == before


@pytest.mark.parametrize(
    "name",
    [
        "diffusion_reasoning_round_feedback_logits",
        "diffusion_reasoning_preserve_feedback",
        "diffusion_reasoning_answer_feedback",
        "diffusion_reasoning_unmasked_feedback",
        "diffusion_reasoning_convergence_stop",
    ],
)
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_reasoning_feedback_controls_reject_non_booleans(
    diffusion_reason_params, name, value
):
    diffusion_reason_params.extra_args[name] = value
    with pytest.raises(VLLMValidationError, match=name):
        diffusion_reason_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_convergence_stop_requires_reasoning_only_when_enabled(
    request, reasoning, enabled
):
    params = request.getfixturevalue(
        "diffusion_reason_params" if reasoning else "diffusion_read_params"
    )
    params.extra_args["diffusion_reasoning_convergence_stop"] = enabled
    before = deepcopy(params.extra_args)
    args = (
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        DiffusionConfig(canvas_length=4),
    )
    if enabled and not reasoning:
        with pytest.raises(VLLMValidationError, match="diffusion_reasoning_positions"):
            params.verify(*args)
    else:
        params.verify(*args)
    assert params.extra_args == before


@pytest.mark.parametrize(
    "name",
    [
        "diffusion_reasoning_round_feedback_logits",
        "diffusion_reasoning_preserve_feedback",
        "diffusion_reasoning_answer_feedback",
        "diffusion_reasoning_unmasked_feedback",
    ],
)
@pytest.mark.parametrize("value", [False, True])
def test_reasoning_feedback_controls_require_a_reasoning_span(
    diffusion_read_params, name, value
):
    diffusion_read_params.extra_args[name] = value
    with pytest.raises(VLLMValidationError, match="diffusion_reasoning_positions"):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("diffusion_reasoning_positions", None),
        ("diffusion_reasoning_positions", []),
        ("diffusion_reasoning_positions", [0, 0]),
        ("diffusion_reasoning_positions", [1, 0]),
        ("diffusion_reasoning_positions", [2, 3]),
        ("diffusion_reasoning_positions", [4]),
        ("diffusion_reasoning_positions", [-1]),
        ("diffusion_reasoning_positions", [True]),
        ("diffusion_reasoning_positions", [0.0]),
        ("diffusion_reasoning_positions", (0, 1)),
        ("diffusion_reasoning_known_positions", []),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [],
            id="delayed_suffix_empty",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [2, 2],
            id="delayed_suffix_duplicate",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [3, 2],
            id="delayed_suffix_unsorted",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [0],
            id="delayed_suffix_reason_overlap",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [1],
            id="delayed_suffix_last_reason_overlap",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [3],
            id="delayed_suffix_answer_overlap",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [4],
            id="delayed_suffix_out_of_range",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [-1],
            id="delayed_suffix_negative",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [True],
            id="delayed_suffix_bool",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            [2.0],
            id="delayed_suffix_float",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            ["2"],
            id="delayed_suffix_string",
        ),
        pytest.param(
            "diffusion_reasoning_delayed_suffix_positions",
            (2,),
            id="delayed_suffix_tuple",
        ),
        ("diffusion_reasoning_known_positions", [2, 2]),
        ("diffusion_reasoning_known_positions", [2, 0]),
        ("diffusion_reasoning_known_positions", [0]),
        ("diffusion_reasoning_known_positions", [3]),
        ("diffusion_reasoning_known_positions", [4]),
        ("diffusion_reasoning_known_positions", [-1]),
        ("diffusion_reasoning_known_positions", [True]),
        ("diffusion_reasoning_known_positions", [2.0]),
        ("diffusion_reasoning_known_positions", (2,)),
        ("diffusion_reasoning_forbidden_token_ids", None),
        ("diffusion_reasoning_forbidden_token_ids", [0, 0]),
        ("diffusion_reasoning_forbidden_token_ids", [-1]),
        ("diffusion_reasoning_forbidden_token_ids", [1024]),
        ("diffusion_reasoning_forbidden_token_ids", [True]),
        ("diffusion_reasoning_forbidden_token_ids", [0.0]),
        ("diffusion_reasoning_forbidden_token_ids", list(range(1024))),
        ("diffusion_reasoning_steps", None),
        ("diffusion_reasoning_steps", 0),
        ("diffusion_reasoning_steps", 65),
        ("diffusion_reasoning_steps", True),
        ("diffusion_reasoning_steps", "8"),
        ("diffusion_reasoning_seed", -1),
        ("diffusion_reasoning_seed", 2**64),
        ("diffusion_reasoning_seed", True),
        ("diffusion_reasoning_seed", False),
        ("diffusion_reasoning_seed", 1.0),
        ("diffusion_reasoning_seed", "1"),
        ("diffusion_reasoning_schedule_steps", None),
        ("diffusion_reasoning_schedule_steps", 7),
        ("diffusion_reasoning_schedule_steps", True),
        ("diffusion_reasoning_schedule_steps", "48"),
        ("diffusion_reasoning_entropy_bound", None),
        ("diffusion_reasoning_entropy_bound", -0.1),
        ("diffusion_reasoning_entropy_bound", float("nan")),
        ("diffusion_reasoning_entropy_bound", float("inf")),
        ("diffusion_reasoning_entropy_bound", 10**400),
        ("diffusion_reasoning_entropy_bound", True),
        ("diffusion_reasoning_entropy_bound", "0.1"),
        ("diffusion_reasoning_answer_noise", None),
        ("diffusion_reasoning_answer_noise", "all"),
        ("diffusion_reasoning_answer_noise", True),
        ("diffusion_reasoning_answer_noise", 0),
        ("diffusion_reasoning_answer_noise", []),
        ("diffusion_reasoning_answer_noise", {}),
        ("diffusion_reasoning_token_noise", None),
        ("diffusion_reasoning_token_noise", "candidates"),
        ("diffusion_reasoning_token_noise", True),
        ("diffusion_reasoning_token_noise", 0),
        ("diffusion_reasoning_token_noise", []),
        ("diffusion_reasoning_token_proposals", None),
        ("diffusion_reasoning_token_proposals", "candidates"),
        ("diffusion_reasoning_token_proposals", True),
        ("diffusion_reasoning_token_proposals", 0),
        ("diffusion_reasoning_token_proposals", []),
    ],
)
def test_reasoning_rejects_invalid_or_overlapping_mutable_span(
    diffusion_reason_params, key, value
):
    diffusion_reason_params.extra_args[key] = value
    with pytest.raises(VLLMValidationError, match=key):
        diffusion_reason_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize(
    "key", ["diffusion_reasoning_positions", "diffusion_reasoning_forbidden_token_ids"]
)
def test_reasoning_rejects_incomplete_options(diffusion_reason_params, key):
    del diffusion_reason_params.extra_args[key]
    with pytest.raises(VLLMValidationError, match=key):
        diffusion_reason_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize(
    "key,mode",
    [
        ("answer_noise", "candidates"),
        ("answer_noise", "vocab"),
        ("token_noise", "allowed"),
        ("token_noise", "vocab"),
        ("token_proposals", "allowed"),
        ("token_proposals", "vocab"),
    ],
)
def test_reasoning_token_modes_require_a_reasoning_span(
    diffusion_read_params, key, mode
):
    diffusion_read_params.extra_args[f"diffusion_reasoning_{key}"] = mode
    with pytest.raises(VLLMValidationError, match="diffusion_reasoning_positions"):
        diffusion_read_params.verify(
            MockModelConfig(is_diffusion=True),
            None,
            None,
            None,
            DiffusionConfig(canvas_length=4),
        )


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]
