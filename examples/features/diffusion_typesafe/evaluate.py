# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate a live diffusion TypeSafe adapter through the public Python SDK."""

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgspec
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
)


@dataclass
class Case:
    name: str
    state: dict[str, Any]
    questions: dict[str, Any]
    expected: dict[str, str | int | bool]


def labeled_cases() -> list[Case]:
    binary_cases = [
        ({"count": 3, "limit": 5}, "Is count below limit?", "Is count equal to limit?"),
        ({"door": "closed"}, "Is the door closed?", "Is the door open?"),
        ({"animal": "cat"}, "Is the animal a cat?", "Is the animal a bird?"),
        ({"invoice": {"paid": True}}, "Is the invoice paid?", "Is the invoice unpaid?"),
        ({"colors": ["red", "blue"]}, "Is red in colors?", "Is green in colors?"),
        ({"country": "France"}, "Is the country France?", "Is the country Germany?"),
        (
            {"temperature_c": -2},
            "Is the temperature negative?",
            "Is the temperature positive?",
        ),
        ({"status": "complete"}, "Is the status complete?", "Is the status pending?"),
    ]
    cases = [
        Case(
            f"noul_{i}",
            state,
            {"yes": Noul(instructions=yes), "no": Noul(instructions=no)},
            {"yes": True, "no": False},
        )
        for i, (state, yes, no) in enumerate(binary_cases)
    ]
    for name, state, instructions, criteria, expected in [
        (
            "ticket",
            {"ticket": "My invoice was charged twice."},
            "Which category describes the ticket?",
            {
                "billing": "Invoices and payments",
                "technical": "Software errors",
                "other": None,
            },
            "billing",
        ),
        (
            "status",
            {"status": "queued"},
            "Select the exact status recorded in state.",
            {"queued": None, "running": None, "complete": None, "failed": None},
            "queued",
        ),
        (
            "unicode",
            {"color": "blue"},
            "Select the label whose description matches the color.",
            {"rouge": "red", "蓝色": "blue", "緑": "green"},
            "蓝色",
        ),
        (
            "twenty_one_labels",
            {"selected": "option_20"},
            "Select the exact value of selected.",
            {f"option_{i:02d}": None for i in range(21)},
            "option_20",
        ),
        (
            "twenty_six_labels",
            {"selected": "option_25"},
            "Select the exact value of selected.",
            {f"option_{i:02d}": None for i in range(26)},
            "option_25",
        ),
        (
            "single_label",
            {"selected": "only"},
            "Select the only available label.",
            {"only": None},
            "only",
        ),
        (
            "long_labels",
            {"problem": "A screen reader cannot announce the button."},
            "Classify the problem.",
            {"accessibility / screen reader": None, "account / billing": None},
            "accessibility / screen reader",
        ),
        (
            "nested_choice_criteria",
            {"priority": "high"},
            "Select the label whose description matches priority.",
            {"routine": {"priority": "low"}, "urgent": {"priority": "high"}},
            "urgent",
        ),
    ]:
        cases.append(
            Case(
                name,
                state,
                {"answer": Choice(instructions=instructions, criteria=criteria)},
                {"answer": expected},
            )
        )

    for count in (0, 3):
        cases.append(
            Case(
                f"score_count_{count}",
                {"count": count},
                {
                    "answer": Score(
                        instructions="Rate the number of items recorded in count.",
                        criteria=["zero items", "one item", "two items", "three items"],
                    )
                },
                {"answer": count},
            )
        )
    cases.extend(
        [
            Case(
                "score_nested_objects",
                {"severity": "minor", "signals": ["cosmetic"]},
                {
                    "answer": Score(
                        instructions=(
                            "Choose the rubric level that matches severity and signals."
                        ),
                        criteria=[
                            {"severity": "none", "signals": []},
                            {"severity": "minor", "signals": ["cosmetic"]},
                            {"severity": "major", "signals": ["outage"]},
                        ],
                    )
                },
                {"answer": 1},
            ),
            Case(
                "score_nested_lists",
                {"outcome": "passed"},
                {
                    "answer": Score(
                        instructions="Choose the level whose rubric describes outcome.",
                        criteria=[
                            [{"outcome": "failed"}, "unsuccessful"],
                            [{"outcome": "passed"}, "successful"],
                        ],
                    )
                },
                {"answer": 1},
            ),
        ]
    )
    return cases


def validate_answer(question: Any, answer: Any) -> tuple[dict[Any, float], list[str]]:
    errors = []
    if isinstance(question, Noul) and isinstance(answer, NoulAnswer):
        probabilities = {False: 1 - answer.noul, True: answer.noul}
    elif isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
        probabilities = answer.probabilities
        if set(probabilities) != set(question.criteria):
            errors.append("Choice did not return exactly all candidate labels")
        if answer.choice not in probabilities:
            errors.append("Choice selected an unknown label")
        elif probabilities[answer.choice] != max(probabilities.values()):
            errors.append("Choice selected a label below the maximum probability")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            errors.append("Choice confidence is nonfinite or outside [0, 1]")
    elif isinstance(question, Score) and isinstance(answer, ScoreAnswer):
        probabilities = answer.probabilities
        expected_keys = set(range(len(question.criteria)))
        if set(probabilities) != expected_keys:
            errors.append("Score did not return all zero-based rubric levels")
        if answer.legend != dict(enumerate(question.criteria)):
            errors.append("Score legend does not preserve the supplied rubric")
        expected_score = sum(
            level * probability for level, probability in probabilities.items()
        )
        if not math.isfinite(answer.score) or not math.isclose(
            answer.score, expected_score, abs_tol=1e-6
        ):
            errors.append("Score is not the probability-weighted rubric expectation")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            errors.append("Score confidence is nonfinite or outside [0, 1]")
    else:
        return {}, ["Answer type does not match the SDK question type"]
    if not probabilities or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
    ):
        errors.append(
            "Candidate probabilities are missing, nonfinite, or outside [0, 1]"
        )
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
        errors.append("Candidate probabilities do not sum to one")
    if not errors and isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p)
        expected_confidence = (
            1.0 - entropy / math.log(len(probabilities))
            if len(probabilities) > 1
            else 1.0
        )
        if not math.isclose(answer.confidence, expected_confidence, abs_tol=1e-6):
            errors.append("Confidence differs from one minus normalized entropy")
    return probabilities, errors


def valid_canvas(record: dict[str, Any], capacity: int = 256) -> bool:
    """Check fixed metadata and the declared mutable answer/reasoning slots."""
    before, after = record["canvas_tokens"], record["output_tokens"]
    position = record["answer_position"]
    content_tokens = record["question_canvas_tokens"]
    reasoning = record.get("reasoning_positions", [])
    if not isinstance(reasoning, (list, tuple)):
        return False
    if reasoning:
        vocab_size = record.get("reasoning_vocab_size")
        forbidden = record.get("reasoning_forbidden_token_ids")
        if (
            type(vocab_size) is not int
            or vocab_size <= 0
            or not isinstance(forbidden, (list, tuple))
            or any(
                type(token) is not int or not 0 <= token < vocab_size
                for token in forbidden
            )
            or any(
                type(index) is not int or not 0 <= index < content_tokens
                for index in reasoning
            )
            or list(reasoning) != list(range(reasoning[0], reasoning[-1] + 1))
            or position in reasoning
            or any(index >= len(after) for index in reasoning)
        ):
            return False
        forbidden = set(forbidden)
        if any(
            type(after[index]) is not int
            or not 0 <= after[index] < vocab_size
            or after[index] in forbidden
            for index in reasoning
        ):
            return False
    mutable = {position, *reasoning}
    return (
        0 < len(before) <= capacity
        and len(after) == len(before)
        and 0 < content_tokens <= len(before)
        and 0 <= position < content_tokens
        and after[position] in record["candidate_token_ids"]
        and all(
            a == b
            for index, (a, b) in enumerate(zip(before, after))
            if index not in mutable
        )
    )


async def request_case(
    client: AsyncTypeSafeClient, case: Case, model: str
) -> dict[str, Any]:
    artifact = {
        "name": case.name,
        "request": {
            "model": model,
            "state": case.state,
            "questions": msgspec.to_builtins(case.questions),
            "seed": 0,
            "samples": 1,
        },
        "expected": case.expected,
        "errors": [],
        "decisions": {},
    }
    start = time.perf_counter()
    try:
        response = await client.system_one(
            case.state,
            case.questions,
            model=model,
            extra_body={"seed": 0, "samples": 1},
        )
        artifact["response"] = response.raw_http_response.json()
        artifact["model"] = response.model
        artifact["request_id"] = response.request_id
        if not response.request_id:
            artifact["errors"].append("Missing request ID")
        if response.model != model:
            artifact["errors"].append("Response model differs from requested model")
        if set(response.answers) != set(case.questions):
            artifact["errors"].append(
                "Response answer names differ from question names"
            )
        for name, question in case.questions.items():
            if name not in response.answers:
                continue
            probabilities, errors = validate_answer(question, response.answers[name])
            artifact["errors"].extend(f"{name}: {error}" for error in errors)
            if errors:
                continue
            selected = max(probabilities, key=probabilities.get)
            decision = {"probabilities": probabilities, "selected": selected}
            if name in case.expected:
                expected = case.expected[name]
                gold_probability = probabilities[expected]
                decision.update(
                    correct=selected == expected,
                    brier=sum(
                        (p - float(label == expected)) ** 2
                        for label, p in probabilities.items()
                    ),
                    nll=-math.log(max(gold_probability, 1e-15)),
                    zero_gold_probability=gold_probability == 0,
                )
            artifact["decisions"][name] = decision
    except Exception as error:
        artifact["errors"].append(f"{type(error).__name__}: {error}")
    artifact["latency_seconds"] = time.perf_counter() - start
    return artifact


def quality_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [
        decision
        for result in artifacts
        for decision in result["decisions"].values()
        if "correct" in decision
    ]
    return {
        "decisions": len(decisions),
        "accuracy": statistics.mean(d["correct"] for d in decisions)
        if decisions
        else None,
        "mean_multiclass_brier": statistics.mean(d["brier"] for d in decisions)
        if decisions
        else None,
        "mean_nll": statistics.mean(d["nll"] for d in decisions) if decisions else None,
        "zero_gold_probability_count": sum(
            d["zero_gold_probability"] for d in decisions
        ),
        "protocol_errors": sum(len(result["errors"]) for result in artifacts),
    }


async def independence_check(
    client: AsyncTypeSafeClient, model: str, tolerance: float
) -> dict[str, Any]:
    state = {"fruit": "apple", "color": "red"}
    target = Noul(instructions="Is the fruit an apple?")
    other = Choice(
        instructions="What is the color?", criteria={"red": None, "blue": None}
    )
    cases = [
        Case("alone", state, {"target": target}, {"target": True}),
        Case(
            "added_question",
            state,
            {"target": target, "unrelated": other},
            {"target": True, "unrelated": "red"},
        ),
        Case(
            "reordered_keys",
            dict(reversed(list(state.items()))),
            {"unrelated": other, "target": target},
            {"target": True, "unrelated": "red"},
        ),
    ]
    artifacts = [await request_case(client, case, model) for case in cases]
    differences = []
    if all(
        not result["errors"] and "target" in result["decisions"] for result in artifacts
    ):
        reference = artifacts[0]["decisions"]["target"]["probabilities"]
        differences = [
            max(
                abs(result["decisions"]["target"]["probabilities"][key] - probability)
                for key, probability in reference.items()
            )
            for result in artifacts[1:]
        ]
    return {
        "tolerance": tolerance,
        "maximum_probability_differences": differences,
        "passed": len(differences) == 2 and max(differences) <= tolerance,
        "requests": artifacts,
    }


async def concurrency_sweep(
    client: AsyncTypeSafeClient, case: Case, model: str
) -> list[dict[str, Any]]:
    results = []
    for concurrency in (1, 4, 16):
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(limit: asyncio.Semaphore = semaphore) -> dict[str, Any]:
            async with limit:
                return await request_case(client, case, model)

        start = time.perf_counter()
        artifacts = await asyncio.gather(
            *(run_one() for _ in range(max(8, concurrency)))
        )
        elapsed = time.perf_counter() - start
        latencies = [artifact["latency_seconds"] for artifact in artifacts]
        succeeded = sum(not artifact["errors"] for artifact in artifacts)
        decisions = sum(len(artifact["decisions"]) for artifact in artifacts)
        results.append(
            {
                "concurrency": concurrency,
                "elapsed_seconds": elapsed,
                "requests_per_second": succeeded / elapsed,
                "decisions_per_second": decisions / elapsed,
                "p50_seconds": statistics.median(latencies),
                "p95_seconds": statistics.quantiles(
                    latencies, n=100, method="inclusive"
                )[94],
                **quality_summary(artifacts),
                "requests": artifacts,
            }
        )
    return results


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "scoring_note": (
            "Descriptive metrics on bounded synthetic factual/state questions; "
            "not a calibration study. Brier uses the full categorical vector; "
            "NLL clips gold probabilities below 1e-15. Adapter confidence is "
            "one minus normalized entropy."
        ),
        "passed": False,
    }
    async with AsyncTypeSafeClient(
        api_key="local-evaluation",
        base_url=args.base_url,
        model=args.model,
        retry=RetryPolicy(max_retries=0),
        timeout=args.timeout,
    ) as client:
        try:
            models = await client.models.list()
            report["models"] = models.raw_http_response.json()
            report["models_request_id"] = models.request_id
            if args.model not in {model.name for model in models.models}:
                raise ValueError(
                    f"Configured model {args.model!r} missing from models.list()"
                )
        except Exception as error:
            report["error"] = f"models.list(): {type(error).__name__}: {error}"
            return report
        cases = labeled_cases()
        report["quality_requests"] = [
            await request_case(client, case, args.model) for case in cases
        ]
        report["quality"] = quality_summary(report["quality_requests"])
        report["independence"] = await independence_check(
            client, args.model, args.independence_tolerance
        )
        report["concurrency"] = await concurrency_sweep(client, cases[0], args.model)
        accuracy = report["quality"]["accuracy"]
        report["minimum_accuracy"] = args.minimum_accuracy
        report["passed"] = (
            report["quality"]["protocol_errors"] == 0
            and report["quality"]["decisions"]
            == sum(len(case.expected) for case in cases)
            and accuracy is not None
            and accuracy >= args.minimum_accuracy
            and report["independence"]["passed"]
            and all(result["protocol_errors"] == 0 for result in report["concurrency"])
        )
    return report


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="diffusion-gemma")
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--minimum-accuracy", type=float, default=0.8)
    parser.add_argument("--independence-tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "quality": report.get("quality"),
                "report": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
