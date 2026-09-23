# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise image-conditioned SDK decisions with reproducible geometric fixtures.

Use --prepare-only to draw the PNGs without making any inference requests.
The live run makes five SDK calls, totaling 22 single-pass model reads.
"""

import argparse
import hashlib
import json
import math
import secrets
import time
from pathlib import Path
from typing import Any

import msgspec
import pybase64 as base64
import regex as re
from PIL import Image, ImageDraw
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

if __package__:
    from .evaluate import valid_canvas, validate_answer
else:
    from evaluate import valid_canvas, validate_answer

STATE = {"task": "Answer from the attached image or images using visible evidence."}
SAMPLES = 2


def prepare_fixtures(output_dir: Path) -> list[dict[str, Any]]:
    """Draw deterministic pixels; filenames and ground truth are never prompted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for index in range(2):
        image = Image.new("RGB", (512, 384), "white")
        draw = ImageDraw.Draw(image)
        red, blue, green = (220, 30, 50), (30, 80, 220), (20, 150, 60)
        draw.ellipse((64, 64, 208, 208), fill=red if index == 0 else blue)
        draw.rectangle((304, 64, 448, 208), fill=blue if index == 0 else red)
        for center in [256] if index == 0 else [96, 256, 416]:
            draw.polygon(
                [(center, 270), (center - 46, 350), (center + 46, 350)], fill=green
            )
        path = output_dir / f"frame-{index:02d}.png"
        image.save(path, format="PNG")
        payload = path.read_bytes()
        fixtures.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mime_type": "image/png",
                "width": 512,
                "height": 384,
                "byte_size": len(payload),
                "expected": {
                    "circle_color": "red" if index == 0 else "blue",
                    "red_left_of_blue": index == 0,
                    "green_triangles": 1 if index == 0 else 3,
                },
            }
        )
    (output_dir / "fixtures.json").write_text(json.dumps(fixtures, indent=2))
    return fixtures


def questions() -> dict[str, Any]:
    return {
        "circle_color": Choice(
            instructions="What color is the circle?",
            criteria={"red": None, "blue": None},
        ),
        "red_left_of_blue": Noul(
            instructions="Is the red shape to the left of the blue shape?"
        ),
        "green_triangles": Score(
            instructions="How many green triangles are visible?",
            criteria=[
                "zero green triangles",
                "one green triangle",
                "two green triangles",
                "three green triangles",
            ],
        ),
    }


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+",
            "<image payload omitted>",
            value,
        )
    return value


def trace_records(path: Path, request_id: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if request_id not in line:
                continue
            record = json.loads(line)
            if record.get("request_id") == request_id:
                records.append(record)
    return records


def check_trace(
    records: list[dict[str, Any]],
    expected_questions: dict[str, Any],
    images: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors = []
    samples = [record for record in records if record.get("event") == "sample"]
    expected = {
        (name, sample) for name in expected_questions for sample in range(SAMPLES)
    }
    if (
        len(samples) != len(expected)
        or {(record["question_id"], record["sample"]) for record in samples} != expected
    ):
        errors.append("Trace did not contain exactly all requested question draws")
    if len([record for record in records if record.get("event") == "response"]) != 1:
        errors.append("Trace did not contain exactly one typed response")
    if len({record["shared_prompt_hash"] for record in samples}) != 1:
        errors.append("Questions did not share one exact multimodal prompt hash")
    for sample in samples:
        if not valid_canvas(sample):
            errors.append(
                "The decision canvas was truncated or violated its constraints"
            )
        if sample.get("question_placement") != "canvas":
            errors.append("Question metadata was not placed on the canvas")
        if sample.get("image_count") != len(images) or [
            image.get("sha256") for image in sample.get("shared_images", [])
        ] != [image["sha256"] for image in images]:
            errors.append("Recorded image identity/order differs from supplied pixels")
        if sample.get("constraint_backend") != "xgrammar":
            errors.append("Trace did not attribute constraints to XGrammar")
        if sample.get("reasoning") is not False or sample.get("denoise_steps") != 1:
            errors.append("Trace configuration is not one-pass, reasoning-off")
        if not all(math.isfinite(value) for value in sample["candidate_logprobs"]):
            errors.append("A candidate logprob is not finite")
    return samples, errors


def run_case(
    client: TypeSafeClient,
    name: str,
    image_fixtures: list[dict[str, Any]],
    question_set: dict[str, Any],
    expected: dict[str, Any],
    cache_salt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = [
        {key: value for key, value in fixture.items() if key != "expected"}
        for fixture in image_fixtures
    ]
    result: dict[str, Any] = {
        "name": name,
        "request": {
            "model": args.model,
            "state": STATE,
            "questions": msgspec.to_builtins(question_set),
            "seed": 0,
            "samples": SAMPLES,
            "cache_salt": cache_salt,
            "images": metadata,
            "image_payloads_omitted": True,
        },
        "expected": expected,
        "decisions": {},
        "errors": [],
    }
    started = time.perf_counter()
    try:
        response = client.system_one(
            STATE,
            question_set,
            model=args.model,
            extra_body={
                "seed": 0,
                "samples": SAMPLES,
                "cache_salt": cache_salt,
                "images": [
                    {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(Path(image["path"]).read_bytes()).decode()
                    }
                    for image in image_fixtures
                ],
            },
        )
        result["request_id"] = response.request_id
        result["response"] = response.raw_http_response.json()
        if response.model != args.model or set(response.answers) != set(question_set):
            result["errors"].append("Model or answer names do not match the request")
        for question_id, question in question_set.items():
            probabilities, errors = validate_answer(
                question, response.answers[question_id]
            )
            result["errors"].extend(f"{question_id}: {error}" for error in errors)
            if errors:
                continue
            selected = max(probabilities, key=probabilities.__getitem__)
            result["decisions"][question_id] = {
                "probabilities": probabilities,
                "selected": selected,
                "correct": selected == expected[question_id],
                "gold_probability": probabilities[expected[question_id]],
            }
        records = trace_records(args.trace, response.request_id)
        samples, errors = check_trace(records, question_set, metadata)
        result["errors"].extend(errors)
        for question_id in question_set:
            draws = [r for r in samples if r["question_id"] == question_id]
            if not draws:
                continue
            labels = draws[0]["candidate_labels"]
            mean = [
                sum(draw["probabilities"][index] for draw in draws) / len(draws)
                for index in range(len(labels))
            ]
            answer = result["response"]["answers"][question_id]
            actual = (
                [answer["noul"], 1 - answer["noul"]]
                if answer["type"] == "noul"
                else [answer["probabilities"][label] for label in labels]
            )
            if any(abs(a - b) > 1e-12 for a, b in zip(mean, actual)):
                result["errors"].append(
                    f"{question_id}: typed probabilities differ from trace draw mean"
                )
        result["sample_records"] = samples
        result["cache"] = {
            "shared_prompt_hashes": sorted({r["shared_prompt_hash"] for r in samples}),
            "cached_tokens": [r.get("cached_prompt_tokens") for r in samples],
            "created_tokens": [r.get("created_cache_tokens") for r in samples],
            "prompt_tokens": [r.get("shared_prompt_tokens") for r in samples],
        }
        with (args.output_dir / "image-decisions.jsonl").open("a") as output:
            for record in records:
                output.write(json.dumps(redact(record), ensure_ascii=False) + "\n")
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {redact(str(error))}")
    result["latency_seconds"] = time.perf_counter() - started
    return result


def compare_cases(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["name"]: result for result in results}
    if any(result["errors"] or "cache" not in result for result in results):
        return {"errors": ["Missing trace evidence for one or more image requests"]}
    cold, warm, changed = [by_name[name] for name in ("cold", "warm", "changed_pixels")]
    prompt_hash = lambda result: result["cache"]["shared_prompt_hashes"]
    checks = {
        "same_image_same_prompt": prompt_hash(cold) == prompt_hash(warm),
        "changed_pixels_changed_prompt": prompt_hash(cold) != prompt_hash(changed),
        "image_order_changed_prompt": prompt_hash(by_name["ordered_pair"])
        != prompt_hash(by_name["reversed_pair"]),
        "paired_state_and_questions_identical": cold["request"]["state"]
        == changed["request"]["state"]
        and cold["request"]["questions"] == changed["request"]["questions"],
    }
    warm_draws = {
        (row["question_id"], row["sample"]): row for row in warm["sample_records"]
    }
    comparisons = []
    for row in cold["sample_records"]:
        other = warm_draws[row["question_id"], row["sample"]]
        comparisons.append(
            {
                "question_id": row["question_id"],
                "sample": row["sample"],
                "same_input_canvas": row["canvas_tokens"] == other["canvas_tokens"],
                "maximum_probability_difference": max(
                    abs(a - b)
                    for a, b in zip(row["probabilities"], other["probabilities"])
                ),
            }
        )
    checks["same_image_same_seeded_canvases"] = all(
        row["same_input_canvas"] for row in comparisons
    )
    counters_reported = all(
        isinstance(count, int)
        for result in (cold, warm, changed)
        for count in result["cache"]["created_tokens"]
    )
    if counters_reported:
        checks["warm_image_no_new_prefix_blocks"] = (
            sum(warm["cache"]["created_tokens"]) == 0
        )
        checks["changed_image_created_new_prefix_blocks"] = (
            sum(changed["cache"]["created_tokens"]) > 0
        )
    return {
        "checks": checks,
        "cache_counters_reported": counters_reported,
        "cold_warm_draw_comparisons": comparisons,
        "paired_answer_flips": [
            {
                "question_id": name,
                "before": cold["decisions"].get(name, {}).get("selected"),
                "after": changed["decisions"].get(name, {}).get("selected"),
                "both_correct": cold["decisions"].get(name, {}).get("correct", False)
                and changed["decisions"].get(name, {}).get("correct", False),
            }
            for name in cold["expected"]
        ],
        "errors": [name for name, passed in checks.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="diffusion-gemma")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    report_path = args.output_dir / "image-showcase.json"
    if not args.prepare_only and (
        report_path.exists() or (args.output_dir / "image-decisions.jsonl").exists()
    ):
        parser.error("Use a new output directory to preserve earlier inference results")
    fixtures = prepare_fixtures(args.output_dir)
    if args.prepare_only:
        print(json.dumps({"fixtures": [image["path"] for image in fixtures]}))
        return
    if args.trace is None or (args.trace.exists() and not args.trace.is_file()):
        parser.error("--trace must name the image adapter's JSONL trace")
    first_question = {
        "first_circle_color": Choice(
            instructions="What color is the circle in the FIRST attached image?",
            criteria={"red": None, "blue": None},
        )
    }
    cases = [
        ("cold", [fixtures[0]], questions(), fixtures[0]["expected"]),
        ("warm", [fixtures[0]], questions(), fixtures[0]["expected"]),
        ("changed_pixels", [fixtures[1]], questions(), fixtures[1]["expected"]),
        ("ordered_pair", fixtures, first_question, {"first_circle_color": "red"}),
        (
            "reversed_pair",
            fixtures[::-1],
            first_question,
            {"first_circle_color": "blue"},
        ),
    ]
    salt = secrets.token_urlsafe(24)
    with TypeSafeClient(
        base_url=args.base_url,
        api_key="local-image-showcase",
        model=args.model,
        timeout=args.timeout,
        retry=RetryPolicy(max_retries=0),
    ) as client:
        results = [run_case(client, *case, salt, args) for case in cases]
    unique = [result for result in results if result["name"] != "warm"]
    decisions = [
        decision for result in unique for decision in result["decisions"].values()
    ]
    comparisons = compare_cases(results)
    summary = {
        "scored_unique_decisions": len(decisions),
        "correct": sum(decision["correct"] for decision in decisions),
        "accuracy": sum(decision["correct"] for decision in decisions) / len(decisions)
        if decisions
        else None,
        "protocol_errors": sum(len(result["errors"]) for result in results)
        + len(comparisons["errors"]),
    }
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "fixtures": fixtures,
        "summary": summary,
        "results": results,
        "comparisons": comparisons,
        "limitations": [
            "Small deterministic visual smoke corpus, not a calibration study.",
            (
                "Same API seed does not fix the physical noise token "
                "when image identity changes."
            ),
            "Cache counters are observations, not proof from prompt hashes alone.",
        ],
        "passed": summary["protocol_errors"] == 0
        and summary["scored_unique_decisions"] == 8
        and summary["correct"] == 8,
    }
    report_path.write_text(json.dumps(redact(report), indent=2, ensure_ascii=False))
    print(
        json.dumps({"passed": report["passed"], **summary, "report": str(report_path)})
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
