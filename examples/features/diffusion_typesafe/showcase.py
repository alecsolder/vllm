# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run richer SDK examples and extract their real traces for the canvas TUI."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

if __package__:
    from .evaluate import valid_canvas, validate_answer
else:
    from evaluate import valid_canvas, validate_answer


@dataclass
class Scenario:
    name: str
    state: dict[str, Any]
    questions: dict[str, Any]
    samples: int = 1


def scenarios() -> list[Scenario]:
    incident = Scenario(
        "incident-triage",
        {
            "incident": "EU checkout errors immediately after a release",
            "timeline": [
                {"at": "09:12", "event": "checkout release deployed"},
                {"at": "09:14", "event": "EU checkout HTTP 502 rate rose to 37%"},
                {"at": "09:19", "event": "error rate still 37%; US unaffected"},
            ],
            "checks": {
                "payment_provider": "healthy",
                "database": "healthy; no missing records found",
                "rollback": "ready and reversible",
                "workaround": "none",
            },
            "policy": {
                "rollback": "Roll back a correlated release if errors exceed 20%.",
                "public_update": "Post if a regional outage lasts over 5 minutes.",
            },
        },
        {
            "owner": Choice(
                instructions="Who should own the initial investigation?",
                criteria={
                    "checkout": "Cart and checkout application",
                    "payments": "External payment authorization and capture",
                    "database": "Database availability and records",
                    "network": "Cross-service transport failures",
                    "identity": "Login and credentials",
                    "billing": "Invoices and subscriptions",
                    "security": "Malicious activity",
                    "support": "Requests without a technical incident",
                },
            ),
            "severity": Score(
                instructions="Select the highest severity supported by the evidence.",
                criteria=[
                    {"level": "none", "condition": "No customer impact"},
                    {"level": "minor", "condition": "Cosmetic problem only"},
                    {"level": "degraded", "condition": "Errors below 10%"},
                    {"level": "major", "condition": "Regional errors above 10%"},
                    {"level": "critical", "condition": "Global outage or data loss"},
                ],
            ),
            "rollback_allowed": Noul(
                instructions="Does the observed incident satisfy the rollback policy?"
            ),
            "confirmed_data_loss": Noul(
                instructions="Is there affirmative evidence that records were lost?"
            ),
            "next_action": Choice(
                instructions="Choose the immediate action best supported by policy.",
                criteria={
                    "rollback_release": "Undo the correlated checkout deployment",
                    "restart_database": "Restart the healthy database",
                    "wait_for_provider": "Wait for the payment provider to recover",
                    "close_incident": "Declare the issue resolved",
                },
            ),
            "public_update_due": Noul(
                instructions="At 09:20, is a public update due under the policy?"
            ),
        },
    )
    catalog = (
        "accounts",
        "authentication",
        "authorization",
        "billing",
        "payments",
        "checkout",
        "catalog",
        "inventory",
        "shipping",
        "returns",
        "search",
        "recommendations",
        "notifications",
        "messaging",
        "media",
        "storage",
        "database",
        "networking",
        "compute",
        "analytics",
        "integrations",
        "localization",
        "privacy",
        "security",
        "documentation",
        "accessibility",
    )
    routing = Scenario(
        "routing-26",
        {
            "report": "Checkout is unusable with a screen reader and keyboard.",
            "steps": [
                "Tab to the address form: focus disappears.",
                "Every control is announced as 'button' without its name.",
                "The same checkout works with a mouse and no screen reader.",
            ],
            "environment": {"locale": "fr-FR", "translation": "correct"},
            "observations": {"payment_errors": 0, "network_errors": 0},
        },
        {
            "specialist_team": Choice(
                instructions=(
                    "Route to the specialist for the failure mechanism, not merely "
                    "the page where the problem appears. Choose among all 26 teams."
                ),
                criteria=dict.fromkeys(catalog),
            ),
            "translation_problem": Noul(
                instructions="Does the report identify incorrect translations?"
            ),
            "assistive_impact": Score(
                instructions="Rate impact on the reported assistive workflow.",
                criteria=[
                    "No effect on task completion",
                    "Minor inconvenience; all controls usable",
                    "Some controls difficult but a workaround is documented",
                    "Cannot complete the reported workflow; no workaround given",
                ],
            ),
        },
    )
    noise = Scenario(
        "mixed-ticket-8-draws",
        {
            "customer": (
                "Since yesterday's plan upgrade, opening our workspace sometimes "
                "returns access denied. Two teammates reported it; we have not "
                "checked how many people are affected."
            ),
            "observations": {
                "billing_dashboard": "payment pending",
                "bank": "one posted charge",
                "incognito": "worked once, then failed",
                "other_workspace": "not tested",
            },
            "missing": ["request IDs", "entitlement audit", "regional error rates"],
        },
        {
            "initial_owner": Choice(
                instructions=(
                    "Choose the most useful FIRST investigation owner, not a "
                    "proven root cause. Use only the available evidence."
                ),
                criteria={
                    "billing_entitlements": "Payment-to-plan sync and permissions",
                    "identity_sessions": "Stale login, session, or account access",
                    "platform_runtime": "Intermittent app or service failures",
                },
            ),
            "unresolved_causes": Noul(
                instructions=(
                    "Do multiple plausible causes remain unresolved by the evidence?"
                )
            ),
            "established_impact": Score(
                instructions=(
                    "Rate impact established by observations. Missing scope checks "
                    "do not imply a global outage."
                ),
                criteria=[
                    "No demonstrated disruption",
                    "Isolated user with a usable workaround",
                    "Small group blocked or workaround unreliable",
                    "Most users in one workspace blocked",
                    "Multiple independent workspaces broadly blocked",
                ],
            ),
        },
        samples=8,
    )
    return [incident, routing, noise]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="diffusion-gemma")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace.touch(exist_ok=True)
    reports = []
    with (
        args.trace.open(encoding="utf-8") as source,
        TypeSafeClient(
            base_url=args.base_url, api_key="local", model=args.model, timeout=120
        ) as client,
    ):
        source.seek(0, 2)
        for scenario in scenarios():
            response = client.system_one(
                state=scenario.state,
                questions=scenario.questions,
                extra_body={"samples": scenario.samples, "seed": 42},
            )
            if (
                response.raw_http_response.headers.get("x-typesafe-constraint-backend")
                != "xgrammar"
            ):
                raise ValueError("The showcase requires the XGrammar adapter")
            if set(response.answers) != set(scenario.questions):
                raise ValueError("The response did not preserve question names")
            for name, question in scenario.questions.items():
                _, errors = validate_answer(question, response.answers[name])
                if errors:
                    raise ValueError(f"{scenario.name}/{name}: {errors}")
            records = [json.loads(line) for line in source if line.strip()]
            records = [r for r in records if r.get("request_id") == response.request_id]
            samples = [r for r in records if r.get("event") == "sample"]
            expected = {
                (name, draw)
                for name in scenario.questions
                for draw in range(scenario.samples)
            }
            if (
                len(samples) != len(expected)
                or {(r["question_id"], r["sample"]) for r in samples} != expected
                or len([r for r in records if r.get("event") == "response"]) != 1
            ):
                raise ValueError("Incomplete trace for the SDK response")
            for record in samples:
                if (
                    not valid_canvas(record)
                    or record["constraint_backend"] != "xgrammar"
                    or record["reasoning"] is not False
                    or record["denoise_steps"] != 1
                    or not all(math.isfinite(p) for p in record["candidate_logprobs"])
                ):
                    raise ValueError("Invalid single-slot XGrammar canvas trace")
            order = {name: i for i, name in enumerate(scenario.questions)}
            samples.sort(key=lambda r: (order[r["question_id"]], r["sample"]))
            trace_path = args.output_dir / f"{scenario.name}.jsonl"
            with trace_path.open("x", encoding="utf-8") as output:
                for record in samples + [
                    r for r in records if r["event"] == "response"
                ]:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            report = {
                "scenario": scenario.name,
                "request_id": response.request_id,
                "state": scenario.state,
                "questions": msgspec.to_builtins(scenario.questions),
                "draws_per_question": scenario.samples,
                "samples_verified": len(samples),
                "response": response.raw_http_response.json(),
                "trace": str(trace_path),
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    with (args.output_dir / "results.json").open("x", encoding="utf-8") as output:
        json.dump(reports, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
