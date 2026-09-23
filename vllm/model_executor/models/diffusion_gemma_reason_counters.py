# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in completed-pass counters; no tokens, logits, or response changes."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReasoningCounterTrace:
    def __init__(self, directory: str, run_id: str, source: str, rank: int):
        if not run_id:
            raise ValueError("Reasoning counter tracing requires a run ID")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = {
            "run_id": run_id,
            "tp_rank": rank,
            "pid": os.getpid(),
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(source).resolve(), Path(__file__).resolve())
            },
        }
        self.pending: dict[int, dict[str, Any]] = {}
        self.sequence = 0

    def start(self, slot, request_id, reasoning_cap, answer_passes, enabled):
        self.abort(slot, "slot_reused")
        self.pending[slot] = {
            **self.binding,
            "sequence": self.sequence,
            "request_id": request_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "requested_reasoning_passes": reasoning_cap,
            "requested_answer_passes": answer_passes,
            "actual_reasoning_passes": 0,
            "actual_answer_passes": 0,
            "convergence_stop": enabled,
            "early_stop": False,
            "terminal_reason": None,
            "terminal_stable": None,
            "terminal_mean_entropy": None,
            "terminal_temperature": None,
        }
        self.sequence += 1

    def reasoning(self, slot, actual, terminal, reason, temperature, statistics):
        record = self.pending[slot]
        assert actual == record["actual_reasoning_passes"] + 1
        record["actual_reasoning_passes"] = actual
        if terminal:
            record["terminal_reason"] = reason
            record["early_stop"] = reason == "stable_confident"
            record["terminal_temperature"] = temperature
            if statistics is not None:
                stable, entropy = statistics
                record["terminal_stable"] = bool(stable.item())
                value = float(entropy.item())
                record["terminal_mean_entropy"] = (
                    value if math.isfinite(value) else None
                )

    def answer(self, slot):
        record = self.pending.get(slot)
        if record is not None:
            assert record["terminal_reason"] is not None
            record["actual_answer_passes"] += 1

    def finish(self, slot):
        record = self.pending.get(slot)
        if record is None:
            return
        assert record["terminal_reason"] is not None
        assert record["actual_answer_passes"] == record["requested_answer_passes"]
        self._write(slot, "completed")

    def abort(self, slot, reason="removed_before_emission"):
        if slot in self.pending:
            self._write(slot, "aborted", reason)

    def _write(self, slot, status, abort_reason=None):
        record = self.pending.pop(slot)
        record.update(
            status=status,
            abort_reason=abort_reason,
            ended_utc=datetime.now(timezone.utc).isoformat(),
            total_child_forwards=(
                record["actual_reasoning_passes"] + record["actual_answer_passes"]
            ),
            count_basis=(
                "Completed sampler passes; failed or in-flight forwards excluded"
            ),
        )
        suffix = hashlib.sha256(record["request_id"].encode()).hexdigest()[:16]
        filename = (
            f"counter-{record['pid']}-{record['tp_rank']}-"
            f"{record['sequence']:06d}-{suffix}.json"
        )
        with (self.directory / filename).open("x") as stream:
            json.dump(record, stream, allow_nan=False)
            stream.write("\n")
