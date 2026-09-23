# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay recorded Diffusion Gemma decisions without running inference.

Usage:
    .venv/bin/python examples/features/diffusion_typesafe/tui.py trace.jsonl --follow
    .venv/bin/python examples/features/diffusion_typesafe/tui.py trace.jsonl --snapshot
"""

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.segment import Segment
from rich.table import Table
from rich.text import Text


class Trace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.samples: list[dict[str, Any]] = []
        self.parents: dict[str, dict[str, Any]] = {}
        self.parents_by_id: dict[str, dict[str, Any]] = {}
        self.responses: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.offset = 0
        self.line = 0
        self.inode: int | None = None

    def read(self, *, final: bool = False) -> None:
        try:
            stat = self.path.stat()
            if stat.st_ino != self.inode or stat.st_size < self.offset:
                self.samples.clear()
                self.parents.clear()
                self.parents_by_id.clear()
                self.responses.clear()
                self.errors.clear()
                self.offset = self.line = 0
                self.inode = stat.st_ino
            with self.path.open("rb") as source:
                source.seek(self.offset)
                while raw := source.readline():
                    if not final and not raw.endswith(b"\n"):
                        break
                    self.offset = source.tell()
                    self.line += 1
                    if not raw.strip():
                        continue
                    try:
                        record = json.loads(raw)
                        if not isinstance(record, dict):
                            raise ValueError("expected a JSON object")
                        event = record.get("event")
                        if event == "sample":
                            self.samples.append(record)
                        elif event == "parent":
                            self.parents[str(record.get("request_id", ""))] = record
                            parent_id = record.get("parent_id")
                            if (
                                record.get("status") == "completed"
                                and isinstance(parent_id, str)
                                and parent_id
                            ):
                                self.parents_by_id[parent_id] = record
                        elif event == "response":
                            self.responses[str(record.get("request_id", ""))] = record
                        elif event == "error":
                            self.errors.append(str(record.get("error", record)))
                    except (ValueError, UnicodeDecodeError) as exc:
                        self.errors.append(f"Line {self.line}: {exc}")
                    self.errors = self.errors[-3:]
        except FileNotFoundError:
            return
        except OSError as exc:
            message = f"Cannot read trace: {exc}"
            if not self.errors or self.errors[-1] != message:
                self.errors.append(message)
                self.errors = self.errors[-3:]


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def metric(value: Any, digits: int = 4) -> str:
    parsed = number(value)
    return f"{parsed:.{digits}f}" if parsed is not None else "not recorded"


def items(record: dict[str, Any], field: str) -> list[Any]:
    value = record.get(field)
    return value if isinstance(value, list) else []


def visible(value: Any) -> str:
    return str(value).replace("\n", "↵").replace("\t", "⇥").replace("\r", "")


def answer_slot(record: dict[str, Any]) -> int | None:
    position = record.get("answer_position")
    return position if isinstance(position, int) and position >= 0 else None


def selected_token(record: dict[str, Any]) -> int | None:
    output = items(record, "output_tokens")
    position = answer_slot(record)
    if position is not None and position < len(output):
        return output[position]
    if len(output) == 1:
        return output[0]
    return None


def canvas_grid_shape(count: int) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    columns = 8 if count == 32 else math.ceil(math.sqrt(count))
    return columns, math.ceil(count / columns)


def _canvas_token_text(
    record: dict[str, Any], index: int, token_id: Any, *, output: bool
) -> str:
    if output:
        decoded = items(record, "output_text")
        if index < len(decoded):
            return str(decoded[index])
        original = items(record, "canvas_tokens")
        unchanged = index < len(original) and token_id == original[index]
        if not unchanged:
            ids = items(record, "candidate_token_ids")
            codes = items(record, "candidate_codes")
            if token_id in ids and ids.index(token_id) < len(codes):
                return str(codes[ids.index(token_id)])
            return f"#{token_id}"
    decoded = items(record, "canvas_text")
    return str(decoded[index]) if index < len(decoded) else f"#{token_id}"


def _canvas_preview(text: str, width: int) -> str:
    aliases = {"<pad>": "PAD", "<turn|>": "END", "<eos>": "EOS", " ": "SP"}
    escaped = aliases.get(text, ascii(text)[1:-1]) or "''"
    return escaped if len(escaped) <= width else escaped[: width - 1] + "…"


def canvas_panel(
    record: dict[str, Any],
    *,
    output: bool = False,
    selected_position: int | None = None,
    compact: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> Panel:
    tokens = items(record, "output_tokens" if output else "canvas_tokens")
    on_canvas = record.get("question_placement") == "canvas"
    title = (
        ("Output JSON canvas" if output else "Seeded JSON canvas")
        if on_canvas
        else ("Output canvas" if output else "Seeded input canvas")
    )
    if not tokens:
        return Panel(
            Text(f"{title} was not recorded.", style="dim"),
            title=title,
            border_style="green" if output else "cyan",
        )
    original = items(record, "canvas_tokens")
    position = answer_slot(record)
    inspected = selected_position if selected_position is not None else position
    if inspected is None or not 0 <= inspected < len(tokens):
        inspected = 0
    columns, rows = canvas_grid_shape(len(tokens))
    cell_width = (
        max(1, (width - 13) // columns - 1)
        if width is not None
        else 3
        if compact
        else 4
    )
    overhead = 8 if on_canvas else 7
    cell_height = (
        max(1, min(3, (height - overhead) // rows)) if height is not None else 1
    )
    header = Text("offset   ", style="dim")
    for column in range(columns):
        header.append(f"{column:02d}".center(cell_width) + " ")
    grid: list[Text] = [header]
    changed = []
    candidate_ids = items(record, "candidate_token_ids")
    for start in range(0, len(tokens), columns):
        end = min(start + columns, len(tokens))
        row = Text(f"{start:03d}–{end - 1:03d}  ", style="dim")
        row_padding = Text(" " * 9)
        for index in range(start, end):
            token_id = tokens[index]
            text = _canvas_token_text(record, index, token_id, output=output)
            mutation = (
                output
                and index != position
                and index < len(original)
                and token_id != original[index]
            )
            if mutation:
                style = "bold white on red"
                changed.append(index)
            elif index == position:
                style = (
                    "bold black on green"
                    if output and (not candidate_ids or token_id in candidate_ids)
                    else "bold white on red"
                    if output
                    else "bold black on yellow"
                )
            elif text == "<pad>" or token_id == record.get("pad_token_id", 0):
                style = "grey50 on grey11"
            else:
                style = "bright_cyan on grey19"
            if index == inspected:
                style += " bold underline"
            preview = _canvas_preview(text, cell_width)
            row.append(preview.center(cell_width), style=style)
            row.append(" ", style="default")
            row_padding.append(" " * cell_width, style=style)
            row_padding.append(" ", style="default")
        grid.append(row)
        grid.extend(row_padding for _ in range(cell_height - 1))
    legend = Text("Fixed  ", style="bright_cyan on grey19")
    legend.append("PAD (fixed)  ", style="grey50 on grey11")
    legend.append(" Seed ", style="black on yellow")
    legend.append(" ")
    legend.append(" Answer ", style="black on green")
    legend.append(" ")
    legend.append(" Invalid/change ", style="white on red")
    legend.append("  inspected", style="underline")
    text = _canvas_token_text(record, inspected, tokens[inspected], output=output)
    details = Text(
        f"Position {inspected:03d} · token ID {tokens[inspected]}"
        f" · text/code {ascii(text)}"
    )
    caption = Text(
        f"{len(tokens) - 1 if position is not None else len(tokens)} fixed positions"
        f" · {1 if position is not None else 0} answer slot\n"
        "Row-major token sequence; grid is not a spatial attention map.",
        style="dim",
    )
    contents: list[Any] = [*grid, legend, details, caption]
    if on_canvas:
        answer_count = int(position is not None and position < len(tokens))
        padding = sum(
            token_id == record.get("pad_token_id", 0) and index != position
            for index, token_id in enumerate(tokens)
        )
        contents.insert(
            -1,
            Text(
                f"Fixed JSON/scaffold: {len(tokens) - padding - answer_count} tokens"
                f" · fixed PAD: {padding} · answer slots: {answer_count}",
                style="cyan",
            ),
        )
    if changed:
        contents.append(
            Text(
                "INVALID: fixed positions changed: "
                + ", ".join(f"{index:03d}" for index in changed),
                style="bold red",
            )
        )
    return Panel(
        Group(*contents),
        title=f"{title} · {len(tokens)} tokens · {columns}×{rows}",
        border_style="green" if output else "cyan",
    )


def candidates_panel(
    record: dict[str, Any],
    limit: int,
    *,
    compact: bool = False,
    show_logprobs: bool = True,
) -> Panel:
    labels = items(record, "candidate_labels")
    codes = items(record, "candidate_codes")
    ids = items(record, "candidate_token_ids")
    logprobs = items(record, "candidate_logprobs")
    probabilities = items(record, "probabilities")
    count = max(map(len, (labels, codes, ids, logprobs, probabilities)))
    selected = selected_token(record)
    order = sorted(
        range(count),
        key=lambda i: (number(probabilities[i]) or 0) if i < len(probabilities) else 0,
        reverse=True,
    )
    shown = order if limit <= 0 else order[:limit]
    table = Table(box=box.SIMPLE, expand=True, padding=(0, 1))
    columns = (
        [
            ("Label", "left"),
            *([("log P", "right")] if show_logprobs else []),
            ("P", "right"),
        ]
        if compact
        else [
            ("Label", "left"),
            ("Code / token ID", "left"),
            ("Logprob", "right"),
            ("Probability", "right"),
            ("", "left"),
        ]
    )
    for name, justify in columns:
        table.add_column(name, justify=justify)
    for index in shown:
        label = str(labels[index]) if index < len(labels) else f"candidate {index}"
        code = visible(codes[index]) if index < len(codes) else "?"
        token_id = ids[index] if index < len(ids) else "?"
        logprob = logprobs[index] if index < len(logprobs) else None
        probability = probabilities[index] if index < len(probabilities) else None
        p = number(probability)
        width = round(max(0.0, min(1.0, p or 0.0)) * 24)
        style = "bold green" if token_id == selected else "white"
        if compact:
            table.add_row(
                Text(label, style=style, overflow="ellipsis", no_wrap=True),
                *([metric(logprob, 3)] if show_logprobs else []),
                f"{p:.6%}" if p is not None else "—",
            )
        else:
            table.add_row(
                Text(label, style=style),
                Text(f"{code!r} / {token_id}", style=style),
                metric(logprob, 6),
                f"{p:.6%}" if p is not None else "—",
                Text("█" * width + "░" * (24 - width), style="green"),
            )
    values = [number(p) for p in probabilities]
    entropy = "not recorded"
    if values and all(p is not None and 0 <= p <= 1 for p in values):
        total = sum(p for p in values if p is not None)
        if math.isclose(total, 1.0, abs_tol=1e-3):
            h = -sum(p * math.log(p) for p in values if p)
            normalized = h / math.log(len(values)) if len(values) > 1 else 0.0
            entropy = f"{normalized:.6f} (0 concentrated · 1 uniform)"
        else:
            entropy = f"invalid probability sum {total:.6f}"
    summary = Text(
        f"Entropy: {entropy.split(' ')[0]}\n"
        if compact
        else f"Normalized entropy: {entropy}\n",
        style="bold",
    )
    summary.append(
        f"Label mass: {metric(record.get('label_mass'), 6)}"
        + (
            f" · {len(probabilities)} labels"
            if compact
            else f" · entropy uses all {len(probabilities)} recorded probabilities"
        ),
        style="dim",
    )
    if len(shown) < count:
        summary.append(f" · showing {len(shown)} of {count} candidates", style="dim")
    return Panel(
        Group(table if count else Text("Waiting for probabilities…"), summary),
        title="Answer probabilities"
        if compact
        else "Candidate logprobs → normalized probabilities",
        border_style="green",
    )


def short_json(value: Any, limit: int = 180) -> str:
    value = json.dumps(value, ensure_ascii=False)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def result_panel(trace: Trace, record: dict[str, Any], *, compact: bool) -> Panel:
    request_id = str(record.get("request_id", ""))
    question_id = str(record.get("question_id", "?"))
    response = trace.responses.get(request_id, {})
    answers = response.get("answers", {})
    answer = answers.get(question_id) if isinstance(answers, dict) else None
    draws = sum(
        sample.get("request_id") == request_id
        and sample.get("question_id") == question_id
        for sample in trace.samples
    )
    if answer is None:
        body: Any = Text("Waiting for the response event…", style="dim")
    elif compact and isinstance(answer, dict):
        kind = answer.get("type")
        if kind == "noul":
            summary = f"Noul · P(yes) = {metric(answer.get('noul'), 8)}"
        elif kind == "choice":
            summary = f"Choice · {answer.get('choice')}"
        else:
            summary = f"Score · {metric(answer.get('score'), 6)}"
        body = Text(summary, style="bold magenta")
        if draws > 1:
            body.append(
                f"\nMean of {draws} draws, not just this cell read.", style="dim"
            )
    else:
        body = Pretty(answer)
        if draws > 1:
            body = Group(
                body,
                Text(
                    "Probabilities are averaged across draws; "
                    "the candidate table shows the selected draw.",
                    style="dim",
                ),
            )
    return Panel(
        body,
        title=f"Typed result · {draws} recorded draw{'s' if draws != 1 else ''}",
        border_style="magenta",
    )


def question_context(record: dict[str, Any]) -> Any:
    question = record.get("question", {})
    if record.get("question_placement") == "canvas":
        question = record.get("fixed_question_json", question)
        if isinstance(question, str):
            with suppress(ValueError):
                question = json.loads(question)
    return question


def shared_image_count(record: dict[str, Any]) -> int | None:
    count = record.get("image_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    if isinstance(record.get("shared_images"), list):
        return len(record["shared_images"])
    return None


def context_panel(
    record: dict[str, Any], *, full: bool = False, preview_limit: int = 180
) -> Panel:
    question = question_context(record)
    on_canvas = record.get("question_placement") == "canvas"
    state_label = "SHARED PROMPT STATE" if on_canvas else "STATE"
    question_label = (
        "FIXED QUESTION JSON ON CANVAS" if on_canvas else "QUESTION / CRITERIA"
    )
    if full:
        content: list[Any] = [
            Text(question_label, style="bold"),
            Pretty(question, expand_all=True),
        ]
        if on_canvas:
            metadata = {
                key: record[key]
                for key in (
                    "shared_prompt_hash",
                    "shared_prompt_tokens",
                    "question_canvas_tokens",
                    "cached_prompt_tokens",
                    "created_cache_tokens",
                )
                if key in record
            }
            if metadata:
                content.extend(
                    [
                        Text("\nRECORDED PROMPT/CANVAS METADATA", style="bold"),
                        Pretty(metadata, expand_all=True),
                        Text(
                            "Prompt hash identifies equal prompts, not cache hits. "
                            "Null cache counts mean upstream usage was unreported.",
                            style="dim",
                        ),
                    ]
                )
            content.extend(
                [
                    Text("\n" + state_label, style="bold"),
                    Pretty(record.get("state"), expand_all=True),
                ]
            )
        else:
            content = [
                Text(state_label, style="bold"),
                Pretty(record.get("state"), expand_all=True),
                Text("\n" + question_label, style="bold"),
                Pretty(question, expand_all=True),
            ]
        if (image_count := shared_image_count(record)) is not None:
            image_metadata = [
                {
                    key: image[key]
                    for key in ("sha256", "mime_type", "width", "height", "byte_size")
                    if key in image
                }
                for image in items(record, "shared_images")
                if isinstance(image, dict)
            ]
            content.extend(
                [
                    Text(f"\nSHARED PROMPT IMAGES: {image_count}", style="bold"),
                    Pretty(image_metadata, expand_all=True),
                    Text(
                        "Images belong to the shared prefill, not the token canvas. "
                        "Only metadata is recorded here; pixels are not rendered.",
                        style="dim",
                    ),
                ]
            )
        return Panel(
            Group(*content),
            title="Shared prompt / fixed question canvas"
            if on_canvas
            else "Complete state and question",
            border_style="dim",
        )
    instructions = (
        question.get("instructions") or question.get("description") or question
        if isinstance(question, dict)
        else question
    )
    context = Text(state_label + "\n", style="bold")
    context.append(short_json(record.get("state"), preview_limit), style="white")
    context.append(
        "\n\n" + ("CANVAS QUESTION" if on_canvas else "QUESTION") + "\n",
        style="bold",
    )
    context.append(short_json(instructions, preview_limit), style="white")
    if image_count := shared_image_count(record):
        context.append(f"\n\nSHARED IMAGES: {image_count} (prefill)", style="bold")
        images = items(record, "shared_images")
        if images and isinstance(images[0], dict):
            image = images[0]
            context.append(
                f"\n1: {image.get('mime_type', '?')} "
                f"{image.get('width', '?')}×{image.get('height', '?')}"
                f" · {image.get('byte_size', '?')} B",
                style="white",
            )
        context.append("\nd details: metadata only", style="dim")
    return Panel(
        context,
        title="Shared prompt / question on canvas"
        if on_canvas
        else "What the model is judging",
        border_style="dim",
    )


def placement_metadata(record: dict[str, Any]) -> Text:
    metadata = Text("Question: fixed JSON canvas", style="cyan")
    if prompt_hash := record.get("shared_prompt_hash"):
        metadata.append(f" · shared prompt {str(prompt_hash)[:12]}")
    for key, label in (
        ("shared_prompt_tokens", "prompt tokens"),
        ("question_canvas_tokens", "used canvas tokens"),
    ):
        if key in record and record[key] is not None:
            metadata.append(f" · {label}: {record[key]}")
    if "cached_prompt_tokens" in record:
        cached = record["cached_prompt_tokens"]
        metadata.append(
            f" · upstream cached: {cached if cached is not None else 'unreported'}"
        )
    return metadata


def draws_panel(trace: Trace, record: dict[str, Any]) -> Panel | None:
    draws = [
        sample
        for sample in trace.samples
        if sample.get("request_id") == record.get("request_id")
        and sample.get("question_id") == record.get("question_id")
    ]
    if len(draws) <= 1:
        return None
    table = Table(box=box.SIMPLE, expand=True)
    for name in ("Draw", "Seed token ID", "Selected label", "Selected P", "Latency ms"):
        table.add_column(name)
    for sample in sorted(draws, key=lambda sample: sample.get("sample", 0)):
        position = answer_slot(sample)
        seed = items(sample, "canvas_tokens")
        seed_id = (
            seed[position] if position is not None and position < len(seed) else "?"
        )
        token_id = selected_token(sample)
        candidates = items(sample, "candidate_token_ids")
        labels = items(sample, "candidate_labels")
        probabilities = items(sample, "probabilities")
        selected = candidates.index(token_id) if token_id in candidates else None
        label = (
            labels[selected] if selected is not None and selected < len(labels) else "?"
        )
        probability = (
            probabilities[selected]
            if selected is not None and selected < len(probabilities)
            else None
        )
        table.add_row(
            str(sample.get("sample", "?")),
            str(seed_id),
            Text(str(label)),
            metric(probability, 8),
            metric(sample.get("latency_ms"), 2),
            style="bold cyan" if sample is record else None,
        )
    return Panel(
        Group(
            table,
            Text(
                "Highlighted row is this draw. The typed result averages full "
                "distributions, not the selected labels.",
                style="dim",
            ),
        ),
        title="Independent draws for this question",
        border_style="cyan",
    )


def parent_context(trace: Trace, record: dict[str, Any]) -> dict[str, Any]:
    request_id = str(record.get("request_id", ""))
    parent = trace.parents_by_id.get(str(record.get("parent_id", "")))
    if parent is None or parent.get("request_id") != record.get("request_id"):
        parent = trace.parents.get(request_id, {})
    context: dict[str, Any] = {}
    for source in (
        trace.responses.get(request_id, {}),
        parent,
        record,
    ):
        context.update(
            {
                key: source[key]
                for key in (
                    "parent_id",
                    "child_count",
                    "children",
                    "status",
                    "execution_evidence",
                )
                if key in source
            }
        )
    if request_id in trace.responses:
        context.setdefault("status", "completed")
    return context if context.get("parent_id") or request_id in trace.parents else {}


def parent_panel(trace: Trace, record: dict[str, Any]) -> Panel | None:
    parent = parent_context(trace, record)
    if not parent:
        return None
    content: list[Any] = [
        Text(f"Parent ID: {parent.get('parent_id') or 'not recorded'}"),
        Text(
            f"Status: {parent.get('status', 'not recorded')}"
            f" · children: {parent.get('child_count', 'not recorded')}"
        ),
    ]
    evidence = parent.get("execution_evidence")
    count = evidence.get("model_forward_count") if isinstance(evidence, dict) else None
    if (
        isinstance(evidence, dict)
        and evidence.get("verified") is True
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(evidence.get("source"), str)
        and evidence["source"]
    ):
        content.append(
            Text(f"Model forwards: {count} (verified)\nEvidence: {evidence['source']}")
        )
    else:
        content.append(Text("Model forwards: unverified", style="dim"))
    children = items(parent, "children")
    if children:
        table = Table(box=box.SIMPLE, expand=True)
        for name in ("Choice index", "Question", "Draw", "Trace"):
            table.add_column(name)
        recorded = {
            (
                sample.get("choice_index"),
                sample.get("question_id"),
                sample.get("sample"),
            )
            for sample in trace.samples
            if sample.get("request_id") == record.get("request_id")
            and sample.get("parent_id") == parent.get("parent_id")
        }
        for child in children:
            if not isinstance(child, dict):
                continue
            choice = child.get("choice_index")
            key = (choice, child.get("question_id"), child.get("sample"))
            table.add_row(
                str(choice),
                Text(str(child.get("question_id", "?"))),
                str(child.get("sample", "?")),
                "recorded" if key in recorded else "not recorded",
                style="bold cyan"
                if key
                == (
                    record.get("choice_index"),
                    record.get("question_id"),
                    record.get("sample"),
                )
                else None,
            )
        content.append(table)
    else:
        content.append(Text("Child mapping was not recorded.", style="dim"))
    return Panel(Group(*content), title="Parallel sampling parent", border_style="cyan")


@dataclass
class DetailScroll:
    offset: int = 0
    maximum: int = 0
    page_size: int = 1

    def move(self, key: str) -> None:
        if key == "home":
            self.offset = 0
        elif key == "end":
            self.offset = self.maximum
        else:
            direction = -1 if key in {"page_up", "["} else 1
            self.offset += direction * max(1, self.page_size - 1)
            self.offset = max(0, min(self.offset, self.maximum))


@dataclass
class DetailsViewport:
    header: Group
    body: Group
    footer: Text
    height: int
    scroll: DetailScroll

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        options = options.update(height=None)
        header = console.render_lines(self.header, options, pad=False)
        footer = console.render_lines(self.footer, options, pad=False)
        body = console.render_lines(self.body, options, pad=False)
        self.scroll.page_size = max(1, self.height - len(header) - len(footer) - 2)
        self.scroll.maximum = max(0, len(body) - self.scroll.page_size)
        self.scroll.offset = min(max(0, self.scroll.offset), self.scroll.maximum)
        start = self.scroll.offset
        end = min(len(body), start + self.scroll.page_size)
        status = Text(
            f"Details lines {start + 1}–{end}/{len(body)} · "
            "PgUp/PgDn or [/] scroll · Home/End · d canvas",
            style="bold yellow",
        )
        for line in header + body[start:end]:
            yield from line
            yield Segment.line()
        yield status
        for index, line in enumerate(footer):
            yield from line
            if index < len(footer) - 1:
                yield Segment.line()


def render(
    trace: Trace,
    index: int,
    playing: bool,
    follow: bool,
    limit: int,
    *,
    output: bool = False,
    selected_position: int | None = None,
    details: bool = False,
    width: int = 120,
    height: int | None = None,
    detail_scroll: DetailScroll | None = None,
) -> Group:
    mode = "FOLLOWING" if follow and playing else "PLAYING" if playing else "PAUSED"
    title = Text("DIFFUSION GEMMA  /  BLOCK CANVAS", style="bold cyan")
    title.append(f"    {mode}", style="bold yellow")
    title.append("    OUTPUT" if output else "    INPUT", style="bold green")
    parts: list[Any] = [title]
    footer = Text(
        "arrows / hjkl inspect · v input/output · n/p records · "
        "space replay · d details · q quit",
        style="bold",
    )
    if trace.samples:
        index = min(max(index, 0), len(trace.samples) - 1)
        record = trace.samples[index]
        if (image_count := shared_image_count(record)) is not None:
            title.append(f"    Shared images: {image_count}", style="cyan")
        question_id = str(record.get("question_id", "?"))
        reasoning = record.get("reasoning", "not recorded")
        reasoning = "off" if reasoning is False else reasoning
        step = record.get("denoise_steps", "not recorded")
        constraint_metadata = ""
        if backend := record.get("constraint_backend"):
            constraint_metadata = f" · constraints {backend}"
            if version := record.get("constraint_version"):
                constraint_metadata += f" {version}"
        record_heading = (
            f"Record {index + 1}/{len(trace.samples)} · question {question_id}"
            f" ({record.get('kind', '?')}) · draw {record.get('sample', '?')}"
            f" · {trace.path.name}"
        )
        if parent := parent_context(trace, record):
            parent_id = visible(parent.get("parent_id") or "not recorded")
            parent_width = max(12, width // 5)
            if len(parent_id) > parent_width:
                parent_id = parent_id[: parent_width - 1] + "…"
            choice = record.get("choice_index", "?")
            ordinal = choice + 1 if isinstance(choice, int) else "?"
            record_heading = (
                f"Parent {parent_id} · child {ordinal}/{parent.get('child_count', '?')}"
                f" (choice {choice}) · {question_id} · draw {record.get('sample', '?')}"
            )
        parts.append(
            Text(
                f"{record_heading}\n"
                f"Configured: reasoning {reasoning} · denoise passes {step}"
                f"{constraint_metadata}"
                f" · recorded latency {metric(record.get('latency_ms'), 2)} ms",
                style="white",
            )
        )
        if record.get("question_placement") == "canvas":
            parts.append(placement_metadata(record))
        if details:
            content: list[Any] = [
                context_panel(record, full=True),
                candidates_panel(record, limit),
            ]
            if parent_details := parent_panel(trace, record):
                content.insert(0, parent_details)
            if draws := draws_panel(trace, record):
                content.append(draws)
            content.append(result_panel(trace, record, compact=False))
            if height is not None:
                if trace.errors:
                    content.append(
                        Panel(
                            Text("\n".join(trace.errors)),
                            title="Trace errors",
                            border_style="red",
                        )
                    )
                return Group(
                    DetailsViewport(
                        Group(*parts),
                        Group(*content),
                        footer,
                        height,
                        detail_scroll if detail_scroll is not None else DetailScroll(),
                    )
                )
            parts.extend(content)
        else:
            sidebar_width = max(38, width // 5)
            canvas_width = width - sidebar_width - 2 if width >= 150 else width
            canvas_height = height - 6 if height is not None else None
            canvas = canvas_panel(
                record,
                output=output,
                selected_position=selected_position,
                compact=width < 96,
                width=canvas_width,
                height=canvas_height,
            )
            if width >= 150:
                extra_rows = max(0, (height or 38) - 38) // 4
                candidate_limit = 6 + extra_rows
                preview_limit = (sidebar_width - 4) * (180 / 52 + extra_rows)
                layout = Table.grid(expand=True, padding=(0, 2))
                layout.add_column(width=canvas_width)
                layout.add_column(ratio=1)
                layout.add_row(
                    canvas,
                    Group(
                        context_panel(record, preview_limit=int(preview_limit)),
                        candidates_panel(
                            record,
                            min(limit, candidate_limit) if limit else candidate_limit,
                            compact=True,
                            show_logprobs=sidebar_width >= 46,
                        ),
                        result_panel(trace, record, compact=True),
                    ),
                )
                parts.append(layout)
            else:
                parts.extend([canvas, result_panel(trace, record, compact=True)])
                on_canvas = record.get("question_placement") == "canvas"
                parts.append(
                    Text(
                        ("Shared prompt state: " if on_canvas else "State: ")
                        + short_json(record.get("state"), 100)
                    )
                )
                parts.append(
                    Text(
                        ("Canvas question: " if on_canvas else "Question: ")
                        + short_json(question_context(record), 100)
                    )
                )
    else:
        if trace.parents:
            latest_parent = next(reversed(trace.parents.values()))
            if parent_details := parent_panel(trace, latest_parent):
                parts.append(parent_details)
        parts.append(
            Panel(
                (
                    "No sample events recorded.\n"
                    if trace.parents
                    else "Waiting for recorded sample events…\n"
                )
                + "This viewer reads a JSONL trace; it never runs inference.",
                border_style="cyan",
            )
        )
    if trace.errors:
        parts.append(
            Panel(
                Text("\n".join(trace.errors)), title="Trace errors", border_style="red"
            )
        )
    parts.append(footer)
    return Group(*parts)


@contextmanager
def keyboard():
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def input_keys(buffer: str) -> tuple[list[str], str]:
    """Parse complete key presses, retaining partial terminal escape sequences."""
    keys = []
    sequences = {
        "A": "up",
        "B": "down",
        "C": "right",
        "D": "left",
        "H": "home",
        "F": "end",
        "1~": "home",
        "4~": "end",
        "5~": "page_up",
        "6~": "page_down",
        "7~": "home",
        "8~": "end",
    }
    while buffer:
        if buffer[0] == "\x1b":
            if len(buffer) < 2 or (buffer[1] in "[O" and len(buffer) < 3):
                break
            if buffer[1] in "[O":
                end = next(
                    (i for i in range(2, len(buffer)) if "@" <= buffer[i] <= "~"),
                    None,
                )
                if end is None:
                    break
                if key := sequences.get(buffer[2 : end + 1]):
                    keys.append(key)
                buffer = buffer[end + 1 :]
                continue
        else:
            keys.append(buffer[0])
        buffer = buffer[1:]
    return keys, buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Recorded JSONL events")
    parser.add_argument("--follow", action="store_true", help="Watch for new events")
    parser.add_argument(
        "--snapshot", "--no-live", action="store_true", help="Render once without a TTY"
    )
    parser.add_argument(
        "--sample", type=int, help="Zero-based sample record to display"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Seconds per input/output frame"
    )
    parser.add_argument("--start-paused", action="store_true", help="Wait for keys")
    parser.add_argument(
        "--details", action="store_true", help="Start with complete decision details"
    )
    parser.add_argument(
        "--output", action="store_true", help="Start on the output canvas"
    )
    parser.add_argument(
        "--max-candidates", type=int, default=12, help="Visible candidates; 0 shows all"
    )
    parser.add_argument(
        "--width", type=int, help="Override terminal width for snapshots"
    )
    parser.add_argument(
        "--export-svg", type=Path, help="Save a snapshot as SVG and exit"
    )
    parser.add_argument(
        "--export-html", type=Path, help="Save a snapshot as HTML and exit"
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    if args.sample is not None and args.sample < 0:
        parser.error("--sample must be nonnegative")
    if args.width is not None and args.width < 40:
        parser.error("--width must be at least 40")
    export = bool(args.export_svg or args.export_html)
    console = Console(record=export, width=args.width)
    trace = Trace(args.trace)
    snapshot = (
        args.snapshot or export or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    trace.read(final=snapshot or not args.follow)
    index = args.sample if args.sample is not None else 0
    output = args.output
    selected_position = None
    details = args.details
    detail_scroll = DetailScroll()
    if snapshot:
        if args.sample is None:
            index = len(trace.samples) - 1
        console.print(
            render(
                trace,
                index,
                False,
                args.follow,
                args.max_candidates,
                output=output,
                details=details,
                width=console.width,
            )
        )
        if args.export_svg:
            console.save_svg(
                str(args.export_svg),
                title="Diffusion Gemma · recorded decisions",
                clear=False,
            )
        if args.export_html:
            console.save_html(str(args.export_html), clear=False)
        return
    playing = not args.start_paused
    advanced = time.monotonic()
    pending_keys = ""
    try:
        with keyboard(), Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                if args.follow:
                    trace.read()
                index = max(0, min(index, len(trace.samples) - 1))
                now = time.monotonic()
                if playing and trace.samples and now - advanced >= args.interval:
                    if not output:
                        output = True
                    elif index < len(trace.samples) - 1:
                        index += 1
                        output = False
                        selected_position = None
                        detail_scroll.offset = 0
                    elif not args.follow:
                        playing = False
                    advanced = now
                live.update(
                    render(
                        trace,
                        index,
                        playing,
                        args.follow,
                        args.max_candidates,
                        output=output,
                        selected_position=selected_position,
                        details=details,
                        width=console.width,
                        height=console.height,
                        detail_scroll=detail_scroll,
                    ),
                    refresh=True,
                )
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                raw = os.read(sys.stdin.fileno(), 1024)
                if not raw:
                    break
                keys, pending_keys = input_keys(
                    pending_keys + raw.decode(errors="ignore")
                )
                for key in keys:
                    if key == "q":
                        return
                    if key in {"n", "p"}:
                        index += 1 if key == "n" else -1
                        index = max(0, min(index, len(trace.samples) - 1))
                        selected_position = None
                        detail_scroll.offset = 0
                        playing = output = False
                    elif key == " ":
                        playing = not playing
                        advanced = time.monotonic()
                    elif key == "v":
                        output = not output
                        playing = False
                    elif key == "d":
                        details = not details
                        detail_scroll.offset = 0
                        playing = False
                    elif details and key in {
                        "page_up",
                        "page_down",
                        "[",
                        "]",
                        "home",
                        "end",
                    }:
                        detail_scroll.move(key)
                        playing = False
                    elif trace.samples:
                        record = trace.samples[index]
                        count = len(items(record, "canvas_tokens"))
                        columns, _ = canvas_grid_shape(count)
                        moves = {
                            "h": -1,
                            "left": -1,
                            "l": 1,
                            "right": 1,
                            "j": columns,
                            "down": columns,
                            "k": -columns,
                            "up": -columns,
                        }
                        if key in moves and count:
                            position = selected_position
                            if position is None:
                                position = answer_slot(record) or 0
                            selected_position = max(
                                0, min(count - 1, position + moves[key])
                            )
                            playing = False
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
