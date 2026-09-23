# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TypeSafe-compatible decisions from constrained Diffusion Gemma reads."""

import argparse
import asyncio
import binascii
import hashlib
import json
import math
import random
import string
import time
import uuid
import warnings
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import pybase64 as base64
import xgrammar as xgr
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

JSONContent = str | dict[str, JsonValue] | list[JsonValue]
CanvasFormat = Literal["json", "chat", "chat_single_user"]
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
MAX_CHOICES = 100
ANSWER_CODES = tuple(string.ascii_uppercase)
EXTENDED_ANSWER_CODES = tuple(
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
)[:MAX_CHOICES]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="allow")
    instructions: JSONContent | None = None


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[str, JSONContent | None] | None = None


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, JSONContent | None] = Field(
        min_length=1, max_length=MAX_CHOICES
    )


class Score(QuestionBase):
    type: Literal["score"]
    criteria: list[JSONContent] = Field(min_length=1, max_length=26)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(repr=False, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64)


class ReasoningBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens: StrictInt = Field(ge=1, le=512)
    steps: StrictInt = Field(ge=1, le=64)


class SystemOneRequest(BaseModel):
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1, max_length=128)
    model: str | None = None
    system_prompt: str | None = Field(
        default=None,
        strict=True,
        repr=False,
        description="Replaces the default system instructions; null uses the default.",
    )
    samples: StrictInt = Field(default=1, ge=1, le=32)
    adaptive_samples: StrictBool = False
    max_samples: StrictInt = Field(default=4, ge=1, le=32)
    adaptive_threshold: float = Field(
        default=0.1, strict=True, ge=0, le=1, allow_inf_nan=False
    )
    steps: StrictInt = Field(default=1, ge=1, le=32)
    self_conditioning: StrictBool = True
    self_conditioning_scope: Literal["answer", "canvas"] = "answer"
    canvas_format: CanvasFormat = "json"
    reason_mode: StrictBool = False
    reason_tokens: StrictInt = Field(default=32, ge=1, le=512)
    reason_steps: StrictInt = Field(default=8, ge=1, le=64)
    reason_ensemble: list[ReasoningBudget] | None = Field(
        default=None, min_length=1, max_length=32
    )
    reason_schedule_steps: StrictInt | None = Field(default=None, ge=1)
    reason_entropy_bound: float | None = Field(
        default=None, strict=True, ge=0, allow_inf_nan=False
    )
    reason_answer_noise: Literal["candidates", "vocab"] = "candidates"
    reason_initial_answer_noise: Literal["candidates", "vocab"] | None = None
    return_diagnostics: StrictBool = False
    seed: StrictInt = 0
    cache_salt: str | None = Field(default=None, min_length=1, max_length=1024)
    images: list[ImageInput] = Field(default_factory=list, max_length=MAX_IMAGES)

    @model_validator(mode="after")
    def validate_adaptive_budget(self) -> "SystemOneRequest":
        if self.adaptive_samples and self.max_samples < self.samples:
            raise ValueError("max_samples must be at least samples in adaptive mode")
        return self

    @model_validator(mode="after")
    def validate_reason_ensemble(self) -> "SystemOneRequest":
        if self.reason_ensemble is not None:
            if not self.reason_mode or self.canvas_format != "chat_single_user":
                raise ValueError(
                    "reason_ensemble requires reason_mode and chat_single_user"
                )
            if self.adaptive_samples:
                raise ValueError("reason_ensemble requires fixed sampling")
            if len(self.reason_ensemble) * self.samples > 32:
                raise ValueError(
                    "reason_ensemble permits at most 32 samples per question"
                )
        return self

    @model_validator(mode="after")
    def validate_reason_schedule(self) -> "SystemOneRequest":
        if self.reason_ensemble is not None:
            if self.reason_schedule_steps is not None and any(
                self.reason_schedule_steps < budget.steps
                for budget in self.reason_ensemble
            ):
                raise ValueError(
                    "reason_schedule_steps must be at least every budget's steps"
                )
            return self
        if (
            self.reason_mode
            and self.reason_schedule_steps is not None
            and self.reason_schedule_steps < self.reason_steps
        ):
            raise ValueError("reason_schedule_steps must be at least reason_steps")
        return self


@dataclass(frozen=True)
class SharedContext:
    messages: list[dict[str, Any]] = field(repr=False)
    prompt_hash: str
    images: list[dict[str, Any]]
    seed_prompt_hash: str | None = None


class InvalidImage(ValueError):
    """An image rejection message that contains no request payload."""


def validation_error_response(
    error: RequestValidationError | ValidationError,
    *,
    prepend_body: bool = False,
) -> JSONResponse:
    fields = {
        "body",
        "images",
        "url",
        "state",
        "questions",
        "model",
        "system_prompt",
        "samples",
        "adaptive_samples",
        "max_samples",
        "adaptive_threshold",
        "steps",
        "self_conditioning",
        "self_conditioning_scope",
        "canvas_format",
        "reason_mode",
        "reason_tokens",
        "reason_steps",
        "reason_ensemble",
        "tokens",
        "reason_schedule_steps",
        "reason_entropy_bound",
        "reason_answer_noise",
        "reason_initial_answer_noise",
        "return_diagnostics",
        "seed",
        "cache_salt",
        "type",
        "instructions",
        "criteria",
    }
    detail = []
    for item in error.errors():
        location = list(item["loc"])
        if prepend_body:
            location.insert(0, "body")
        detail.append(
            {
                "type": item["type"],
                "loc": [
                    part if isinstance(part, int) or part in fields else "<field>"
                    for part in location
                ],
                "msg": "Invalid request field; see error type and location",
            }
        )
    return JSONResponse(status_code=422, content={"detail": detail})


def prepare_images(
    images: list[ImageInput],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    total_bytes = 0
    for index, source in enumerate(images):
        header, separator, encoded = source.url.partition(",")
        mime_type = header.removeprefix("data:").removesuffix(";base64")
        if (
            not separator
            or mime_type not in IMAGE_FORMATS
            or header != f"data:{mime_type};base64"
        ):
            raise InvalidImage(
                f"Image {index} must be an inline PNG, JPEG, or WebP base64 data URL"
            )
        if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise InvalidImage(f"Image {index} exceeds {MAX_IMAGE_BYTES} decoded bytes")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise InvalidImage(f"Image {index} has invalid base64") from None
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise InvalidImage(
                f"Image {index} must contain 1..{MAX_IMAGE_BYTES} decoded bytes"
            )
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise InvalidImage(
                f"Images exceed {MAX_TOTAL_IMAGE_BYTES} total decoded bytes"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(
                    BytesIO(data), formats=list(IMAGE_FORMATS.values())
                ) as image:
                    if image.format != IMAGE_FORMATS[mime_type]:
                        raise InvalidImage(
                            f"Image {index} MIME type does not match its bytes"
                        )
                    width, height = image.size
                    if width * height > MAX_IMAGE_PIXELS:
                        raise InvalidImage(
                            f"Image {index} exceeds {MAX_IMAGE_PIXELS} pixels"
                        )
                    if getattr(image, "n_frames", 1) != 1:
                        raise InvalidImage(
                            f"Image {index} must be a single still image"
                        )
                    image.verify()
                with Image.open(
                    BytesIO(data), formats=[IMAGE_FORMATS[mime_type]]
                ) as image:
                    image.load()
        except InvalidImage:
            raise
        except (
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            raise InvalidImage(
                f"Image {index} is not a valid decodable image"
            ) from None
        metadata.append(
            {
                "sha256": hashlib.sha256(data).hexdigest(),
                "mime_type": mime_type,
                "width": width,
                "height": height,
                "byte_size": len(data),
            }
        )
        url = f"data:{mime_type};base64," + base64.b64encode(data).decode("ascii")
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks, metadata


@dataclass(frozen=True)
class Template:
    tokens: tuple[int, ...]
    answer_position: int
    candidate_ids: tuple[int, ...]
    codes: tuple[str, ...]
    question_json: dict[str, Any]
    content_tokens: int
    trace_canvas_text: tuple[str, ...] | None
    reasoning_positions: tuple[int, ...] = ()
    reasoning_token_ids: tuple[int, ...] = ()
    reasoning_forbidden_token_ids: tuple[int, ...] = ()
    canvas_format: CanvasFormat = "json"


@dataclass(frozen=True)
class PreparedRead:
    question_id: str
    question: QuestionBase
    sample_index: int
    template: Template
    canvas: tuple[int, ...]
    steps: int
    self_conditioning: bool
    self_conditioning_scope: Literal["answer", "canvas"]
    reasoning_steps: int = 0
    reasoning_answer_noise: Literal["candidates", "vocab"] = "candidates"
    reasoning_schedule_steps: int | None = None
    reasoning_entropy_bound: float | None = None
    reasoning_seed: int | None = None

    def descriptor(self) -> dict[str, list[int] | int | float | bool | str]:
        descriptor: dict[str, list[int] | int | float | bool | str] = {
            "seed_canvas": list(self.canvas),
            "answer_positions": [self.template.answer_position],
            "logprob_token_ids": list(self.template.candidate_ids),
            "answer_only_logprobs": True,
            "steps": self.steps,
            "self_conditioning": self.self_conditioning,
        }
        if self.self_conditioning_scope != "answer":
            descriptor["self_conditioning_scope"] = self.self_conditioning_scope
        if self.template.reasoning_positions:
            descriptor.update(
                {
                    "reasoning_positions": list(self.template.reasoning_positions),
                    "reasoning_forbidden_token_ids": list(
                        self.template.reasoning_forbidden_token_ids
                    ),
                    "reasoning_steps": self.reasoning_steps,
                }
            )
            if self.reasoning_answer_noise != "candidates":
                descriptor["reasoning_answer_noise"] = self.reasoning_answer_noise
            if self.reasoning_schedule_steps is not None:
                descriptor["reasoning_schedule_steps"] = self.reasoning_schedule_steps
            if self.reasoning_entropy_bound is not None:
                descriptor["reasoning_entropy_bound"] = self.reasoning_entropy_bound
            if self.reasoning_seed is not None:
                descriptor["reasoning_seed"] = self.reasoning_seed
        return descriptor


COMMON_INSTRUCTIONS = (
    "Read the shared state. The bidirectional answer canvas contains a fixed JSON "
    "decision form with a name, description, and options. Fill only its answer "
    "field with the correct allowed option code. All other fields and tokens "
    "must remain unchanged. Do not explain your answer."
)
REASON_INSTRUCTIONS = (
    "Solve the decision in the canvas using the shared state. Write concise "
    "working in reasoning, including any calculation or rule needed to choose an "
    "option. Use the available space for relevant checks, with no JSON field "
    "names or copied state. Then set answer to the option code that matches your "
    "reasoning. Leave all metadata and JSON punctuation unchanged. Only plain "
    "text is permitted inside the reasoning string."
)
CHAT_INSTRUCTIONS = (
    "Answer a fixed set of questions about the state the user provides. "
    "Each question lists its allowed answers; reply with exactly one label "
    "per question.\n\n"
    'Reply with one line per question, in this order, formatted as "id: label".'
)
THOUGHT_INSTRUCTIONS = (
    "Answer the question about the shared state. Work out the relevant facts "
    "and calculations in the thought channel. Check the exact question and any "
    "negation, quantifiers, or exceptions. After the thought channel closes, "
    "reply with the allowed label. Keep the question and option text unchanged."
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_probabilities(logprobs: Sequence[float]) -> list[float]:
    if not logprobs or not all(math.isfinite(p) for p in logprobs):
        raise ValueError("The model must return a finite logprob for every candidate")
    peak = max(logprobs)
    weights = [math.exp(p - peak) for p in logprobs]
    total = sum(weights)
    return [weight / total for weight in weights]


def confidence(probabilities: Sequence[float]) -> float:
    """One minus normalized entropy; this is not a calibration estimate."""
    if len(probabilities) == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def labels_for(question: QuestionBase) -> list[str]:
    if isinstance(question, Noul):
        return ["yes", "no"]
    if isinstance(question, Choice):
        return list(question.criteria)
    assert isinstance(question, Score)
    return [str(index) for index in range(len(question.criteria))]


def decision_metadata(question: QuestionBase) -> tuple[tuple[str, ...], dict[str, Any]]:
    count = len(labels_for(question))
    if isinstance(question, Noul):
        codes = ("yes", "no")
        criteria = question.criteria or {}
        options = {
            code: criteria.get(key) if criteria.get(key) is not None else code
            for code, key in zip(codes, ("true", "false"))
        }
    else:
        codes = (
            ANSWER_CODES[:count]
            if count <= len(ANSWER_CODES)
            else EXTENDED_ANSWER_CODES[:count]
        )
        if isinstance(question, Choice):
            options = {
                code: label
                if description is None
                else {"label": label, "description": description}
                for code, (label, description) in zip(codes, question.criteria.items())
            }
        else:
            assert isinstance(question, Score)
            options = dict(zip(codes, question.criteria))
    metadata = {"name": "decision", "type": question.type, "options": options}
    if question.instructions is not None:
        metadata["description"] = question.instructions
    return tuple(codes), metadata


def answer_for(question: QuestionBase, probabilities: list[float]) -> dict[str, Any]:
    if isinstance(question, Noul):
        return {"type": "noul", "noul": probabilities[0]}
    labels = labels_for(question)
    answer: dict[str, Any] = {
        "type": question.type,
        "probabilities": dict(zip(labels, probabilities)),
        "confidence": confidence(probabilities),
    }
    if isinstance(question, Choice):
        answer["choice"] = labels[
            max(range(len(labels)), key=probabilities.__getitem__)
        ]
    else:
        answer["score"] = sum(i * p for i, p in enumerate(probabilities))
        answer["legend"] = dict(zip(labels, question.criteria))
    return answer


class DecisionReader:
    def __init__(
        self,
        tokenizer: Any,
        upstream: str,
        model: str,
        canvas_length: int = 256,
        concurrency: int = 32,
        timeout: float = 120,
        trace: Path | None = None,
        api_key: str | None = None,
        grammar_compiler: xgr.GrammarCompiler | None = None,
        trim_canvas_padding: bool = False,
        minimum_canvas_padding: int = 0,
        full_canvas_types: frozenset[str] = frozenset(),
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.canvas_length = canvas_length
        self.trace = trace
        self._templates: dict[str, Template] = {}
        self.vocab_size = len(tokenizer)
        terminator = tokenizer.encode("<turn|>", add_special_tokens=False)
        if len(terminator) != 1 or terminator[0] not in tokenizer.all_special_ids:
            raise ValueError("This example requires the Diffusion Gemma tokenizer")
        self.terminator_id = terminator[0]
        self._reasoning_token_ids: tuple[int, ...] | None = None
        self._reasoning_forbidden_token_ids: tuple[int, ...] = ()
        self._thought_token_ids: tuple[int, ...] | None = None
        self._thought_forbidden_token_ids: tuple[int, ...] = ()
        self._tokenizer_info = (
            None
            if grammar_compiler is not None
            else xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=self.vocab_size,
                stop_token_ids=[self.terminator_id],
            )
        )
        self.grammar_compiler = grammar_compiler or xgr.GrammarCompiler(
            self._tokenizer_info,
            max_threads=1,
            cache_enabled=True,
        )
        self.constraint_version = version("xgrammar")
        self.trim_canvas_padding = trim_canvas_padding
        self.minimum_canvas_padding = minimum_canvas_padding
        self.full_canvas_types = full_canvas_types
        self.semaphore = asyncio.Semaphore(concurrency)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=upstream.rstrip("/") + "/",
            timeout=timeout,
            headers=headers,
            limits=httpx.Limits(max_connections=concurrency),
        )

    async def close(self) -> None:
        await self.client.aclose()

    def record(self, event: dict[str, Any]) -> None:
        if self.trace is not None:
            with self.trace.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"timestamp": time.time(), **event}) + "\n")

    def template(
        self,
        name: str,
        question: QuestionBase,
        reason_tokens: int = 0,
        canvas_format: CanvasFormat = "json",
    ) -> Template:
        codes, metadata = decision_metadata(question)
        count = len(codes)
        key = canonical_json(metadata)
        if canvas_format != "json":
            cache_key = canonical_json([canvas_format, metadata, reason_tokens])
            if cache_key not in self._templates:
                self._templates[cache_key] = (
                    self._single_user_template(
                        self.template(name, question, reason_tokens, "chat")
                    )
                    if canvas_format == "chat_single_user"
                    else self._chat_template(
                        name, question, codes, metadata, reason_tokens
                    )
                )
            return self._templates[cache_key]
        cache_key = (
            key if not reason_tokens else canonical_json([metadata, reason_tokens])
        )
        if cached := self._templates.get(cache_key):
            return cached

        def completion(code: str) -> str:
            # Keep the answer last, outside the canonical fixed-metadata prefix.
            reasoning = ',"reasoning":"x"' if reason_tokens else ""
            return key[:-1] + reasoning + ',"answer":' + json.dumps(code) + "}"

        # An extra alternative locates the slot even for a one-option question.
        probes = codes if len(codes) > 1 else (codes[0], "B")
        forms = [completion(code) for code in probes]
        encoded = [
            self.tokenizer.encode(form, add_special_tokens=False) for form in forms
        ]
        base = encoded[0]
        reasoning_positions: tuple[int, ...] = ()
        reasoning_token_ids: tuple[int, ...] = ()
        if reason_tokens:
            alternate = self.tokenizer.encode(
                forms[0].replace(
                    ',"reasoning":"x","answer":', ',"reasoning":"y","answer":'
                ),
                add_special_tokens=False,
            )
            changed = [
                index
                for index, pair in enumerate(zip(base, alternate))
                if pair[0] != pair[1]
            ]
            if len(alternate) != len(base) or len(changed) != 1:
                raise ValueError("Reasoning must occupy a separate JSON string token")
            start = changed[0]
            reasoning_token_ids = self.reasoning_tokens()
            if (
                base[start] not in reasoning_token_ids
                or self.tokenizer.decode([base[start]]) != "x"
            ):
                raise ValueError("Reasoning tokens must preserve the JSON boundaries")
            encoded = [
                row[:start] + [base[start]] * reason_tokens + row[start + 1 :]
                for row in encoded
            ]
            base = encoded[0]
            reasoning_positions = tuple(range(start, start + reason_tokens))
        content_tokens = len(base) + 1
        if content_tokens > self.canvas_length:
            raise ValueError(
                f"Question {name!r} requires {content_tokens} canvas tokens; "
                f"the served canvas holds {self.canvas_length}. Shorten the question "
                "metadata or use a larger configured canvas. Nothing was truncated."
            )
        positions = {
            i
            for row in encoded[1:]
            for i, (a, b) in enumerate(zip(base, row))
            if a != b
        }
        if any(len(row) != len(base) for row in encoded) or len(positions) != 1:
            raise ValueError("Answer codes must occupy the same single token position")
        position = positions.pop()
        candidate_ids = tuple(row[position] for row in encoded[:count])
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Answer codes must map to distinct tokens")
        if reason_tokens:
            self._validate_reasoning_grammar(
                base, position, candidate_ids, key, codes, reasoning_positions
            )
        else:
            self._validate_grammar(
                base, position, candidate_ids, key[:-1] + ',"answer":', codes
            )
        tokens = base + [self.terminator_id]
        padding = self.canvas_length - len(tokens)
        if self.trim_canvas_padding and question.type not in self.full_canvas_types:
            padding = min(padding, self.minimum_canvas_padding)
        tokens += [self.tokenizer.pad_token_id] * padding
        template = Template(
            tuple(tokens),
            position,
            candidate_ids,
            tuple(codes),
            metadata,
            content_tokens,
            (
                tuple(self.tokenizer.decode([token]) for token in tokens)
                if self.trace is not None
                else None
            ),
            reasoning_positions,
            reasoning_token_ids,
            self._reasoning_forbidden_token_ids if reason_tokens else (),
        )
        self._templates[cache_key] = template
        return template

    def _chat_template(
        self,
        name: str,
        question: QuestionBase,
        codes: tuple[str, ...],
        metadata: dict[str, Any],
        reason_tokens: int,
    ) -> Template:
        instructions = question.instructions
        if instructions is None:
            instructions = ""
        elif not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        text = f"Question 0: {instructions.strip()}\n"
        if isinstance(question, Noul):
            criteria = question.criteria or {}
            for code, key in zip(codes, ("true", "false")):
                description = criteria.get(key)
                text += (
                    f"  {code}: {str(description).strip()}\n"
                    if description
                    else f"  {code}\n"
                )
        else:
            if isinstance(question, Score):
                count = len(codes)
                if count <= 9:
                    codes = tuple(str(index + 1) for index in range(count))
                choices = [(str(level), None) for level in question.criteria]
            else:
                assert isinstance(question, Choice)
                choices = list(question.criteria.items())
            for code, (label, description) in zip(codes, choices):
                text += f"  {code}: {label}"
                if description:
                    text += f" ({str(description).strip()})"
                text += "\n"
        head_text = (
            "<|turn>user\n" + text + "<turn|>\n<|turn>model\n<|channel>thought\n"
        )
        head = self.tokenizer.encode(head_text, add_special_tokens=False)
        if self.tokenizer.decode(head) != head_text:
            raise ValueError("Tokenizer changed the fixed chat canvas")
        probes = codes if len(codes) > 1 else (codes[0], "B")
        forms = [
            "<channel|>0: " + (json.dumps(code) if len(codes) > 26 else code)
            for code in probes
        ]
        tails = [
            self.tokenizer.encode(form, add_special_tokens=False) for form in forms
        ]
        changed = {
            index
            for tail in tails[1:]
            for index, (before, after) in enumerate(zip(tails[0], tail))
            if before != after
        }
        if any(len(tail) != len(tails[0]) for tail in tails) or len(changed) != 1:
            raise ValueError("Answer codes must occupy the same single token position")
        tail_answer = changed.pop()
        candidates = tuple(tail[tail_answer] for tail in tails[: len(codes)])
        if len(set(candidates)) != len(codes):
            raise ValueError("Answer codes must map to distinct tokens")
        if any(self.tokenizer.decode(tail) != form for tail, form in zip(tails, forms)):
            raise ValueError("Tokenizer changed a constrained chat answer")
        reasoning_positions = tuple(range(len(head), len(head) + reason_tokens))
        if reason_tokens:
            placeholders = [
                self.tokenizer.encode(
                    '{"reasoning":' + json.dumps(value) + ',"answer":"A"}',
                    add_special_tokens=False,
                )
                for value in ("x", "y")
            ]
            changed = [
                index
                for index, pair in enumerate(zip(*placeholders))
                if pair[0] != pair[1]
            ]
            if len(placeholders[0]) != len(placeholders[1]) or len(changed) != 1:
                raise ValueError("Reasoning must occupy separate token positions")
            placeholder = placeholders[0][changed[0]]
            if (
                placeholder not in self.reasoning_tokens()
                or self.tokenizer.decode([placeholder]) != "x"
            ):
                raise ValueError("Tokenizer changed the reasoning placeholder")
            head += [placeholder] * reason_tokens
        position = len(head) + tail_answer
        tokens = head + tails[0] + [self.terminator_id]
        content_tokens = len(tokens)
        if content_tokens > self.canvas_length:
            raise ValueError(
                f"Question {name!r} requires {content_tokens} canvas tokens; "
                f"the served canvas holds {self.canvas_length}. "
                "Nothing was truncated."
            )
        physical = min(self.canvas_length, ((content_tokens + 15) // 16) * 16)
        tokens += [self.tokenizer.pad_token_id] * (physical - content_tokens)
        allowed = self.thought_tokens() if reason_tokens else ()
        return Template(
            tokens=tuple(tokens),
            answer_position=position,
            candidate_ids=candidates,
            codes=codes,
            question_json=metadata,
            content_tokens=content_tokens,
            trace_canvas_text=(
                tuple(self.tokenizer.decode([token]) for token in tokens)
                if self.trace is not None
                else None
            ),
            reasoning_positions=reasoning_positions,
            reasoning_token_ids=allowed,
            reasoning_forbidden_token_ids=(
                self._thought_forbidden_token_ids if reason_tokens else ()
            ),
            canvas_format="chat",
        )

    def _single_user_template(self, template: Template) -> Template:
        opening = self.tokenizer.encode("<|turn>user\n", add_special_tokens=False)
        separator = self.tokenizer.encode("\n\n", add_special_tokens=False)
        if (
            len(opening) != 3
            or len(separator) != 1
            or template.tokens[:3] != tuple(opening)
            or self.tokenizer.decode(separator) != "\n\n"
        ):
            raise ValueError("Tokenizer changed the single-user chat boundary")
        tokens = (
            separator[0],
            *template.tokens[3:],
            self.tokenizer.pad_token_id,
            self.tokenizer.pad_token_id,
        )
        return replace(
            template,
            tokens=tokens,
            answer_position=template.answer_position - 2,
            reasoning_positions=tuple(p - 2 for p in template.reasoning_positions),
            content_tokens=template.content_tokens - 2,
            trace_canvas_text=(
                tuple(self.tokenizer.decode([token]) for token in tokens)
                if self.trace is not None
                else None
            ),
            canvas_format="chat_single_user",
        )

    def thought_tokens(self) -> tuple[int, ...]:
        if self._thought_token_ids is None:
            vocabulary = (
                self._tokenizer_info.decoded_vocab
                if self._tokenizer_info is not None
                else [
                    self.tokenizer.decode([token]).encode()
                    for token in range(self.vocab_size)
                ]
            )
            special = set(self.tokenizer.all_special_ids) | {
                self.tokenizer.pad_token_id
            }
            candidates = []
            for token, encoded in enumerate(vocabulary):
                try:
                    text = encoded.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if (
                    token not in special
                    and text
                    and "\ufffd" not in text
                    and all(char.isprintable() or char in "\n\t" for char in text)
                ):
                    candidates.append(token)
            allowed = set(candidates)
            if not set(self.reasoning_tokens()).issubset(allowed):
                raise ValueError("Thought vocabulary must permit the initial reasoning")
            self._thought_token_ids = tuple(candidates)
            self._thought_forbidden_token_ids = tuple(
                token for token in range(self.vocab_size) if token not in allowed
            )
        return self._thought_token_ids

    def reasoning_tokens(self) -> tuple[int, ...]:
        if self._reasoning_token_ids is None:
            if self._tokenizer_info is None:
                vocabulary = [
                    self.tokenizer.decode([token]).encode()
                    for token in range(self.vocab_size)
                ]
            else:
                vocabulary = self._tokenizer_info.decoded_vocab
            special = set(self.tokenizer.all_special_ids) | {
                self.tokenizer.pad_token_id
            }
            candidates = []
            for token, encoded in enumerate(vocabulary):
                try:
                    text = encoded.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if (
                    token not in special
                    and text
                    and text.isprintable()
                    and '"' not in text
                    and "\\" not in text
                    and "\ufffd" not in text
                ):
                    candidates.append(token)
            if not candidates:
                raise ValueError("The tokenizer has no safe reasoning tokens")
            self._reasoning_token_ids = tuple(candidates)
            allowed = set(candidates)
            self._reasoning_forbidden_token_ids = tuple(
                token for token in range(self.vocab_size) if token not in allowed
            )
        return self._reasoning_token_ids

    def _validate_reasoning_grammar(
        self,
        tokens: list[int],
        position: int,
        candidates: tuple[int, ...],
        metadata: str,
        codes: tuple[str, ...],
        reasoning_positions: tuple[int, ...],
    ) -> None:
        alternatives = " | ".join(json.dumps(json.dumps(code)) for code in codes)
        prefix = metadata[:-1] + ',"reasoning":"'
        suffix = '","answer":'
        grammar = (
            f"root ::= {json.dumps(prefix)} safe* {json.dumps(suffix)} "
            f"({alternatives}) {json.dumps('}')}\n"
            r'safe ::= [^"\\\x00-\x1f]'
        )
        compiled = self.grammar_compiler.compile_grammar(grammar)
        matcher = xgr.GrammarMatcher(
            compiled, override_stop_tokens=[self.terminator_id]
        )
        start = reasoning_positions[0]
        if not all(matcher.accept_token(token) for token in tokens[:start]):
            raise ValueError("XGrammar rejected the fixed reasoning prefix")
        bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        matcher.fill_next_token_bitmask(bitmask)
        masks = bitmask[0].tolist()
        for token in self.reasoning_tokens():
            if not masks[token // 32] & (1 << (token % 32)):
                raise ValueError("XGrammar rejected a reasoning token")
        if not all(matcher.accept_token(token) for token in tokens[start:position]):
            raise ValueError("XGrammar rejected the fixed reasoning suffix")
        for token in candidates:
            completion = matcher.fork()
            if (
                not all(
                    completion.accept_token(candidate)
                    for candidate in [
                        token,
                        *tokens[position + 1 :],
                        self.terminator_id,
                    ]
                )
                or not completion.is_terminated()
            ):
                raise ValueError("XGrammar rejected an answer after reasoning")

    def _validate_grammar(
        self,
        tokens: list[int],
        position: int,
        candidates: tuple[int, ...],
        prefix_text: str,
        codes: tuple[str, ...],
    ) -> None:
        alternatives = " | ".join(json.dumps(json.dumps(code)) for code in codes)
        grammar = (
            f"root ::= {json.dumps(prefix_text)} ({alternatives}) {json.dumps('}')}"
        )
        compiled = self.grammar_compiler.compile_grammar(grammar)
        prefix = xgr.GrammarMatcher(compiled, override_stop_tokens=[self.terminator_id])
        if not all(prefix.accept_token(token) for token in tokens[:position]):
            raise ValueError("XGrammar rejected the fixed answer prefix")
        bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        prefix.fill_next_token_bitmask(bitmask)
        for candidate in candidates:
            if not int(bitmask[0, candidate // 32]) & (1 << (candidate % 32)):
                raise ValueError("XGrammar rejected an answer candidate")
            matcher = prefix.fork()
            completion = [candidate, *tokens[position + 1 :], self.terminator_id]
            if (
                not all(matcher.accept_token(token) for token in completion)
                or not matcher.is_terminated()
            ):
                raise ValueError(
                    "XGrammar rejected an answer candidate with the fixed suffix"
                )

    @staticmethod
    def prompt(state: JSONContent) -> str:
        return "Shared state:\n" + canonical_json(state)

    def shared_context(self, request: SystemOneRequest) -> SharedContext:
        image_blocks, metadata = prepare_images(request.images)
        content: Any = self.prompt(request.state)
        if image_blocks:
            content = [{"type": "text", "text": content}, *image_blocks]
        legacy_prompt = (
            REASON_INSTRUCTIONS if request.reason_mode else COMMON_INSTRUCTIONS
        )
        system_prompt = (
            legacy_prompt if request.system_prompt is None else request.system_prompt
        )
        seed_prompt_hash = None
        if request.canvas_format != "json":
            seed_messages = [
                {"role": "system", "content": legacy_prompt},
                {"role": "user", "content": content},
            ]
            seed_prompt_hash = hashlib.sha256(
                canonical_json([seed_messages, {"enable_thinking": False}]).encode()
            ).hexdigest()
            system_prompt = (
                (THOUGHT_INSTRUCTIONS if request.reason_mode else CHAT_INSTRUCTIONS)
                if request.system_prompt is None
                else request.system_prompt
            )
            if request.reason_mode:
                system_prompt = "<|think|>\n" + system_prompt
        if request.canvas_format == "chat_single_user" and image_blocks:
            # Continuation stops at the last text block, so keep it after images.
            content = [*content, {"type": "text", "text": ""}]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        prompt_options = {"enable_thinking": False}
        if request.canvas_format == "chat_single_user":
            prompt_options["continue_final_message"] = True
        prompt_hash = hashlib.sha256(
            canonical_json([messages, prompt_options]).encode()
        ).hexdigest()
        return SharedContext(messages, prompt_hash, metadata, seed_prompt_hash)

    def prepare_read(
        self,
        request: SystemOneRequest,
        question_id: str,
        question: QuestionBase,
        sample_index: int,
        context: SharedContext,
    ) -> PreparedRead:
        template = self.template(
            question_id,
            question,
            request.reason_tokens if request.reason_mode else 0,
            "chat"
            if request.canvas_format == "chat_single_user"
            else request.canvas_format,
        )
        seed_material = canonical_json(
            [
                context.seed_prompt_hash or context.prompt_hash,
                template.question_json,
                request.seed,
                sample_index,
            ]
        ).encode()
        noise = random.Random(hashlib.sha256(seed_material).digest())
        canvas = list(template.tokens)
        canvas[template.answer_position] = (
            noise.choice(template.candidate_ids)
            if request.reason_mode
            and (request.reason_initial_answer_noise or request.reason_answer_noise)
            == "candidates"
            else noise.randrange(self.vocab_size)
        )
        if template.reasoning_positions:
            reason_noise = random.Random(
                hashlib.sha256(seed_material + b"reasoning").digest()
            )
            for position in template.reasoning_positions:
                canvas[position] = reason_noise.choice(self.reasoning_tokens())
        reasoning_seed = None
        if request.canvas_format == "chat_single_user":
            if request.reason_mode:
                reasoning_seed = int.from_bytes(
                    hashlib.sha256(json.dumps(canvas).encode()).digest()[:8], "little"
                )
            template = self.template(
                question_id,
                question,
                request.reason_tokens if request.reason_mode else 0,
                request.canvas_format,
            )
            canvas = [template.tokens[0], *canvas[3:], *template.tokens[-2:]]
        return PreparedRead(
            question_id,
            question,
            sample_index,
            template,
            tuple(canvas),
            request.steps,
            request.self_conditioning,
            request.self_conditioning_scope,
            request.reason_steps if request.reason_mode else 0,
            request.reason_answer_noise if request.reason_mode else "candidates",
            request.reason_schedule_steps if request.reason_mode else None,
            request.reason_entropy_bound if request.reason_mode else None,
            reasoning_seed,
        )

    def decode_choice(
        self, read: PreparedRead, choice: dict[str, Any]
    ) -> dict[str, Any]:
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise ValueError(
                "The upstream response contains an unfinished or failed child"
            )
        content = choice["logprobs"]["content"]
        if not isinstance(content, list) or len(content) not in {
            1,
            len(read.canvas),
        }:
            raise ValueError("The upstream response has an invalid decision canvas")

        def token_id(token: Any) -> int:
            if not isinstance(token, str) or not token.startswith("token_id:"):
                raise ValueError("The upstream response contains an invalid token ID")
            value = token.removeprefix("token_id:")
            if not value.isascii() or not value.isdecimal():
                raise ValueError("The upstream response contains an invalid token ID")
            result = int(value)
            if not 0 <= result < self.vocab_size:
                raise ValueError(
                    "The upstream response contains an out-of-vocabulary ID"
                )
            return result

        def logprob(value: Any) -> float:
            try:
                score = float(value) if type(value) in (int, float) else math.nan
            except OverflowError:
                score = math.nan
            if not math.isfinite(score) or score > 0:
                raise ValueError(
                    "The model must return a finite logprob for every candidate"
                )
            return score

        template = read.template
        reasoning = bool(template.reasoning_positions)
        full_canvas = reasoning or template.canvas_format != "json"
        if full_canvas:
            output_tokens = choice.get("token_ids")
            if (
                not isinstance(output_tokens, list)
                or len(output_tokens) != len(read.canvas)
                or any(
                    type(token) is not int or not 0 <= token < self.vocab_size
                    for token in output_tokens
                )
            ):
                raise ValueError(
                    "The upstream response must return the complete canvas"
                )
            mutable = {template.answer_position, *template.reasoning_positions}
            forbidden = set(template.reasoning_forbidden_token_ids)
            for index, (expected, actual) in enumerate(zip(read.canvas, output_tokens)):
                if index not in mutable and expected != actual:
                    raise ValueError(
                        "The upstream response changed a fixed canvas token"
                    )
                if index in template.reasoning_positions and actual in forbidden:
                    raise ValueError(
                        "The upstream response selected an illegal reasoning token"
                    )
        if len(content) == 1:
            row = content[0]
            selected = token_id(row["token"])
            if full_canvas:
                if selected != output_tokens[template.answer_position]:
                    raise ValueError(
                        "The upstream answer disagrees with the returned canvas"
                    )
            elif self.trace is not None:
                output_tokens = list(read.canvas)
                output_tokens[template.answer_position] = selected
        else:
            content_tokens = [token_id(row["token"]) for row in content]
            if full_canvas:
                if content_tokens != output_tokens:
                    raise ValueError(
                        "The upstream token logprobs disagree with its canvas"
                    )
            else:
                output_tokens = content_tokens
                for index, (expected, actual) in enumerate(
                    zip(read.canvas, output_tokens)
                ):
                    if index != template.answer_position and expected != actual:
                        raise ValueError(
                            "The upstream response changed a fixed canvas token"
                        )
            selected = output_tokens[template.answer_position]
            row = content[template.answer_position]
        if selected not in template.candidate_ids:
            raise ValueError("The upstream response selected an illegal answer token")
        entries = row["top_logprobs"]
        if not isinstance(entries, list):
            raise ValueError("The upstream response has malformed candidate logprobs")
        logprobs: dict[int, float] = {}
        for item in entries:
            candidate = token_id(item["token"])
            if candidate not in template.candidate_ids or candidate in logprobs:
                raise ValueError("The upstream response has unexpected candidate IDs")
            logprobs[candidate] = logprob(item["logprob"])
        selected_logprob = logprob(row["logprob"])
        if selected in logprobs and logprobs[selected] != selected_logprob:
            raise ValueError("The upstream response has inconsistent selected logprobs")
        logprobs[selected] = selected_logprob
        if set(logprobs) != set(template.candidate_ids):
            raise ValueError(
                "The model must return a finite logprob for every candidate"
            )
        values = [logprobs[candidate] for candidate in template.candidate_ids]
        probabilities = normalized_probabilities(values)
        record = {
            "question_id": read.question_id,
            "probabilities": probabilities,
        }
        if reasoning:
            reason_text = self.tokenizer.decode(
                [output_tokens[index] for index in template.reasoning_positions]
            )
            if template.canvas_format == "json":
                document = json.loads(
                    self.tokenizer.decode(output_tokens[: template.content_tokens - 1])
                )
                expected = {
                    **template.question_json,
                    "reasoning": reason_text,
                    "answer": template.codes[template.candidate_ids.index(selected)],
                }
                if document != expected:
                    raise ValueError(
                        "The upstream reasoning changed the JSON decision form"
                    )
            record["reasoning"] = reason_text
        if self.trace is None:
            return record

        peak = max(values)
        label_mass = math.exp(peak) * sum(math.exp(value - peak) for value in values)
        assert template.trace_canvas_text is not None
        canvas_text = list(template.trace_canvas_text)
        canvas_text[template.answer_position] = self.tokenizer.decode(
            [read.canvas[template.answer_position]]
        )
        for position in template.reasoning_positions:
            canvas_text[position] = self.tokenizer.decode([read.canvas[position]])
        record.update(
            {
                "kind": read.question.type,
                "sample": read.sample_index,
                "question": read.question.model_dump(exclude_none=True),
                "question_placement": "canvas",
                (
                    "fixed_question_json"
                    if template.canvas_format == "json"
                    else "seed_question_json"
                ): template.question_json,
                "question_canvas_tokens": template.content_tokens,
                "canvas_tokens": read.canvas,
                "canvas_text": canvas_text,
                "answer_position": template.answer_position,
                "candidate_labels": labels_for(read.question),
                "candidate_codes": template.codes,
                "candidate_token_ids": template.candidate_ids,
                "output_tokens": output_tokens,
                "candidate_logprobs": values,
                "label_mass": label_mass,
            }
        )
        if reasoning:
            record.update(
                {
                    "reasoning_positions": template.reasoning_positions,
                    "reasoning_vocab_size": self.vocab_size,
                    "reasoning_forbidden_token_ids": (
                        template.reasoning_forbidden_token_ids
                    ),
                    "reason_tokens": len(template.reasoning_positions),
                    "reason_steps": read.reasoning_steps,
                    "reason_answer_noise": read.reasoning_answer_noise,
                    "output_text": [
                        self.tokenizer.decode([token]) for token in output_tokens
                    ],
                }
            )
            if read.reasoning_schedule_steps is not None:
                record["reason_schedule_steps"] = read.reasoning_schedule_steps
            if read.reasoning_entropy_bound is not None:
                record["reason_entropy_bound"] = read.reasoning_entropy_bound
            if read.reasoning_seed is not None:
                record["reasoning_seed"] = read.reasoning_seed
        if template.canvas_format != "json":
            record.update(
                canvas_format=template.canvas_format,
                output_text=[self.tokenizer.decode([token]) for token in output_tokens],
            )
        return record

    async def _read_round(
        self,
        request: SystemOneRequest,
        request_id: str,
        context: SharedContext,
        reads: list[PreparedRead],
        round_index: int,
    ) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
        child_count = len(reads)
        body = {
            "model": self.model,
            "messages": context.messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "n": child_count,
            "diffusion_reads": [read.descriptor() for read in reads],
            "max_tokens": self.canvas_length,
            "ignore_eos": True,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logprobs": True,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        }
        if request.cache_salt is not None:
            body["cache_salt"] = request.cache_salt
        if request.reason_mode:
            body["return_token_ids"] = True
        if request.canvas_format != "json":
            body.update(
                add_generation_prompt=False,
                max_tokens=max(len(read.canvas) for read in reads),
                return_token_ids=True,
            )
        if request.canvas_format == "chat_single_user":
            body["continue_final_message"] = True
        parent: dict[str, Any] = {
            "event": "parent",
            "request_id": request_id,
            "parent_id": None,
            "child_count": child_count,
            "children": [
                {
                    "choice_index": index,
                    "question_id": read.question_id,
                    "sample": read.sample_index,
                    **(
                        {
                            "reason_tokens": len(read.template.reasoning_positions),
                            "reason_steps": read.reasoning_steps,
                        }
                        if request.reason_ensemble is not None
                        else {}
                    ),
                }
                for index, read in enumerate(reads)
            ],
        }
        round_fields = {"round": round_index} if request.adaptive_samples else {}
        parent.update(round_fields)
        upstream_started = time.perf_counter()
        try:
            async with self.semaphore:
                response = await self.client.post("v1/chat/completions", json=body)
                response.raise_for_status()
                upstream = response.json()
            if not isinstance(upstream, dict):
                raise ValueError("The upstream response must be an object")
            parent_id = upstream.get("id")
            if (
                not isinstance(parent_id, str)
                or not 1 <= len(parent_id) <= 256
                or not all(
                    c.isascii() and (c.isalnum() or c in "-_") for c in parent_id
                )
            ):
                raise ValueError("The upstream response has an invalid parent ID")
            parent["parent_id"] = parent_id
            choices = upstream.get("choices")
            if not isinstance(choices, list) or len(choices) != child_count:
                raise ValueError(
                    "The upstream response must contain exactly one choice per child"
                )
            indexed: dict[int, dict[str, Any]] = {}
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError(
                        "The upstream response contains a malformed choice"
                    )
                index = choice.get("index")
                if (
                    type(index) is not int
                    or not 0 <= index < child_count
                    or index in indexed
                ):
                    raise ValueError(
                        "The upstream response must contain unique in-range "
                        "choice indices"
                    )
                indexed[index] = choice
            records = [
                self.decode_choice(read, indexed[index])
                for index, read in enumerate(reads)
            ]
            counts = upstream.get("usage")
            if not isinstance(counts, dict) or any(
                type(counts.get(key)) is not int or counts[key] < 0
                for key in ("prompt_tokens", "completion_tokens")
            ):
                raise ValueError(
                    "The upstream response must report valid parent token usage"
                )
            expected_completion_tokens = sum(len(read.canvas) for read in reads)
            if counts["completion_tokens"] != expected_completion_tokens:
                raise ValueError(
                    "The upstream completion usage does not match the complete canvases"
                )
            parent_usage = {
                key: counts[key] for key in ("prompt_tokens", "completion_tokens")
            }
            if "total_tokens" in counts:
                if type(counts["total_tokens"]) is not int or counts[
                    "total_tokens"
                ] != sum(parent_usage.values()):
                    raise ValueError(
                        "The upstream response has inconsistent total token usage"
                    )
                parent_usage["total_tokens"] = counts["total_tokens"]
        except BaseException as error:
            self.record(
                {
                    **parent,
                    "status": "error",
                    "error": type(error).__name__,
                    "latency_ms": (time.perf_counter() - upstream_started) * 1000,
                }
            )
            raise
        latency_ms = (time.perf_counter() - upstream_started) * 1000
        if self.trace is not None:
            self.record(
                {
                    **parent,
                    "status": "completed",
                    "upstream_usage": parent_usage,
                    "latency_ms": latency_ms,
                }
            )
        for index, record in enumerate(records):
            if self.trace is not None:
                self.record(
                    {
                        "event": "sample",
                        "request_id": request_id,
                        "parent_id": parent_id,
                        "choice_index": index,
                        "child_count": child_count,
                        "state": request.state,
                        "shared_prompt_hash": context.prompt_hash,
                        "image_count": len(context.images),
                        "shared_images": context.images,
                        "shared_prompt_tokens": parent_usage["prompt_tokens"],
                        "cached_prompt_tokens": None,
                        "created_cache_tokens": None,
                        "latency_ms": latency_ms,
                        "latency_scope": "parent",
                        "reasoning": request.reason_mode,
                        "denoise_steps": reads[index].steps
                        + reads[index].reasoning_steps,
                        "self_conditioning": request.self_conditioning,
                        "self_conditioning_scope": request.self_conditioning_scope,
                        "constraint_backend": (
                            "fixed-token-candidates"
                            if request.canvas_format != "json"
                            else "xgrammar"
                        ),
                        "constraint_version": self.constraint_version,
                        **round_fields,
                        **(
                            {"seed_prompt_hash": context.seed_prompt_hash}
                            if context.seed_prompt_hash is not None
                            else {}
                        ),
                        **(
                            {
                                "reason_initial_answer_noise": (
                                    request.reason_initial_answer_noise
                                )
                            }
                            if request.reason_initial_answer_noise is not None
                            else {}
                        ),
                        **record,
                    }
                )
        return parent_id, parent_usage, records

    async def system_one(
        self, request: SystemOneRequest, request_id: str
    ) -> dict[str, Any]:
        if request.model not in (None, self.model, "jev-latest"):
            raise HTTPException(404, f"Unknown model: {request.model}")
        started = time.perf_counter()
        try:
            budget_requests = (
                [request]
                if request.reason_ensemble is None
                else [
                    request.model_copy(
                        update={
                            "reason_tokens": budget.tokens,
                            "reason_steps": budget.steps,
                        }
                    )
                    for budget in request.reason_ensemble
                ]
            )
            for name, question in request.questions.items():
                for budget_request in budget_requests:
                    self.template(
                        name,
                        question,
                        budget_request.reason_tokens if request.reason_mode else 0,
                        request.canvas_format,
                    )
            context = (
                await asyncio.to_thread(self.shared_context, request)
                if request.images
                else self.shared_context(request)
            )
            reads = [
                self.prepare_read(budget_request, name, question, index, context)
                for name, question in request.questions.items()
                for budget_request in budget_requests
                for index in range(request.samples)
            ]
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        distributions: dict[str, list[list[float]]] = {
            name: [] for name in request.questions
        }
        reasons: dict[str, list[str]] = {name: [] for name in request.questions}
        usage = {"input_tokens": 0, "output_tokens": 0}
        parent_ids = []
        child_count = 0
        for round_index in range(2 if request.adaptive_samples else 1):
            if not reads:
                break
            parent_id, parent_usage, records = await self._read_round(
                request, request_id, context, reads, round_index
            )
            parent_ids.append(parent_id)
            child_count += len(reads)
            usage["input_tokens"] += len(reads) * parent_usage["prompt_tokens"]
            usage["output_tokens"] += parent_usage["completion_tokens"]
            for record in records:
                distributions[record["question_id"]].append(record["probabilities"])
                if request.reason_mode:
                    reasons[record["question_id"]].append(record["reasoning"])
            if request.adaptive_samples and round_index == 0:
                reads = []
                for name, question in request.questions.items():
                    draws = distributions[name]
                    mean = [sum(column) / len(draws) for column in zip(*draws)]
                    if 1.0 - confidence(mean) > request.adaptive_threshold:
                        reads.extend(
                            self.prepare_read(request, name, question, index, context)
                            for index in range(request.samples, request.max_samples)
                        )
        answers = {}
        question_diagnostics = {}
        for name, question in request.questions.items():
            draws = distributions[name]
            mean = [sum(column) / len(draws) for column in zip(*draws)]
            answers[name] = answer_for(question, mean)
            if request.reason_mode:
                answers[name]["reasoning"] = reasons[name]
            if request.return_diagnostics:
                count = len(draws)
                winner = max(range(len(mean)), key=mean.__getitem__)
                errors = [
                    math.sqrt(
                        sum((value - average) ** 2 for value in column)
                        / (count * (count - 1))
                    )
                    if count > 1
                    else None
                    for average, column in zip(mean, zip(*draws))
                ]
                question_diagnostics[name] = {
                    "samples": count,
                    "standard_errors": dict(zip(labels_for(question), errors)),
                    "argmax_agreement": sum(
                        max(range(len(draw)), key=draw.__getitem__) == winner
                        for draw in draws
                    )
                    / count,
                }
        result = {"model": self.model, "usage": usage, "answers": answers}
        if request.return_diagnostics:
            result["diagnostics"] = {
                "samples": request.samples,
                "steps": request.steps,
                "self_conditioning": request.self_conditioning,
                "self_conditioning_scope": request.self_conditioning_scope,
                "seed": request.seed,
                "questions": question_diagnostics,
            }
            if request.adaptive_samples:
                result["diagnostics"].update(
                    {
                        "adaptive_samples": True,
                        "max_samples": request.max_samples,
                        "adaptive_threshold": request.adaptive_threshold,
                    }
                )
            if request.reason_mode:
                result["diagnostics"].update(
                    {
                        "reason_mode": True,
                        **(
                            {
                                "reason_tokens": request.reason_tokens,
                                "reason_steps": request.reason_steps,
                            }
                            if request.reason_ensemble is None
                            else {
                                "reason_ensemble": [
                                    budget.model_dump()
                                    for budget in request.reason_ensemble
                                ]
                            }
                        ),
                        "reason_answer_noise": request.reason_answer_noise,
                    }
                )
                if request.reason_schedule_steps is not None:
                    result["diagnostics"]["reason_schedule_steps"] = (
                        request.reason_schedule_steps
                    )
                if request.reason_entropy_bound is not None:
                    result["diagnostics"]["reason_entropy_bound"] = (
                        request.reason_entropy_bound
                    )
            if request.canvas_format != "json":
                result["diagnostics"]["canvas_format"] = request.canvas_format
            if request.reason_initial_answer_noise is not None:
                result["diagnostics"]["reason_initial_answer_noise"] = (
                    request.reason_initial_answer_noise
                )
        if self.trace is not None:
            self.record(
                {
                    "event": "response",
                    "request_id": request_id,
                    "parent_id": parent_ids[0],
                    "child_count": child_count,
                    **(
                        {"parent_ids": parent_ids, "rounds": len(parent_ids)}
                        if request.adaptive_samples
                        else {}
                    ),
                    **result,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        return result


def create_app(reader: DecisionReader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await reader.close()

    app = FastAPI(title="Diffusion Gemma TypeSafe example", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return validation_error_response(error)

    @app.get("/v1/models")
    async def models(response: Response):
        response.headers["x-typesafe-request-id"] = f"ts-{uuid.uuid4().hex}"
        return {
            "models": [
                {
                    "name": reader.model,
                    "description": "Diffusion Gemma single-pass candidate scores",
                    "release_date": "2026-06-10",
                }
            ]
        }

    @app.post("/v1/systemone")
    async def system_one(request: SystemOneRequest, response: Response):
        request_id = f"ts-{uuid.uuid4().hex}"
        response.headers["x-typesafe-request-id"] = request_id
        response.headers["x-typesafe-constraint-backend"] = (
            "fixed-token-candidates" if request.canvas_format != "json" else "xgrammar"
        )
        try:
            return await reader.system_one(request, request_id)
        except httpx.HTTPStatusError as error:
            detail = (
                f"Upstream rejected the image read (HTTP {error.response.status_code})"
                if request.images
                else f"Upstream rejected the read: {error.response.text}"
            )
            raise HTTPException(502, detail) from error
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
            detail = (
                f"{type(error).__name__} while reading image-conditioned output"
                if request.images
                else str(error)
            )
            reader.record({"event": "error", "request_id": request_id, "error": detail})
            raise HTTPException(
                502, f"Invalid upstream decision read: {detail}"
            ) from error

    return app


def main() -> None:
    import uvicorn
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default="diffusion-gemma")
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument(
        "--trim-canvas-padding",
        action="store_true",
        help="Send only each decision form's meaningful canvas prefix.",
    )
    parser.add_argument(
        "--minimum-canvas-padding",
        type=int,
        default=0,
        help="Trailing pad tokens retained when canvas padding is trimmed.",
    )
    parser.add_argument(
        "--full-canvas-type",
        action="append",
        choices=("noul", "choice", "score"),
        default=[],
        help="Question type to keep at the full served canvas length (repeatable).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Maximum concurrent upstream parent requests (each contains N reads).",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--upstream-api-key")
    args = parser.parse_args()
    if args.canvas < 1 or args.concurrency < 1 or args.minimum_canvas_padding < 0:
        parser.error(
            "canvas and concurrency must be positive; padding cannot be negative"
        )
    reader = DecisionReader(
        AutoTokenizer.from_pretrained(args.tokenizer),
        args.upstream,
        args.model,
        args.canvas,
        args.concurrency,
        args.timeout,
        args.trace,
        args.upstream_api_key,
        trim_canvas_padding=args.trim_canvas_padding,
        minimum_canvas_padding=args.minimum_canvas_padding,
        full_canvas_types=frozenset(args.full_canvas_type),
    )
    uvicorn.run(create_app(reader), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
