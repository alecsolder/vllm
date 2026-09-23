# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in in-process TypeSafe endpoint for the vLLM API server."""

import os
import uuid
from argparse import Namespace
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import State

from examples.features.diffusion_typesafe.server import (
    DecisionReader,
    InvalidImage,
    SystemOneRequest,
    validation_error_response,
)
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)

_RAW_REQUEST: ContextVar[Request | None] = ContextVar(
    "diffusion_typesafe_raw_request", default=None
)


async def _cache_request_body(raw_request: Request) -> None:
    await raw_request.body()


class _DirectResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _InProcessChatClient:
    def __init__(self, handler: OpenAIServingChat):
        self.handler = handler

    async def post(self, path: str, *, json: dict[str, Any]) -> _DirectResponse:
        if path != "v1/chat/completions":
            raise ValueError(f"Unsupported in-process path: {path}")
        request = ChatCompletionRequest.model_validate(json)
        response = await self.handler.create_chat_completion(
            request, _RAW_REQUEST.get()
        )
        if isinstance(response, ErrorResponse):
            raw = httpx.Response(
                response.error.code,
                json=response.model_dump(),
                request=httpx.Request("POST", "http://vllm/v1/chat/completions"),
            )
            raw.raise_for_status()
        if not isinstance(response, ChatCompletionResponse):
            raise ValueError("The in-process endpoint does not support streaming")
        return _DirectResponse(response.model_dump())

    async def aclose(self) -> None:
        return None


class DiffusionTypeSafeEndpointPlugin:
    """Serve TypeSafe decisions without a second HTTP server."""

    name = "diffusion_typesafe"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/models")
        async def models(raw_request: Request):
            handler = raw_request.app.state.openai_serving_models
            response = await handler.show_available_models()
            payload = response.model_dump()
            payload["models"] = [
                {
                    "name": model["id"],
                    "description": "Diffusion Gemma single-pass candidate scores",
                    "release_date": "2026-06-10",
                }
                for model in payload["data"]
            ]
            return JSONResponse(
                payload,
                headers={"x-typesafe-request-id": f"ts-{uuid.uuid4().hex}"},
            )

        # Endpoint plugins are attached after core routes. Put the combined response
        # before the core OpenAI-only route so both wire contracts share this path.
        models_route = app.router.routes.pop()
        core_models_index = next(
            (
                index
                for index, route in enumerate(app.router.routes)
                if getattr(route, "path", None) == "/v1/models"
            ),
            len(app.router.routes),
        )
        app.router.routes.insert(core_models_index, models_route)

        @app.post(
            "/v1/systemone",
            dependencies=[Depends(_cache_request_body)],
        )
        @with_cancellation
        @load_aware_call
        async def system_one(raw_request: Request):
            request_id = f"ts-{uuid.uuid4().hex}"
            headers = {
                "x-typesafe-request-id": request_id,
                "x-typesafe-constraint-backend": "xgrammar",
            }
            content_type = raw_request.headers.get("content-type", "").lower()
            if content_type.split(";", maxsplit=1)[0] != "application/json":
                return JSONResponse(
                    {
                        "detail": [
                            {
                                "type": "json_type",
                                "loc": ["body"],
                                "msg": (
                                    "Invalid request field; see error type and location"
                                ),
                            }
                        ]
                    },
                    status_code=422,
                )
            try:
                request = SystemOneRequest.model_validate_json(await raw_request.body())
            except ValidationError as error:
                return validation_error_response(error, prepend_body=True)
            chat_raw_request = raw_request
            if request.adaptive_samples and "x-request-id" in raw_request.headers:
                scope = {
                    **raw_request.scope,
                    "headers": [
                        (name, value)
                        for name, value in raw_request.scope["headers"]
                        if name.lower() != b"x-request-id"
                    ],
                }
                chat_raw_request = Request(scope, receive=raw_request.receive)
            context_token = _RAW_REQUEST.set(chat_raw_request)
            try:
                result = (
                    await raw_request.app.state.diffusion_typesafe_reader.system_one(
                        request, request_id
                    )
                )
                return JSONResponse(result, headers=headers)
            except InvalidImage as error:
                return JSONResponse(
                    {"detail": str(error)}, status_code=422, headers=headers
                )
            except httpx.HTTPStatusError as error:
                detail = (
                    "Upstream rejected the image read "
                    f"(HTTP {error.response.status_code})"
                    if request.images
                    else f"Upstream rejected the read: {error.response.text}"
                )
                return JSONResponse(
                    {"detail": detail}, status_code=502, headers=headers
                )
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
                detail = (
                    f"{type(error).__name__} while reading image-conditioned output"
                    if request.images
                    else str(error)
                )
                raw_request.app.state.diffusion_typesafe_reader.record(
                    {"event": "error", "request_id": request_id, "error": detail}
                )
                return JSONResponse(
                    {"detail": f"Invalid upstream decision read: {detail}"},
                    status_code=502,
                    headers=headers,
                )
            finally:
                _RAW_REQUEST.reset(context_token)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        if engine_client is None:
            state.diffusion_typesafe_reader = None
            return
        handler: OpenAIServingChat | None = state.openai_serving_chat
        if handler is None:
            raise RuntimeError("TypeSafe requires the chat-completions handler")
        tokenizer = state.online_renderer.renderer.tokenizer
        if tokenizer is None:
            raise RuntimeError("TypeSafe requires a tokenizer")
        diffusion_config = state.vllm_config.diffusion_config
        if diffusion_config is None or diffusion_config.canvas_length is None:
            raise RuntimeError("TypeSafe requires a configured diffusion canvas")
        served_names = args.served_model_name or [args.model]
        trace_value = os.environ.get("VLLM_TYPESAFE_TRACE")
        trim_padding = os.environ.get("VLLM_TYPESAFE_TRIM_CANVAS_PADDING", "0")
        if trim_padding not in {"0", "1"}:
            raise ValueError("VLLM_TYPESAFE_TRIM_CANVAS_PADDING must be 0 or 1")
        minimum_padding = int(
            os.environ.get("VLLM_TYPESAFE_MINIMUM_CANVAS_PADDING", "0")
        )
        if minimum_padding < 0:
            raise ValueError("VLLM_TYPESAFE_MINIMUM_CANVAS_PADDING cannot be negative")
        full_canvas_types = frozenset(
            value.strip()
            for value in os.environ.get("VLLM_TYPESAFE_FULL_CANVAS_TYPES", "").split(
                ","
            )
            if value.strip()
        )
        if not full_canvas_types <= {"noul", "choice", "score"}:
            raise ValueError("VLLM_TYPESAFE_FULL_CANVAS_TYPES has an unknown type")
        reader = DecisionReader(
            tokenizer,
            "http://unused",
            served_names[0],
            canvas_length=diffusion_config.canvas_length,
            trace=Path(trace_value) if trace_value else None,
            trim_canvas_padding=trim_padding == "1",
            minimum_canvas_padding=minimum_padding,
            full_canvas_types=full_canvas_types,
        )
        await reader.client.aclose()
        reader.client = _InProcessChatClient(handler)  # type: ignore[assignment]
        state.diffusion_typesafe_reader = reader


def install_plugin() -> None:
    """Inject the opt-in endpoint plugin before the regular vLLM CLI starts."""
    import vllm.plugins

    original_loader = vllm.plugins.load_endpoint_plugins

    def load_endpoint_plugins(supported_tasks=None):
        plugins = original_loader(supported_tasks)
        if supported_tasks is None or "generate" not in supported_tasks:
            return plugins
        return [*plugins, DiffusionTypeSafeEndpointPlugin()]

    vllm.plugins.load_endpoint_plugins = load_endpoint_plugins


def main() -> None:
    install_plugin()
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
