# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from copy import deepcopy
from typing import Annotated, Any, ClassVar, Literal

from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    StrictBool,
    StrictInt,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.generate.base.protocol import (
    AnyResponseFormat,
    DeltaMessage,
    FunctionCall,
    FunctionDefinition,
    PerRequestMetrics,
    StopParam,
    StreamOptions,
    ToolCall,
    structured_outputs_from_response_format,
    validate_cache_salt,
    validate_structural_tag_response_format,
    validate_structured_outputs_structural_tag,
)
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import (
    MAX_LOGPROB_TOKEN_IDS,
    BeamSearchParams,
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
    ThinkingTokenBudget,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    annotations: OpenAIAnnotation | None = None
    audio: OpenAIChatCompletionAudio | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # vLLM-specific fields that are not in OpenAI spec
    reasoning: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if len(data.get("tool_calls", [])) == 0:
            data.pop("tool_calls", None)
        return data


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[set[str] | None] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None = None


class DiffusionAnswerDistribution(OpenAIBaseModel):
    aggregation: Literal["mean"] = "mean"
    normalization: Literal["allowed_token_ids"] = "allowed_token_ids"
    answer_passes: int
    answer_position: int
    token_ids: list[int]
    probabilities: list[float]
    log_probabilities: list[float]


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    # ``SerializeAsAny`` lets pydantic honor subclasses of ``ChatMessage``
    # (e.g. ``vllm.entrypoints.cohere.cohere_chat_message.CohereChatMessage``)
    # so that added fields like ``citations`` survive JSON serialization
    # instead of being stripped down to the base schema. Plain
    # ``ChatMessage`` instances serialize identically to before.
    message: SerializeAsAny[ChatMessage]
    logprobs: ChatCompletionLogProbs | None = None
    diffusion_answer_distribution: DiffusionAnswerDistribution | None = None
    # per OpenAI spec this is the default
    finish_reason: str | None = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: int | str | None = None
    # not part of the OpenAI spec but is useful for tracing the tokens
    # in agent scenarios
    token_ids: list[int] | None = None
    # Per-token expert routing decisions, base64-encoded ``.npy`` bytes
    # (numpy serialization). Shape after decode:
    #   (num_tokens - 1, num_layers, num_experts_per_tok) dtype uint8/uint16/int32
    # ``num_tokens - 1`` because the last sampled token has not been
    # forwarded yet and therefore has no routing data.
    # Decode:
    #   np.load(io.BytesIO(base64.b64decode(s)))
    # ``None`` if (a) the request was aborted before any forward pass,
    # or (b) ``enable_return_routed_experts`` is off server-side.
    routed_experts: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.diffusion_answer_distribution is None:
            data.pop("diffusion_answer_distribution", None)
        return data


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo

    # vLLM-specific fields that are not in OpenAI spec
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    prompt_token_ids: list[int] | None = None
    # Rendered prompt text from chat templating (only set when
    # ``return_prompt_text=True`` on the request).
    prompt_text: str | None = None
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None, description="ECTransfer parameters."
    )
    metrics: PerRequestMetrics | None = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    # ``SerializeAsAny`` lets pydantic honor subclasses of ``DeltaMessage``
    # (e.g. ``vllm.entrypoints.cohere.cohere_chat_message.CohereDeltaMessage``)
    # so streaming ``citations`` survive JSON serialization. Plain
    # ``DeltaMessage`` instances serialize identically to before.
    delta: SerializeAsAny[DeltaMessage]
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    # not part of the OpenAI spec but for tracing the tokens
    token_ids: list[int] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)
    # Set only on the final chunk of a stream to mirror non-streaming responses
    # without the per-chunk serialization overhead.
    system_fingerprint: str | None = None
    # not part of the OpenAI spec but for tracing the tokens
    prompt_token_ids: list[int] | None = None
    # Rendered prompt text from chat templating (only set when
    # ``return_prompt_text=True`` on the request); only sent on the first chunk.
    prompt_text: str | None = None
    metrics: PerRequestMetrics | None = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition
    defer_loading: bool | None = None

    @model_validator(mode="after")
    def _propagate_defer_loading(self) -> "ChatCompletionToolsParam":
        if self.defer_loading is not None and self.function.defer_loading is None:
            self.function.defer_loading = self.defer_loading
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data = {k: v for k, v in data.items() if k in type(self).model_fields}
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class DiffusionReadSpec(BaseModel):
    """One canvas with a constrained answer and an optional reasoning span."""

    model_config = ConfigDict(extra="forbid")

    seed_canvas: list[Annotated[StrictInt, Field(ge=0)]] = Field(min_length=1)
    answer_positions: list[Annotated[StrictInt, Field(ge=0)]] = Field(
        min_length=1, max_length=1
    )
    logprob_token_ids: list[Annotated[StrictInt, Field(ge=0)]] = Field(
        min_length=1, max_length=MAX_LOGPROB_TOKEN_IDS
    )
    answer_only_logprobs: bool = False
    steps: StrictInt = Field(default=1, ge=1, le=32)
    answer_pass_aggregation: Literal["last", "mean"] = "last"
    self_conditioning: StrictBool = True
    self_conditioning_scope: Literal["answer", "canvas"] = "answer"
    reasoning_positions: list[Annotated[StrictInt, Field(ge=0)]] | None = Field(
        default=None, min_length=1
    )
    reasoning_forbidden_token_ids: list[Annotated[StrictInt, Field(ge=0)]] | None = None
    reasoning_steps: StrictInt | None = Field(default=None, ge=1, le=64)
    reasoning_convergence_stop: StrictBool = False
    reasoning_seed: StrictInt | None = Field(default=None, ge=0, le=2**64 - 1)
    reasoning_known_positions: list[Annotated[StrictInt, Field(ge=0)]] | None = Field(
        default=None, min_length=1, strict=True
    )
    reasoning_delayed_suffix_positions: (
        list[Annotated[StrictInt, Field(ge=0)]] | None
    ) = Field(default=None, min_length=1, strict=True)
    reasoning_answer_noise: Literal["candidates", "vocab"] | None = None
    reasoning_token_noise: Literal["allowed", "vocab"] | None = None
    reasoning_token_proposals: Literal["allowed", "vocab"] | None = None
    reasoning_schedule_steps: StrictInt | None = Field(default=None, ge=1)
    reasoning_entropy_bound: (
        Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None
    ) = None
    reasoning_round_feedback_logits: StrictBool = False
    reasoning_preserve_feedback: StrictBool = False
    reasoning_answer_feedback: StrictBool = False
    reasoning_unmasked_feedback: StrictBool = False

    @model_validator(mode="after")
    def validate_answer(self) -> "DiffusionReadSpec":
        if self.answer_positions[0] >= len(self.seed_canvas):
            raise ValueError("answer_positions must be within seed_canvas")
        if len(set(self.logprob_token_ids)) != len(self.logprob_token_ids):
            raise ValueError("logprob_token_ids must be unique")
        positions = self.reasoning_positions
        if self.answer_pass_aggregation == "mean" and (
            positions is None
            or not self.answer_only_logprobs
            or self.reasoning_preserve_feedback
        ):
            raise ValueError(
                "Mean answer reads require reasoning, answer-only logprobs, "
                "and reasoning_preserve_feedback=False"
            )
        if positions is None:
            if (
                self.reasoning_forbidden_token_ids is not None
                or self.reasoning_steps is not None
                or self.reasoning_convergence_stop
                or self.reasoning_seed is not None
                or self.reasoning_known_positions is not None
                or self.reasoning_delayed_suffix_positions is not None
                or self.reasoning_answer_noise is not None
                or self.reasoning_token_noise is not None
                or self.reasoning_token_proposals is not None
                or self.reasoning_schedule_steps is not None
                or self.reasoning_entropy_bound is not None
                or self.reasoning_round_feedback_logits
                or self.reasoning_preserve_feedback
                or self.reasoning_answer_feedback
                or self.reasoning_unmasked_feedback
            ):
                raise ValueError("Reasoning options require reasoning_positions")
            return self
        if (
            positions[-1] >= len(self.seed_canvas)
            or positions != sorted(set(positions))
            or set(positions).intersection(self.answer_positions)
        ):
            raise ValueError(
                "reasoning_positions must be strictly increasing within seed_canvas "
                "and disjoint from answer_positions"
            )
        forbidden = self.reasoning_forbidden_token_ids
        known = self.reasoning_known_positions
        if known is not None and (
            known[-1] >= len(self.seed_canvas)
            or known != sorted(set(known))
            or set(known).intersection((*positions, *self.answer_positions))
        ):
            raise ValueError(
                "reasoning_known_positions must be strictly increasing within "
                "seed_canvas and disjoint from reasoning and answer positions"
            )
        delayed = self.reasoning_delayed_suffix_positions
        if delayed is not None and (
            delayed[0] <= positions[-1]
            or delayed[-1] >= len(self.seed_canvas)
            or delayed != sorted(set(delayed))
            or set(delayed).intersection(self.answer_positions)
        ):
            raise ValueError(
                "reasoning_delayed_suffix_positions must be strictly increasing "
                "within seed_canvas, after all reasoning positions and disjoint "
                "from answer_positions"
            )
        if forbidden is None or len(set(forbidden)) != len(forbidden):
            raise ValueError(
                "reasoning_forbidden_token_ids must be supplied and unique"
            )
        if self.reasoning_schedule_steps is not None and (
            self.reasoning_schedule_steps
            < (self.reasoning_steps if self.reasoning_steps is not None else 8)
        ):
            raise ValueError(
                "reasoning_schedule_steps must be at least reasoning_steps"
            )
        return self


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of "
        "the max_completion_tokens field",
    )
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: AnyResponseFormat | None = None
    seed: int | None = Field(None, ge=_INT64_MIN, le=_INT64_MAX)
    stop: StopParam = []
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: (
        Literal["none"]
        | Literal["auto"]
        | Literal["required"]
        | ChatCompletionNamedToolChoiceParam
        | None
    ) = "none"
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = Field(
        default=None,
        description=(
            "Constrains effort on reasoning for reasoning models. "
            "Currently supported values are none, minimal, low, medium, "
            "high, xhigh, and max. Reducing reasoning effort can result in "
            "faster responses and fewer tokens used on reasoning in a response. "
            "Note that 'max' is specific to the DeepSeek V4 series and is not "
            "part of the standard OpenAI API specification."
        ),
    )
    thinking_token_budget: ThinkingTokenBudget = None
    include_reasoning: bool = True
    parallel_tool_calls: bool | None = True

    # NOTE this will be ignored by vLLM
    user: str | None = None

    # --8<-- [start:chat-completion-sampling-params]
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    watermarking: bool = True
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_INT64_MAX)] | None = None
    truncation_side: Literal["left", "right"] | None = Field(
        default=None,
        description=(
            "Which side to truncate from when truncate_prompt_tokens is active. "
            "'right' keeps the first N tokens. "
            "'left' keeps the last N tokens."
        ),
    )
    prompt_logprobs: int | None = None
    logprob_token_ids: list[int] | None = Field(
        default=None,
        description=(
            "Specific vocab token IDs to return logprobs for at each generated "
            "position, in addition to the sampled token. More efficient than "
            "`top_logprobs=-1` when only a small fixed label set is needed "
            "(e.g. multilabel scoring "
            "where each label corresponds to a known vocab id). When set, "
            "this explicit token selection takes precedence over the natural "
            "top-k selected by `top_logprobs`. Requires `logprobs=True`."
        ),
    )
    allowed_token_ids: list[int] | None = None
    bad_words: list[str] = Field(default_factory=list)
    # --8<-- [end:chat-completion-sampling-params]

    # --8<-- [start:chat-completion-extra-params]
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."
        ),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    media_io_kwargs: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the media IO connectors, "
            "keyed by modality. Merged with engine-level media_io_kwargs."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )
    priority: int = Field(
        default=0,
        ge=_INT64_MIN,
        le=_INT64_MAX,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Stable session identity shared by related requests. Unlike "
            "request_id, this value is expected to remain stable across "
            "multiple requests in the same conversation or agent session."
        ),
    )

    return_tokens_as_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."
        ),
    )
    return_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified, the result will include token IDs alongside the "
            "generated text. In streaming mode, prompt_token_ids is included "
            "only in the first chunk, and token_ids contains the delta tokens "
            "for each chunk. This is useful for debugging or when you "
            "need to map generated text back to input tokens."
        ),
    )
    routed_experts_prompt_start: int = Field(
        default=0,
        ge=0,
        description="Skip the first N prompt tokens from returned routed-expert data.",
    )
    return_token_offsets: bool | None = Field(
        default=False,
        description=(
            "If true, return char-level (start, end) offsets for each "
            "token relative to the tokenized source string in the "
            "`token_offsets` field of the rendered response. Only "
            "supported on the `/v1/completions/render` and "
            "`/v1/chat/completions/render` endpoints; ignored on regular "
            "generation endpoints. Honored only for Fast (Rust-backed) "
            "tokenizers; otherwise `token_offsets` is null. For chat "
            "requests, offsets are relative to the templated prompt "
            "string (after applying the chat template). Multimodal "
            "inputs and pre-tokenized inputs always yield null."
        ),
    )
    return_prompt_text: bool | None = Field(
        default=None,
        description=(
            "If true, the response will include ``prompt_text`` containing the "
            "prompt string produced by chat templating. In streaming mode it "
            "is sent only on the first chunk. This is useful for inspecting "
            "exactly what was fed into the model."
        ),
    )

    return_assistant_tokens_mask: bool = Field(
        default=False,
        description=(
            "If true, the /render response will include an "
            "``assistant_tokens_mask`` field — a per-token list of 0/1 "
            "values indicating which tokens were assistant-generated. "
            "Requires the chat template to use ``{% generation %}`` "
            "tags.  When the template does not support it, "
            "``assistant_tokens_mask`` will be ``null``."
        ),
    )

    cache_salt: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )

    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )

    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )

    diffusion_reads: list[DiffusionReadSpec] | None = Field(
        default=None,
        min_length=1,
        description=(
            "One single-answer diffusion canvas per output, in choice-index order. "
            "Requires n to match the number of canvases and logprobs=True. "
            "Streaming, beam search, transfers, and sessions are unsupported."
        ),
    )

    repetition_detection: RepetitionDetectionParams | None = Field(
        default=None,
        description="Parameters for detecting repetitive N-gram patterns "
        "in output tokens. If such repetition is detected, generation will "
        "be ended early. LLMs can sometimes generate repetitive, unhelpful "
        "token patterns, stopping only when they hit the maximum output length "
        "(e.g. 'abcdabcdabcd...' or '\\emoji \\emoji \\emoji ...'). This feature "
        "can detect such behavior and terminate early, saving time and tokens.",
    )

    stream_interval: Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description=(
            "Number of tokens to batch into each streamed chunk. Raises the "
            "server's `--stream-interval` for this request. Values below the "
            "server setting are clamped up to it. The first and last chunks "
            "are always sent immediately. Ignored for non-streaming requests."
        ),
    )

    # --8<-- [end:chat-completion-extra-params]

    @field_validator("vllm_xargs", mode="wrap")
    @classmethod
    def preserve_reasoning_known_positions(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        name = "diffusion_reasoning_known_positions"
        if isinstance(value, dict) and name in value:
            positions = value[name]
            if positions is None:
                return handler(
                    {key: item for key, item in value.items() if key != name}
                )
            if not isinstance(positions, list) or any(
                type(pos) is not int for pos in positions
            ):
                raise VLLMValidationError(
                    f"{name} must be a list of integer positions or null.",
                    parameter=f"vllm_xargs.{name}",
                )
        return handler(value)

    @field_validator("vllm_xargs", mode="wrap")
    @classmethod
    def preserve_reasoning_delayed_suffix_positions(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        name = "diffusion_reasoning_delayed_suffix_positions"
        if isinstance(value, dict) and name in value:
            positions = value[name]
            if positions is None:
                return handler(
                    {key: item for key, item in value.items() if key != name}
                )
            if not isinstance(positions, list) or any(
                type(pos) is not int for pos in positions
            ):
                raise VLLMValidationError(
                    f"{name} must be a list of integer positions or null.",
                    parameter=f"vllm_xargs.{name}",
                )
        return handler(value)

    @field_validator("vllm_xargs", mode="wrap")
    @classmethod
    def preserve_reasoning_seed(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        name = "diffusion_reasoning_seed"
        if isinstance(value, dict) and name in value:
            seed = value[name]
            if seed is None:
                return handler(
                    {key: item for key, item in value.items() if key != name}
                )
            if type(seed) is not int or not 0 <= seed < 2**64:
                raise VLLMValidationError(
                    f"{name} must be an unsigned 64-bit integer or null.",
                    parameter=f"vllm_xargs.{name}",
                )
        return handler(value)

    @field_validator("vllm_xargs", mode="wrap")
    @classmethod
    def preserve_reasoning_feedback_booleans(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        feedback = {}
        if isinstance(value, dict):
            for name in (
                "diffusion_reasoning_round_feedback_logits",
                "diffusion_reasoning_preserve_feedback",
                "diffusion_reasoning_answer_feedback",
                "diffusion_reasoning_unmasked_feedback",
                "diffusion_reasoning_convergence_stop",
            ):
                if name in value:
                    if type(value[name]) is not bool:
                        raise VLLMValidationError(
                            f"{name} must be a boolean.",
                            parameter=f"vllm_xargs.{name}",
                        )
                    feedback[name] = value[name]
        parsed = handler(value)
        if feedback:
            parsed.update(feedback)
        return parsed

    @field_validator("vllm_xargs", mode="before")
    @classmethod
    def validate_reasoning_token_noise(cls, value: Any) -> Any:
        name = "diffusion_reasoning_token_noise"
        if (
            isinstance(value, dict)
            and name in value
            and (
                type(value[name]) is not str or value[name] not in ("allowed", "vocab")
            )
        ):
            raise VLLMValidationError(
                f"{name} must be 'allowed' or 'vocab'.",
                parameter=f"vllm_xargs.{name}",
            )
        return value

    @field_validator("vllm_xargs", mode="before")
    @classmethod
    def validate_reasoning_token_proposals(cls, value: Any) -> Any:
        name = "diffusion_reasoning_token_proposals"
        if (
            isinstance(value, dict)
            and name in value
            and (
                type(value[name]) is not str or value[name] not in ("allowed", "vocab")
            )
        ):
            raise VLLMValidationError(
                f"{name} must be 'allowed' or 'vocab'.",
                parameter=f"vllm_xargs.{name}",
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_reasoning_controls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        xargs = data.get("vllm_xargs")
        if isinstance(xargs, dict):
            for name in (
                "diffusion_reasoning_schedule_steps",
                "diffusion_reasoning_entropy_bound",
            ):
                if isinstance(xargs.get(name), bool):
                    raise VLLMValidationError(
                        f"{name} must be numeric, not boolean.",
                        parameter=f"vllm_xargs.{name}",
                    )
        return data

    @model_validator(mode="after")
    def validate_diffusion_reads(self) -> "ChatCompletionRequest":
        if (
            self.vllm_xargs is not None
            and "diffusion_answer_pass_aggregation" in self.vllm_xargs
        ):
            raise VLLMValidationError(
                "answer_pass_aggregation is supported only within diffusion_reads.",
                parameter="vllm_xargs.diffusion_answer_pass_aggregation",
            )
        if self.diffusion_reads is None:
            return self
        n = self.n if self.n is not None else 1
        if len(self.diffusion_reads) != n:
            raise VLLMValidationError(
                "diffusion_reads must contain exactly n canvases.",
                parameter="diffusion_reads",
            )
        if not self.logprobs:
            raise VLLMValidationError(
                "diffusion_reads requires logprobs=True.",
                parameter="logprobs",
            )
        unsupported = {
            "stream": bool(self.stream),
            "use_beam_search": self.use_beam_search,
            "logprob_token_ids": self.logprob_token_ids is not None,
            "top_logprobs": self.top_logprobs not in (None, 0),
            "kv_transfer_params": self.kv_transfer_params is not None,
            "ec_transfer_params": self.ec_transfer_params is not None,
            "session_id": self.session_id is not None,
            "truncate_prompt_tokens": self.truncate_prompt_tokens is not None,
            "reasoning_effort": self.reasoning_effort not in (None, "none"),
        }
        for name, present in unsupported.items():
            if present:
                raise VLLMValidationError(
                    f"diffusion_reads does not support {name}.", parameter=name
                )
        chat_kwargs = self.chat_template_kwargs or {}
        if (
            "enable_thinking" in chat_kwargs
            and chat_kwargs["enable_thinking"] is not False
        ):
            raise VLLMValidationError(
                "diffusion_reads requires enable_thinking=False when supplied.",
                parameter="chat_template_kwargs.enable_thinking",
            )
        conflicting_xargs = {
            "diffusion_read_only",
            "diffusion_seed_canvas",
            "diffusion_answer_positions",
            "diffusion_answer_only_logprobs",
            "diffusion_read_steps",
            "diffusion_answer_pass_aggregation",
            "diffusion_read_self_conditioning",
            "diffusion_read_self_conditioning_scope",
            "diffusion_reasoning_positions",
            "diffusion_reasoning_forbidden_token_ids",
            "diffusion_reasoning_steps",
            "diffusion_reasoning_convergence_stop",
            "diffusion_reasoning_seed",
            "diffusion_reasoning_known_positions",
            "diffusion_reasoning_delayed_suffix_positions",
            "diffusion_reasoning_answer_noise",
            "diffusion_reasoning_token_noise",
            "diffusion_reasoning_token_proposals",
            "diffusion_reasoning_schedule_steps",
            "diffusion_reasoning_entropy_bound",
            "diffusion_reasoning_round_feedback_logits",
            "diffusion_reasoning_preserve_feedback",
            "diffusion_reasoning_answer_feedback",
            "diffusion_reasoning_unmasked_feedback",
            "diffusion_canvas_length",
            "diffusion_max_steps",
            "kv_transfer_params",
            "ec_transfer_params",
            "session_id",
        }
        if conflicting_xargs.intersection(self.vllm_xargs or {}):
            raise VLLMValidationError(
                "diffusion_reads conflicts with diffusion, transfer, or session xargs.",
                parameter="vllm_xargs",
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        validate_cache_salt(data.get("cache_salt"))
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_messages_before(cls, data: Any) -> Any:
        """Pre-process message dicts before Pydantic field validation.

        Performs two normalizations in a single pass:
        - Converts tool_calls generators/iterators to lists so one-shot
          generators are not consumed during union type matching.
        - Renames the deprecated ``reasoning_content`` field to
          ``reasoning`` so downstream code only needs to check one field.
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                msg["tool_calls"] = list(tool_calls)
            reasoning_content = msg.pop("reasoning_content", None)
            if reasoning_content is not None and msg.get("reasoning") is None:
                msg["reasoning"] = reasoning_content
        return data

    @model_validator(mode="after")
    def _materialize_tool_calls_after(self) -> "ChatCompletionRequest":
        """Convert Pydantic ValidatorIterator wrappers back to lists.

        Even after the "before" validator converts iterables to lists,
        Pydantic re-wraps them in a ValidatorIterator when validating
        against ChatCompletionAssistantMessageParam's Iterable[...] type.
        This "after" pass materialises those wrappers so downstream code
        (tokenizers, model_dump_json) always sees plain lists.
        """
        for msg in self.messages:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                msg["tool_calls"] = list(tool_calls)
        return self

    _grammar_from_parser: bool = PrivateAttr(default=False)
    """CAUTION: Should only be set by the parser-engine adapter's adjust_request."""

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        extra_kwargs: dict[str, Any] = dict(
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
            documents=self.documents,
            reasoning_effort=self.reasoning_effort,
        )

        # When reasoning is requested, activate thinking for models whose
        # chat templates require explicit opt-in (e.g., Gemma4 defaults
        # enable_thinking to false). For templates that don't declare the
        # variable, resolve_chat_template_kwargs filters it out harmlessly.
        user_kwargs = self.chat_template_kwargs or {}
        if self.diffusion_reads is not None:
            extra_kwargs["enable_thinking"] = False
        elif self.reasoning_effort is not None and "enable_thinking" not in user_kwargs:
            extra_kwargs["enable_thinking"] = self.reasoning_effort != "none"

        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                extra_kwargs,
            ),
            media_io_kwargs=self.media_io_kwargs,
            return_assistant_tokens_mask=bool(self.return_assistant_tokens_mask),
            # No-tools requests default to tool_choice="none" at the API
            # layer. Collapse that default before rendering, so K3 emits a
            # model-visible tool-choice instruction only for requests with a
            # tools block.
            tool_choice=self.tool_choice if self.tools else None,
            response_format=self.response_format,
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        if self.max_completion_tokens is not None:
            max_output_tokens: int | None = self.max_completion_tokens
            max_output_tokens_param = "max_completion_tokens"
        else:
            max_output_tokens = self.max_tokens
            max_output_tokens_param = "max_tokens"

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=max_output_tokens or 0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            add_special_tokens=self.add_special_tokens,
            needs_detokenization=bool(self.echo and not self.return_token_ids),
            max_total_tokens_param="max_model_len",
            max_output_tokens_param=max_output_tokens_param,
            return_token_offsets=bool(self.return_token_offsets),
        )

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_beam_search_params(
        self, max_tokens: int, default_sampling_params: dict
    ) -> BeamSearchParams:
        n = self.n if self.n is not None else 1
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

        return BeamSearchParams(
            beam_width=n,
            max_tokens=max_tokens,
            ignore_eos=self.ignore_eos,
            temperature=temperature,
            length_penalty=self.length_penalty,
            include_stop_str_in_output=self.include_stop_str_in_output,
            skip_special_tokens=self.skip_special_tokens,
        )

    def extract_structured_outputs(self) -> StructuredOutputsParams | None:
        """Normalize request constraints into ``StructuredOutputsParams``."""
        return structured_outputs_from_response_format(
            self.structured_outputs,
            self.response_format,
        )

    def to_sampling_params(
        self,
        max_tokens: int,
        default_sampling_params: dict,
    ) -> SamplingParams:
        # Default parameters
        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"]
            )
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"]
            )
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"]
            )

        # Merge server-default stop_token_ids (e.g., model-specific tokens
        # like </call> for gpt-oss) with any request-specified ones
        stop_token_ids = self.stop_token_ids
        default_stop_ids = default_sampling_params.get("stop_token_ids")
        if default_stop_ids:
            if not stop_token_ids:
                stop_token_ids = list(default_stop_ids)
            else:
                stop_token_ids = list(
                    dict.fromkeys([*stop_token_ids, *default_stop_ids])
                )

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.top_logprobs

        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        if self.ec_transfer_params:
            # Pass in ec_transfer_params via extra_args
            extra_args["ec_transfer_params"] = self.ec_transfer_params
        return SamplingParams.from_optional(
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            watermarking=self.watermarking,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=stop_token_ids,
            logprobs=(
                self.top_logprobs
                if self.logprobs and not self.logprob_token_ids
                else None
            ),
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=self.logprob_token_ids or None,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            output_kind=(
                RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
            ),
            stream_interval=self.stream_interval,
            structured_outputs=self.extract_structured_outputs(),
            logit_bias=self.logit_bias,
            bad_words=self.bad_words,
            thinking_token_budget=self.thinking_token_budget,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=extra_args or None,
            skip_clone=True,  # Created fresh per request, safe to skip clone
            repetition_detection=self.repetition_detection,
            routed_experts_prompt_start=self.routed_experts_prompt_start,
        )

    def to_child_sampling_params(
        self, sampling_params: SamplingParams
    ) -> tuple[SamplingParams, ...] | None:
        """Build independent child parameters without changing the shared parent."""
        if self.diffusion_reads is None:
            return None
        children = []
        for read in self.diffusion_reads:
            if (
                sampling_params.max_tokens is None
                or len(read.seed_canvas) > sampling_params.max_tokens
            ):
                raise VLLMValidationError(
                    "Each diffusion_reads canvas must contain at most max_tokens "
                    "tokens; every complete canvas must fit within max_model_len.",
                    parameter="diffusion_reads",
                )
            child = deepcopy(sampling_params)
            child.n = 1
            child.max_tokens = len(read.seed_canvas)
            optimize_token_output = bool(
                self.return_tokens_as_token_ids and read.answer_only_logprobs
            )
            child.detokenize = not optimize_token_output
            child.flat_logprobs = optimize_token_output
            child.logprobs = None
            child.logprob_token_ids = list(read.logprob_token_ids)
            child.extra_args = {
                **(child.extra_args or {}),
                "diffusion_read_only": True,
                "diffusion_seed_canvas": list(read.seed_canvas),
                "diffusion_answer_positions": list(read.answer_positions),
                "diffusion_answer_only_logprobs": read.answer_only_logprobs,
                "diffusion_read_steps": read.steps,
                "diffusion_read_self_conditioning": read.self_conditioning,
                "diffusion_read_self_conditioning_scope": read.self_conditioning_scope,
            }
            if read.answer_pass_aggregation == "mean":
                child.extra_args["diffusion_answer_pass_aggregation"] = "mean"
            if read.reasoning_positions is not None:
                assert read.reasoning_forbidden_token_ids is not None
                child.extra_args.update(
                    diffusion_reasoning_positions=list(read.reasoning_positions),
                    diffusion_reasoning_forbidden_token_ids=list(
                        read.reasoning_forbidden_token_ids
                    ),
                    diffusion_reasoning_steps=(
                        read.reasoning_steps if read.reasoning_steps is not None else 8
                    ),
                    diffusion_reasoning_answer_noise=(
                        read.reasoning_answer_noise or "candidates"
                    ),
                )
                if read.reasoning_schedule_steps is not None:
                    child.extra_args["diffusion_reasoning_schedule_steps"] = (
                        read.reasoning_schedule_steps
                    )
                if read.reasoning_convergence_stop:
                    child.extra_args["diffusion_reasoning_convergence_stop"] = True
                if read.reasoning_token_noise == "vocab":
                    child.extra_args["diffusion_reasoning_token_noise"] = "vocab"
                if read.reasoning_token_proposals == "vocab":
                    child.extra_args["diffusion_reasoning_token_proposals"] = "vocab"
                if read.reasoning_seed is not None:
                    child.extra_args["diffusion_reasoning_seed"] = read.reasoning_seed
                if read.reasoning_known_positions is not None:
                    child.extra_args["diffusion_reasoning_known_positions"] = list(
                        read.reasoning_known_positions
                    )
                if read.reasoning_delayed_suffix_positions is not None:
                    child.extra_args["diffusion_reasoning_delayed_suffix_positions"] = (
                        list(read.reasoning_delayed_suffix_positions)
                    )
                if read.reasoning_entropy_bound is not None:
                    child.extra_args["diffusion_reasoning_entropy_bound"] = (
                        read.reasoning_entropy_bound
                    )
                if read.reasoning_round_feedback_logits:
                    child.extra_args["diffusion_reasoning_round_feedback_logits"] = True
                if read.reasoning_preserve_feedback:
                    child.extra_args["diffusion_reasoning_preserve_feedback"] = True
                if read.reasoning_answer_feedback:
                    child.extra_args["diffusion_reasoning_answer_feedback"] = True
                if read.reasoning_unmasked_feedback:
                    child.extra_args["diffusion_reasoning_unmasked_feedback"] = True
            children.append(child)
        return tuple(children)

    @model_validator(mode="before")
    @classmethod
    def validate_response_format(cls, data):
        if not isinstance(data, dict):
            return data
        response_format = data.get("response_format")
        if response_format is None:
            return data

        rf_type = (
            response_format.get("type")
            if isinstance(response_format, dict)
            else getattr(response_format, "type", None)
        )

        if rf_type == "json_schema":
            json_schema = (
                response_format.get("json_schema")
                if isinstance(response_format, dict)
                else getattr(response_format, "json_schema", None)
            )
            if json_schema is None:
                raise VLLMValidationError(
                    "When response_format type is 'json_schema', the "
                    "'json_schema' field must be provided.",
                    parameter="response_format",
                )

        if rf_type == "structural_tag":
            validate_structural_tag_response_format(response_format)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("stream_options") and not data.get("stream"):
            raise VLLMValidationError(
                "Stream options can only be defined when `stream=True`.",
                parameter="stream_options",
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("logprob_token_ids") and data.get("use_beam_search"):
            raise VLLMValidationError(
                "`logprob_token_ids` is not supported with beam search.",
                parameter="logprob_token_ids",
            )

        if data.get("logprob_token_ids") and not data.get("logprobs"):
            raise VLLMValidationError(
                "when using `logprob_token_ids`, `logprobs` must be set to true.",
                parameter="logprob_token_ids",
            )

        # These fields are integers, but `mode="before"` runs on the raw
        # request data, so a non-numeric value (e.g. a JSON string) would
        # reach the comparisons below and raise TypeError -> HTTP 500. Reject
        # it here so the client gets a clean 400 instead.
        for field_name in ("prompt_logprobs", "top_logprobs"):
            field_value = data.get(field_name)
            if field_value is not None and not isinstance(field_value, (int, float)):
                raise VLLMValidationError(
                    f"`{field_name}` must be an integer.",
                    parameter=field_name,
                    value=field_value,
                )
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and (prompt_logprobs > 0 or prompt_logprobs == -1):
                raise VLLMValidationError(
                    "`prompt_logprobs` are not available when `stream=True`.",
                    parameter="prompt_logprobs",
                )

            if prompt_logprobs < 0 and prompt_logprobs != -1:
                raise VLLMValidationError(
                    "`prompt_logprobs` must be a positive value or -1.",
                    parameter="prompt_logprobs",
                    value=prompt_logprobs,
                )
        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0 and top_logprobs != -1:
                raise VLLMValidationError(
                    "`top_logprobs` must be a positive value or -1.",
                    parameter="top_logprobs",
                    value=top_logprobs,
                )

            if (top_logprobs == -1 or top_logprobs > 0) and not data.get("logprobs"):
                raise VLLMValidationError(
                    "when using `top_logprobs`, `logprobs` must be set to true.",
                    parameter="top_logprobs",
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_structured_outputs_count(cls, data):
        if isinstance(data, ValueError):
            raise data
        if not isinstance(data, dict):
            return data

        if data.get("structured_outputs", None) is None:
            return data

        structured_outputs_kwargs = data["structured_outputs"]
        # structured_outputs may arrive as a dict (from JSON/raw kwargs) or
        # as a StructuredOutputsParams dataclass instance.
        is_dataclass = isinstance(structured_outputs_kwargs, StructuredOutputsParams)
        count = sum(
            (
                getattr(structured_outputs_kwargs, k, None)
                if is_dataclass
                else structured_outputs_kwargs.get(k)
            )
            is not None
            for k in ("json", "regex", "choice")
        )
        # you can only use one kind of constraints for structured outputs
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice').",
            )
        # you can only either use structured outputs or tools, not both
        if count > 0 and data.get("tool_choice", "none") not in (
            "none",
            "auto",
            "required",
        ):
            raise VLLMValidationError(
                "You can only either use constraints for structured outputs "
                "or tools, not both.",
            )
        validate_structured_outputs_structural_tag(structured_outputs_kwargs)
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):
        if isinstance(data, ValueError):
            raise data
        if not isinstance(data, dict):
            return data

        # Reject empty tools array, matching OpenAI API behavior
        if data.get("tools") == []:
            raise VLLMValidationError(
                "`tools` must not be an empty array. "
                "Either provide at least one tool or omit the field entirely.",
                parameter="tools",
            )

        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- no validation is needed for tools
        if "tool_choice" in data and data["tool_choice"] == "none":
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data and data["tool_choice"] is not None:
            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise VLLMValidationError(
                    "When using `tool_choice`, `tools` must be set.",
                    parameter="tool_choice",
                )

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in ["auto", "required"] and not isinstance(
                data["tool_choice"], dict
            ):
                raise VLLMValidationError(
                    f"Invalid value for `tool_choice`: {data['tool_choice']}! "
                    'Only named tools, "none", "auto" or "required" '
                    "are supported.",
                    parameter="tool_choice",
                )

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            correct_usage_message = (
                'Correct usage: `{"type": "function",'
                ' "function": {"name": "my_function"}}`'
            )
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                function = data["tool_choice"].get("function")
                if not isinstance(function, dict):
                    raise VLLMValidationError(
                        f"Invalid value for `function`: `{function}` in "
                        f"`tool_choice`! {correct_usage_message}",
                        parameter="tool_choice.function",
                    )
                if "name" not in function:
                    raise VLLMValidationError(
                        f"Expected field `name` in `function` in "
                        f"`tool_choice`! {correct_usage_message}",
                        parameter="tool_choice.function.name",
                    )
                function_name = function["name"]
                if not isinstance(function_name, str) or len(function_name) == 0:
                    raise VLLMValidationError(
                        f"Invalid `name` in `function`: `{function_name}`"
                        f" in `tool_choice`! {correct_usage_message}",
                        parameter="tool_choice.function.name",
                    )
                for tool in data["tools"]:
                    if tool["function"]["name"] == function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise VLLMValidationError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`",
                        parameter="tool_choice",
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise VLLMValidationError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True.",
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_system_message_content_type(cls, data):
        """Warn if system messages contain non-text content.

        According to OpenAI API spec, system messages can only be of type
        'text'. We log a warning instead of rejecting to avoid breaking
        users who intentionally send multimodal system messages.
        See: https://platform.openai.com/docs/api-reference/chat/create#chat_create-messages-system_message
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages", [])
        for msg in messages:
            # Check if this is a system message
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")

                # If content is a list (multimodal format)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            part_type = part.get("type")
                            # Infer type when 'type' field is not explicit
                            if part_type is None:
                                if "image_url" in part or "image_pil" in part:
                                    part_type = "image_url"
                                elif "image_embeds" in part:
                                    part_type = "image_embeds"
                                elif "audio_url" in part:
                                    part_type = "audio_url"
                                elif "input_audio" in part:
                                    part_type = "input_audio"
                                elif "audio_embeds" in part:
                                    part_type = "audio_embeds"
                                elif "video_url" in part:
                                    part_type = "video_url"

                            # Warn about non-text content in system messages
                            if part_type and part_type != "text":
                                logger.warning_once(
                                    "System messages should only contain text "
                                    "content according to the OpenAI API spec. "
                                    "Found content type: '%s'.",
                                    part_type,
                                )

        return data


class BatchChatCompletionRequest(OpenAIBaseModel):
    """Request model for the /v1/chat/completions/batch endpoint.

    Accepts the same fields as ChatCompletionRequest except that ``messages``
    is a list of conversations (each conversation is a
    ``list[ChatCompletionMessageParam]``).  Each conversation is processed
    independently and the response contains one choice per conversation,
    indexed 0, 1, ..., N-1.

    Current limitations compared to the single-conversation endpoint:
    - Streaming is not supported (``stream`` must be False or omitted).
    - Tool use is not supported (``tools`` must be omitted).
    - Beam search is not supported (``use_beam_search`` must be False or omitted).
    - The ``n`` parameter must be 1 (or omitted).
    """

    messages: list[Annotated[list[ChatCompletionMessageParam], Field(min_length=1)]] = (
        Field(..., min_length=1)
    )
    model: str | None = None

    # Shared sampling / generation fields — mirror ChatCompletionRequest.
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    logprob_token_ids: list[int] | None = Field(
        default=None,
        description=(
            "Specific vocab token IDs to return logprobs for at each generated "
            "position, in addition to the sampled token. Requires "
            "`logprobs=True`."
        ),
    )
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: Any | None = None
    seed: int | None = Field(None, ge=_INT64_MIN, le=_INT64_MAX)
    stop: StopParam = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    tool_choice: Literal["none"] | None = "none"
    include_reasoning: bool = True

    # vLLM extensions
    best_of: int | None = None
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float | None = 1.0
    early_stopping: bool = False
    structured_outputs: StructuredOutputsParams | None = None
    request_id: str | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    media_io_kwargs: dict[str, dict[str, Any]] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
    priority: int = Field(default=0, ge=_INT64_MIN, le=_INT64_MAX)
    cache_salt: str | None = None
    include_stop_str_in_output: bool = False
    guided_decoding_backend: str | None = None
    echo: bool = False
    # None falls back to the server-level --return-tokens-as-token-ids default,
    # matching ChatCompletionRequest.return_tokens_as_token_ids.
    return_tokens_as_token_ids: bool | None = None
    return_token_ids: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_batch_mode(cls, data: Any) -> Any:
        if isinstance(data, BatchChatCompletionRequest):
            data = data.model_dump(exclude_unset=True)
        if not isinstance(data, dict):
            return data
        if data.get("use_beam_search"):
            raise VLLMValidationError(
                "Batch chat completions do not support beam search. "
                "Please set `use_beam_search` to False.",
                parameter="use_beam_search",
            )
        if data.get("logprob_token_ids") and not data.get("logprobs"):
            raise VLLMValidationError(
                "when using `logprob_token_ids`, `logprobs` must be set to true.",
                parameter="logprob_token_ids",
            )
        response_format = data.get("response_format")
        if response_format is not None:
            rf_type = (
                response_format.get("type")
                if isinstance(response_format, dict)
                else getattr(response_format, "type", None)
            )
            if rf_type == "structural_tag":
                validate_structural_tag_response_format(response_format)
        if (structured_outputs := data.get("structured_outputs")) is not None:
            validate_structured_outputs_structural_tag(structured_outputs)
        n = data.get("n", 1)
        if n is not None and n != 1:
            raise VLLMValidationError(
                "Batch chat completions do not support `n > 1`. Please set `n` to 1.",
                parameter="n",
                value=n,
            )
        return data

    def to_chat_completion_request(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletionRequest:
        """Build a single-conversation ChatCompletionRequest from one conversation."""
        data = self.model_dump(exclude={"messages"}, exclude_none=True)
        data["messages"] = messages
        return ChatCompletionRequest.model_validate(data)
