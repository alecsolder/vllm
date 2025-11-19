# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
import os
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.harmony_utils import parse_chat_output
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)

TRIGGERS = ["<|channel|>", "<|start|>assistant"]
BASE_TAGS = [
    # Allow normal reasoning messages as the first message
    {
        "type": "tag",
        "begin": "<|channel|>analysis",
        "content": {"type": "regex", "pattern": "(?:)"},
        "end": "<|message|>",
    },
    {
        "type": "tag",
        "begin": "<|channel|>commentary",
        "content": {"type": "regex", "pattern": "(?:)"},
        "end": "<|message|>",
    },
    # Allow final messages as the first message
    {
        "type": "tag",
        "begin": "<|channel|>final",
        "content": {"type": "regex", "pattern": "(?:)"},
        "end": "<|message|>",
    },
    # Allow final messages as the last message
    {
        "type": "tag",
        "begin": "<|start|>assistant<|channel|>final",
        "content": {"type": "regex", "pattern": "(?:)"},
        "end": "<|message|>",
    },
]


STRUCTURAL_TAG_TEMPLATE = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<|channel|>", "<|start|>assistant"],
        "tags": [],
        "at_least_one": True,
        "stop_after_first": False,
    },
}


def create_tool_tags(channel_name: str, tool_name: str) -> list[dict]:
    """
    Generate tool-specific tags based on channel name and tool name.

    Args:
        channel_name: The channel name (e.g., "analysis", "commentary")
        tool_name: The tool name (e.g., "python", "container")

    Returns:
        List of two tag dictionaries for first and last message positions
    """
    analysis_content_type = "code"
    commentary_content_type = "<|constrain|>json"
    content_type = (
        analysis_content_type if channel_name == "analysis" else commentary_content_type
    )
    return [
        # Tool as first message
        {
            "type": "tag",
            "begin": f"<|channel|>{channel_name} to={tool_name}",
            "content": {"type": "regex", "pattern": "(?:)"},
            "end": f" {content_type}<|message|>",
        },
        # Tool as last message
        {
            "type": "tag",
            "begin": f"<|start|>assistant<|channel|>{channel_name} to={tool_name}",
            "content": {"type": "regex", "pattern": "(?:)"},
            "end": f" {content_type}<|message|>",
        },
    ]


def get_structural_tags(analysis_tools: set[str], commentary_tools: set[str]):
    # Start with base tags
    tags = BASE_TAGS.copy()

    # Add tool-specific tags for commentary channel
    for tool_name in commentary_tools:
        if tool_name:  # Skip empty strings from split
            tags.extend(create_tool_tags("commentary", tool_name))

    # Add tool-specific tags for analysis channel
    for tool_name in analysis_tools:
        if tool_name:  # Skip empty strings from split
            tags.extend(create_tool_tags("analysis", tool_name))

    # Build the complete structural tag
    structural_tags = copy.deepcopy(STRUCTURAL_TAG_TEMPLATE)
    structural_tags["format"]["tags"] = tags
    print(structural_tags)
    return json.dumps(structural_tags)


class GptOssReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # The model can output some special tokens between "final" and "<|message|>"
        # So we need to look for both sequences to determine the end of reasoning.
        self.reasoning_end_token_ids_prefix = self.model_tokenizer.encode(
            "<|channel|>final"
        )
        self.reasoning_end_token_ids_suffix = self.model_tokenizer.encode("<|message|>")
        self.reasoning_max_num_between_tokens = 20

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_token_ids_prefix = self.reasoning_end_token_ids_prefix
        end_token_ids_suffix = self.reasoning_end_token_ids_suffix
        assert len(end_token_ids_prefix) > 0, "reasoning_end_token_ids_prefix is empty"
        assert len(end_token_ids_suffix) > 0, "reasoning_end_token_ids_suffix is empty"
        # Check if the end sequence is present in the input_ids.
        # We search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - len(end_token_ids_prefix), -1, -1):
            if input_ids[i : i + len(end_token_ids_prefix)] == end_token_ids_prefix:
                # We have found the prefix, now we look for the suffix after the prefix.
                suffix_start = i + len(end_token_ids_prefix)
                for j in range(
                    suffix_start, len(input_ids) - len(end_token_ids_suffix) + 1
                ):
                    if j - suffix_start >= self.reasoning_max_num_between_tokens:
                        break
                    if (
                        input_ids[j : j + len(end_token_ids_suffix)]
                        == end_token_ids_suffix
                    ):
                        return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content, _ = parse_chat_output(input_ids)
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content, _ = parse_chat_output(list(previous_token_ids))
        cur_reasoning, cur_content, _ = parse_chat_output(list(current_token_ids))
        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            if cur_reasoning.startswith(prev_r):
                reasoning_delta = cur_reasoning[len(prev_r) :] or None
            else:
                reasoning_delta = cur_reasoning
        if cur_content is not None:
            prev_c = prev_content or ""
            if cur_content.startswith(prev_c):
                content_delta = cur_content[len(prev_c) :] or None
            else:
                content_delta = cur_content
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "gpt-oss has a special branch for parsing reasoning in non-streaming mode. This method shouldn't be used."  # noqa: E501
        )

    # This function prepares the structural tag to format reasoning output
    def prepare_structured_tag(
        self, original_tag: str | None, analysis_tools, commentary_tools
    ) -> str:
        if original_tag is None:
            return get_structural_tags(analysis_tools, commentary_tools)
        else:
            # There is potential risk for appending the tag to the original tag
            return original_tag
