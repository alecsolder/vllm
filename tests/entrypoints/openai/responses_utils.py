# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Utility functions for Response API testing.

This module provides reusable helpers to reduce code duplication across
response API test files.
"""


def verify_tool_on_channel(output_messages, tool_prefix, expected_channel):
    """Verify tool calls and responses are on expected channel.

    This helper eliminates the common pattern of looping through output_messages
    to verify tool messages are on the correct channel (analysis vs commentary).

    Args:
        output_messages: response.output_messages list
        tool_prefix: Tool name prefix to match (e.g., "memory.", "python")
        expected_channel: Expected channel name ("analysis" or "commentary")

    Raises:
        AssertionError: If tool calls/responses not found or on wrong channel

    Example:
        >>> verify_tool_on_channel(response.output_messages, "memory.", "commentary")
    """
    tool_call_found = False
    tool_response_found = False

    for message in output_messages:
        # Check for tool call (message sent TO the tool)
        recipient = message.get("recipient", "")
        if recipient and recipient.startswith(tool_prefix):
            tool_call_found = True
            assert message.get("channel") == expected_channel, (
                f"Tool call should be on {expected_channel} channel, "
                f"got {message.get('channel')}"
            )

        # Check for tool response (message FROM the tool)
        author = message.get("author", {})
        author_name = author.get("name", "")
        if author.get("role") == "tool" and author_name.startswith(tool_prefix):
            tool_response_found = True
            assert message.get("channel") == expected_channel, (
                f"Tool response should be on {expected_channel} channel, "
                f"got {message.get('channel')}"
            )

    assert tool_call_found, f"Should have found at least one {tool_prefix} tool call"
    assert tool_response_found, (
        f"Should have found at least one {tool_prefix} tool response"
    )


def build_conversation_from_response(response, new_user_message):
    """Build conversation history from previous response + new message.

    Extracts text from response output and builds a message list suitable
    for use as the input parameter in the next request. This enables manual
    conversation continuation without using store=True or previous_response_id.

    Args:
        response: Previous response object with input_messages and output
        new_user_message: New user message text to append

    Returns:
        List of message dicts in format:
        [
            {"role": "user", "content": "<original_input>"},
            {"role": "assistant", "content": "<assistant_response>"},
            {"role": "user", "content": "<new_user_message>"}
        ]

    Example:
        >>> history = build_conversation_from_response(
        ...     response1, "What did I just store?"
        ... )
        >>> response2 = client.responses.create(
        ...     model=model_name, input=history, tools=[...]
        ... )
    """
    from openai.types.responses.response_output_item import ResponseOutputMessage

    # Extract original user input from input_messages
    original_input = None
    for msg in response.input_messages:
        if msg.get("author", {}).get("role") == "user":
            # Get text from first user message
            content = msg.get("content", [])
            if isinstance(content, str):
                original_input = content
            elif isinstance(content, list) and len(content) > 0:
                # Handle structured content format
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("input_text")
                        if text:
                            original_input = text
                            break
            if original_input:
                break

    if original_input is None:
        # Fallback: use empty string if we can't find original input
        original_input = ""

    # Extract assistant response text from output
    text_outputs = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    text_outputs.append(content_item.text)
    assistant_text = " ".join(text_outputs)

    # Build conversation history
    return [
        {"role": "user", "content": original_input},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": new_user_message},
    ]
