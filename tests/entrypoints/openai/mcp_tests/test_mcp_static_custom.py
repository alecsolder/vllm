# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for static MCP tools (custom/non-elevated) using --tool-server."""

import json
import subprocess

import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer, find_free_port

from ..responses_utils import build_conversation_from_response, verify_tool_on_channel
from .memory_mcp_server import start_test_server

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def memory_mcp_server():
    """Start Memory MCP server as subprocess."""
    # Find a free port
    port = find_free_port()

    # Start memory MCP server using helper
    process = start_test_server(port)

    yield f"http://localhost:{port}/sse", port

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture(scope="module")
def memory_custom_server(monkeypatch_module, memory_mcp_server):
    """vLLM server with Memory MCP tool as custom (not elevated) via --tool-server."""
    server_url, port = memory_mcp_server
    args = ["--enforce-eager", "--tool-server", f"localhost:{port}"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        # NO GPT_OSS_SYSTEM_TOOL_MCP_LABELS - memory is custom tool
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def memory_custom_client(memory_custom_server):
    async with memory_custom_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_custom(memory_custom_client, model_name: str):
    """Test Memory MCP tool as custom (not elevated).

    When memory is NOT in GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
    - Tool should be in developer message (not system message)
    - Tool calls should be on commentary channel
    - Tool responses should be on commentary channel
    """
    response = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store and memory.retrieve tools. "
            "Never simulate tool execution."
        ),
        input=("Store the key 'test_key' with value 'test_value' and then retrieve it"),
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "test-session-custom"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0

    # Verify input messages: Should have developer message with tool
    developer_messages = [
        msg for msg in response.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_messages) > 0, "Developer message expected for custom tools"

    # Verify output messages: Tool calls and responses on commentary channel
    verify_tool_on_channel(response.output_messages, "memory.", "commentary")

    # Verify McpCall items (tool invocations)
    from openai.types.responses.response_output_item import McpCall

    mcp_calls = [
        item for item in reversed(response.output) if isinstance(item, McpCall)
    ]

    assert len(mcp_calls) > 0, "Should have at least one McpCall"

    for mcp_call in mcp_calls:
        # Verify it's a memory tool call
        assert mcp_call.server_label == "memory"
        assert mcp_call.name in ["store", "retrieve"]

        # Verify arguments make sense
        assert mcp_call.arguments is not None
        args = json.loads(mcp_call.arguments)
        assert "key" in args

        # Verify output was populated
        assert mcp_call.output is not None
        assert len(mcp_call.output) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_with_headers(memory_custom_client, model_name: str):
    """Test Memory MCP tool with custom headers for memory isolation.

    Different x-memory-id headers should provide isolated memory spaces.
    This tests that headers are properly passed through the MCP protocol.
    """
    # First request with session-1
    response1 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store tool. Never simulate tool execution."
        ),
        input="Store the key 'isolated_key' with value 'session_1_value'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "session-1"},
            }
        ],
    )

    assert response1.status == "completed"
    assert response1.usage.output_tokens_details.tool_output_tokens > 0

    # Second request with session-2 (different memory space)
    response2 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.retrieve tool. Never simulate tool execution."
        ),
        input="Retrieve the value for key 'isolated_key'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "session-2"},
            }
        ],
    )

    assert response2.status == "completed"
    # The key should NOT be found in session-2 (memory isolation working)
    # Check McpCall output field for exact error message
    from openai.types.responses.response_output_item import McpCall

    mcp_call_output = None
    for item in response2.output:
        if isinstance(item, McpCall) and item.output:
            mcp_call_output = item.output
            break

    # Memory isolation: key from session-1 should not be in session-2
    assert mcp_call_output is not None, "Should have McpCall with output"
    assert "Key 'isolated_key' not found in memory space 'session-2'" in mcp_call_output


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_conversation_continuation_with_store(
    memory_custom_client, model_name: str
):
    """Test conversation continuation with MCP tools using store=True.

    This test demonstrates:
    1. First request: Store a value in memory with store=True
    2. Second request: Use previous_response_id for stateful continuation
    3. Verify that conversation context is maintained across requests
    """
    # First request: Store a value in memory with store=True
    response1 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store tool to store data. "
            "Never simulate tool execution."
        ),
        input="Store the key 'favorite_food' with value 'pizza'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": "http://unused",
                "headers": {"x-memory-id": "store-continuation-test"},
            }
        ],
        store=True,
        extra_body={"enable_response_messages": True},
    )

    assert response1.status == "completed"
    assert response1.usage.output_tokens_details.tool_output_tokens > 0

    # Verify first response has MCP calls
    from openai.types.responses.response_output_item import McpCall

    mcp_calls_1 = [item for item in response1.output if isinstance(item, McpCall)]
    assert len(mcp_calls_1) > 0, "First request should have MCP calls"

    # Verify store was called
    store_call = next((c for c in mcp_calls_1 if c.name == "store"), None)
    assert store_call is not None, "Should have called store tool"
    assert store_call.output is not None, "Store call should have output"

    # Second request: Continue conversation using previous_response_id
    # With store=True, we can use previous_response_id instead of manually
    # building history
    response2 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.retrieve tool to retrieve data. "
            "Never simulate tool execution."
        ),
        input="What food did I just store? Retrieve it from memory.",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": "http://unused",
                "headers": {"x-memory-id": "store-continuation-test"},
            }
        ],
        previous_response_id=response1.id,
        extra_body={"enable_response_messages": True},
    )

    assert response2.status == "completed"
    assert response2.usage.output_tokens_details.tool_output_tokens > 0

    # Verify second response has MCP calls
    mcp_calls_2 = [item for item in response2.output if isinstance(item, McpCall)]
    assert len(mcp_calls_2) > 0, "Second request should have MCP calls"

    # Verify retrieve was called
    retrieve_call = next((c for c in mcp_calls_2 if c.name == "retrieve"), None)
    assert retrieve_call is not None, "Should have called retrieve tool"
    assert retrieve_call.output is not None, "Retrieve call should have output"

    # Verify the retrieved value matches what we stored
    assert "pizza" in retrieve_call.output.lower(), (
        "Retrieved value should contain 'pizza'"
    )

    # Verify input messages include conversation history
    assert len(response2.input_messages) > len(response1.input_messages), (
        "Second request should have more input messages (conversation history)"
    )

    # Verify there are 2 user messages (original + follow-up)
    user_messages = [
        msg for msg in response2.input_messages if msg["author"]["role"] == "user"
    ]
    assert len(user_messages) == 2, (
        "Second request should have 2 user messages when using previous_response_id"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_conversation_continuation_manual_history(
    memory_custom_client, model_name: str
):
    """Test conversation continuation with MCP tools using manual history.

    This test demonstrates:
    1. First request: Store a value in memory (without store=True)
    2. Second request: Manually pass input_messages + output_messages as history
    3. Verify that conversation context is maintained across requests
    """
    # First request: Store a value in memory without store=True
    response1 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store tool to store data. "
            "Never simulate tool execution."
        ),
        input="Store the key 'favorite_color' with value 'blue'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": "http://unused",
                "headers": {"x-memory-id": "manual-continuation-test"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response1.status == "completed"
    assert response1.usage.output_tokens_details.tool_output_tokens > 0

    # Verify first response has MCP calls
    from openai.types.responses.response_output_item import McpCall

    mcp_calls_1 = [item for item in response1.output if isinstance(item, McpCall)]
    assert len(mcp_calls_1) > 0, "First request should have MCP calls"

    # Verify store was called
    store_call = next((c for c in mcp_calls_1 if c.name == "store"), None)
    assert store_call is not None, "Should have called store tool"
    assert store_call.output is not None, "Store call should have output"

    # Second request: Manually build conversation history
    conversation_history = build_conversation_from_response(
        response1, "What color did I just store? Retrieve it from memory."
    )

    response2 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.retrieve tool to retrieve data. "
            "Never simulate tool execution."
        ),
        input=conversation_history,
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": "http://unused",
                "headers": {"x-memory-id": "manual-continuation-test"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response2.status == "completed"
    assert response2.usage.output_tokens_details.tool_output_tokens > 0

    # Verify second response has MCP calls
    mcp_calls_2 = [item for item in response2.output if isinstance(item, McpCall)]
    assert len(mcp_calls_2) > 0, "Second request should have MCP calls"

    # Verify retrieve was called
    retrieve_call = next((c for c in mcp_calls_2 if c.name == "retrieve"), None)
    assert retrieve_call is not None, "Should have called retrieve tool"
    assert retrieve_call.output is not None, "Retrieve call should have output"

    # Verify the retrieved value matches what we stored
    assert "blue" in retrieve_call.output.lower(), (
        "Retrieved value should contain 'blue'"
    )

    # Verify input messages include conversation history
    assert len(response2.input_messages) > len(response1.input_messages), (
        "Second request should have more input messages (conversation history)"
    )

    # Verify there are 2 user messages (original + follow-up)
    user_messages = [
        msg for msg in response2.input_messages if msg["author"]["role"] == "user"
    ]
    assert len(user_messages) == 2, (
        "Second request should have 2 user messages when using manual history"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_then_switch_to_function_tool(
    memory_custom_client, model_name: str
):
    """Test switching from MCP tools to function tools in continuation.

    This test demonstrates:
    1. First request: Use MCP tool with store=True
    2. Second request: Switch to function tool using previous_response_id
    3. Verify that function tool call is made correctly
    """
    # First request: Use MCP tool to store data with store=True
    response1 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store tool to store data. "
            "Never simulate tool execution."
        ),
        input="Store the key 'weather_city' with value 'Tokyo'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": "http://unused",
                "headers": {"x-memory-id": "switch-tools-test"},
            }
        ],
        store=True,
        extra_body={"enable_response_messages": True},
    )

    assert response1.status == "completed"
    assert response1.usage.output_tokens_details.tool_output_tokens > 0

    # Verify first response has MCP calls
    from openai.types.responses.response_output_item import McpCall

    mcp_calls_1 = [item for item in response1.output if isinstance(item, McpCall)]
    assert len(mcp_calls_1) > 0, "First request should have MCP calls"

    # Second request: Switch to function tool (independent task)
    response2 = await memory_custom_client.responses.create(
        model=model_name,
        input="What's the weather like in London?",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": ("Get current temperature for a city in celsius."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        previous_response_id=response1.id,
        temperature=0.0,  # More deterministic
        extra_body={"enable_response_messages": True},
    )

    assert response2.status == "completed"

    # Verify we got some output
    assert len(response2.output) > 0, "Should have output"

    # Check if we have a function call for get_weather with London
    from openai.types.responses import ResponseFunctionToolCall

    function_calls = [
        item for item in response2.output if isinstance(item, ResponseFunctionToolCall)
    ]

    # The test passes if either:
    # 1. We got a function call with London, OR
    # 2. We got a text response (model may refuse to call without real data)
    if len(function_calls) > 0:
        import json

        weather_call = next(
            (c for c in function_calls if c.name == "get_weather"), None
        )
        if weather_call:
            args = json.loads(weather_call.arguments)
            assert "london" in args["city"].lower()
