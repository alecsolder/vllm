# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dynamic MCP tools (custom/non-elevated) using server_url."""

import json
import subprocess

import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer, find_free_port

from ..responses_utils import verify_tool_on_channel
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
def server_without_static_tools(monkeypatch_module):
    """vLLM server WITHOUT static --tool-server for dynamic registration tests."""
    args = ["--enforce-eager"]  # No --tool-server!

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        # NO GPT_OSS_SYSTEM_TOOL_MCP_LABELS - tools will be custom
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client_without_static_tools(server_without_static_tools):
    async with server_without_static_tools.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_dynamic_custom(
    client_without_static_tools, memory_mcp_server, model_name: str
):
    """Test Memory MCP tool with dynamic per-request registration (custom/non-elevated).

    This tests the dynamic MCP server registration flow where:
    1. No static --tool-server is configured
    2. Tool server is registered dynamically via server_url in the request
    3. Tool should be custom (not elevated)
    4. Tool calls/responses should be on commentary channel
    """
    server_url, port = memory_mcp_server

    response = await client_without_static_tools.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store and memory.retrieve tools. "
            "Never simulate tool execution."
        ),
        input=(
            "Store the key 'dynamic_test_key' with value 'dynamic_test_value' "
            "and then retrieve it"
        ),
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": server_url,  # Dynamic registration via server_url!
                "headers": {"x-memory-id": "test-session-dynamic-custom"},
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
