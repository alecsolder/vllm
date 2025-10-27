# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dynamic MCP tools (elevated) using server_url."""

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
def server_for_dynamic_elevated(monkeypatch_module):
    """vLLM server WITHOUT static --tool-server but WITH elevation config."""
    args = ["--enforce-eager"]  # No --tool-server!

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
        m.setenv("GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "memory")  # Elevate memory tool
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client_for_dynamic_elevated(server_for_dynamic_elevated):
    async with server_for_dynamic_elevated.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_dynamic_elevated(
    client_for_dynamic_elevated, memory_mcp_server, model_name: str
):
    """Test Memory MCP tool with dynamic per-request registration (elevated).

    This tests the dynamic MCP server registration flow where:
    1. No static --tool-server is configured
    2. Tool server is registered dynamically via server_url in the request
    3. Tool IS in GPT_OSS_SYSTEM_TOOL_MCP_LABELS (elevated)
    4. Tool calls/responses should be on analysis channel
    """
    server_url, port = memory_mcp_server

    response = await client_for_dynamic_elevated.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store and memory.retrieve tools. "
            "Never simulate tool execution. Call the tool using json "
            "on the analysis channel like a normal system tool."
        ),
        input=(
            "Store the key 'dynamic_elevated_key' with value "
            "'dynamic_elevated_value' and retrieve it"
        ),
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                "server_url": server_url,  # Dynamic registration via server_url!
                "headers": {"x-memory-id": "test-session-dynamic-elevated"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0

    # Verify input messages: Should have system message with tool, NO developer message
    developer_messages = [
        msg for msg in response.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_messages) == 0, (
        "No developer message expected for elevated tools"
    )

    # Verify output messages: Tool calls and responses on analysis channel
    verify_tool_on_channel(response.output_messages, "memory.", "analysis")

    # Verify functional correctness
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    output_text += content_item.text
    assert (
        "dynamic_elevated_value" in output_text.lower()
        or "successfully" in output_text.lower()
    )
