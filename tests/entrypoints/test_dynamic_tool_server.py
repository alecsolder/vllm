# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DynamicToolServer."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai_harmony import ToolDescription, ToolNamespaceConfig

from vllm.entrypoints.tool_server import DynamicToolServer, MCPToolServer


@pytest.fixture
def static_tool_server():
    """Create a mock static tool server."""
    server = Mock(spec=MCPToolServer)
    server.has_namespace = Mock(return_value=False)
    server.get_tool_description = Mock(return_value=None)
    server.new_session = AsyncMock()
    return server


@pytest.fixture
def tool_namespace_config():
    """Create a mock tool namespace config."""
    return ToolNamespaceConfig(
        name="weather",
        description="Weather tools",
        tools=[
            ToolDescription.new(
                name="get_weather",
                description="Get current weather",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )


@pytest.mark.asyncio
async def test_dynamic_tool_server_initialization():
    """Test DynamicToolServer initialization."""
    static_server = Mock()
    server = DynamicToolServer(static_tool_server=static_server)

    assert server.static_server == static_server
    assert len(server.dynamic_tool_descriptions) == 0
    assert len(server.dynamic_urls) == 0


@pytest.mark.asyncio
async def test_dynamic_tool_server_async_context_manager():
    """Test DynamicToolServer as async context manager."""
    server = DynamicToolServer(static_tool_server=None)

    async with server as s:
        assert s is server
        # Manually add something to verify cleanup
        s.dynamic_tool_descriptions["test"] = Mock()

    # After exiting context, should be cleaned up
    assert len(server.dynamic_tool_descriptions) == 0


@pytest.mark.asyncio
async def test_register_dynamic_mcp_server_success(
    static_tool_server, tool_namespace_config
):
    """Test successful registration of a dynamic MCP server."""
    server = DynamicToolServer(static_tool_server=static_tool_server)

    # Mock the list_server_and_tools function
    mock_initialize_response = Mock()
    mock_initialize_response.serverInfo.name = "weather"
    mock_initialize_response.instructions = "Weather service"

    mock_tool = Mock()
    mock_tool.name = "get_weather"
    mock_tool.description = "Get weather"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    mock_list_tools_response = Mock()
    mock_list_tools_response.tools = [mock_tool]

    with patch(
        "vllm.entrypoints.tool_server.list_server_and_tools"
    ) as mock_list, patch(
        "vllm.entrypoints.tool_server.post_process_tools_description"
    ) as mock_post_process:
        mock_list.return_value = (
            mock_initialize_response,
            mock_list_tools_response,
        )
        # Mock post_process to return the same response
        mock_post_process.return_value = mock_list_tools_response

        await server.register_dynamic_mcp_server(
            server_label="weather",
            server_url="http://localhost:8080/sse",
        )

    assert server.has_namespace("weather")
    assert server.get_tool_description("weather") is not None
    assert "weather" in server.dynamic_urls


@pytest.mark.asyncio
async def test_register_dynamic_mcp_server_namespace_collision(
    static_tool_server,
):
    """Test that namespace collision with static server raises error."""
    # Configure static server to already have the namespace
    static_tool_server.has_namespace = Mock(return_value=True)

    server = DynamicToolServer(static_tool_server=static_tool_server)

    with pytest.raises(ValueError, match="namespace already exists"):
        await server.register_dynamic_mcp_server(
            server_label="existing",
            server_url="http://localhost:8080",
        )


@pytest.mark.asyncio
async def test_register_dynamic_mcp_server_connection_failure(
    static_tool_server,
):
    """Test that connection failure raises ConnectionError."""
    server = DynamicToolServer(static_tool_server=static_tool_server)

    with patch(
        "vllm.entrypoints.tool_server.list_server_and_tools"
    ) as mock_list:
        mock_list.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await server.register_dynamic_mcp_server(
                server_label="test",
                server_url="http://invalid:8080",
            )


@pytest.mark.asyncio
async def test_has_namespace_checks_dynamic_first(static_tool_server):
    """Test that has_namespace checks dynamic registry before static."""
    server = DynamicToolServer(static_tool_server=static_tool_server)

    # Add to dynamic registry
    server.dynamic_tool_descriptions["dynamic_tool"] = Mock()

    # Check dynamic tool
    assert server.has_namespace("dynamic_tool")
    # Static server should not be queried
    static_tool_server.has_namespace.assert_not_called()


@pytest.mark.asyncio
async def test_has_namespace_falls_back_to_static(static_tool_server):
    """Test that has_namespace falls back to static server."""
    static_tool_server.has_namespace = Mock(return_value=True)

    server = DynamicToolServer(static_tool_server=static_tool_server)

    assert server.has_namespace("static_tool")
    static_tool_server.has_namespace.assert_called_once_with("static_tool")


@pytest.mark.asyncio
async def test_get_tool_description_checks_dynamic_first(
    static_tool_server, tool_namespace_config
):
    """Test that get_tool_description checks dynamic registry first."""
    server = DynamicToolServer(static_tool_server=static_tool_server)

    # Add to dynamic registry
    server.dynamic_tool_descriptions["dynamic_tool"] = tool_namespace_config

    result = server.get_tool_description("dynamic_tool")
    assert result == tool_namespace_config
    # Static server should not be queried
    static_tool_server.get_tool_description.assert_not_called()


@pytest.mark.asyncio
async def test_get_tool_description_falls_back_to_static(
    static_tool_server, tool_namespace_config
):
    """Test that get_tool_description falls back to static server."""
    static_tool_server.get_tool_description = Mock(
        return_value=tool_namespace_config
    )

    server = DynamicToolServer(static_tool_server=static_tool_server)

    result = server.get_tool_description("static_tool")
    assert result == tool_namespace_config
    static_tool_server.get_tool_description.assert_called_once_with(
        "static_tool"
    )


@pytest.mark.asyncio
async def test_cleanup_clears_dynamic_registrations():
    """Test that cleanup clears all dynamic registrations."""
    server = DynamicToolServer(static_tool_server=None)

    # Add some dynamic registrations
    server.dynamic_tool_descriptions["tool1"] = Mock()
    server.dynamic_tool_descriptions["tool2"] = Mock()
    server.dynamic_urls["tool1"] = "http://localhost:8080"
    server.dynamic_urls["tool2"] = "http://localhost:8081"

    server.cleanup()

    assert len(server.dynamic_tool_descriptions) == 0
    assert len(server.dynamic_urls) == 0


@pytest.mark.asyncio
async def test_new_session_for_dynamic_tool():
    """Test creating a session for a dynamic tool."""
    from contextlib import asynccontextmanager

    server = DynamicToolServer(static_tool_server=None)

    # Register a dynamic URL
    server.dynamic_urls["weather"] = "http://localhost:8080/sse"

    # Mock the sse_client and ClientSession
    mock_session = Mock()
    mock_session.initialize = AsyncMock()

    @asynccontextmanager
    async def mock_sse(**kwargs):
        yield (Mock(), Mock())

    @asynccontextmanager
    async def mock_client(*args):
        yield mock_session

    with patch(
        "vllm.entrypoints.tool_server.sse_client", side_effect=mock_sse
    ), patch(
        "vllm.entrypoints.tool_server.ClientSession", side_effect=mock_client
    ):
        async with server.new_session("weather", "session123") as session:
            assert session == mock_session
            mock_session.initialize.assert_called()


@pytest.mark.asyncio
async def test_new_session_falls_back_to_static():
    """Test that new_session falls back to static server for non-dynamic tools."""
    from contextlib import asynccontextmanager

    mock_static_session = Mock()
    static_tool_server = Mock()

    @asynccontextmanager
    async def mock_new_session(namespace, session_id, headers):
        yield mock_static_session

    static_tool_server.new_session = mock_new_session

    server = DynamicToolServer(static_tool_server=static_tool_server)

    async with server.new_session("static_tool", "session123") as session:
        assert session == mock_static_session


@pytest.mark.asyncio
async def test_new_session_raises_for_unknown_namespace():
    """Test that new_session raises KeyError for unknown namespace."""
    server = DynamicToolServer(static_tool_server=None)

    with pytest.raises(KeyError, match="not found in dynamic or static servers"):
        async with server.new_session("unknown", "session123"):
            pass
