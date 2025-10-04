"""Utilities to run the MCP orchestrator inside a Streamlit session."""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.app.interfaces.mcp_agent import (
    SERVER_SCRIPT,
    PYTHON_EXECUTABLE,
    OrchestratorAgent,
    SpecializedAgent,
    build_llm,
    create_tools_from_mcp,
    group_tools_by_domain,
)


@dataclass
class MCPServiceState:
    """Holds runtime dependencies required for servicing requests."""

    orchestrator: OrchestratorAgent
    client_session: ClientSession


class MCPOrchestratorService:
    """Background helper that keeps a persistent MCP session alive."""

    def __init__(self, *, server_script=SERVER_SCRIPT, python_executable: Optional[str] = None) -> None:
        if not server_script.exists():
            raise FileNotFoundError(f"Не найден MCP сервер по пути {server_script}")

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._lock = threading.Lock()
        self._state: Optional[MCPServiceState] = None
        self._server_script = server_script
        self._python_executable = python_executable or PYTHON_EXECUTABLE
        self._stdio_ctx = None
        self._session_ctx = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def ensure_started(self) -> None:
        if self._state is not None:
            return

        with self._lock:
            if self._state is not None:
                return

            future = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
            self._state = future.result()

    async def _async_init(self) -> MCPServiceState:
        llm = build_llm()

        server_params = StdioServerParameters(
            command=self._python_executable,
            args=[str(self._server_script)],
            env=os.environ.copy(),
        )

        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        session = await self._session_ctx.__aenter__()
        await session.initialize()

        tools = await create_tools_from_mcp(session)
        if not tools:
            raise RuntimeError("MCP сервер не предоставил ни одного инструмента")

        tools_by_domain = group_tools_by_domain(tools)
        orchestrator = OrchestratorAgent(llm)

        for domain, domain_tools in tools_by_domain.items():
            if not domain_tools:
                continue
            agent = SpecializedAgent(domain, domain_tools, llm)
            orchestrator.add_agent(agent)

        return MCPServiceState(orchestrator=orchestrator, client_session=session)

    def process_request(self, user_input: str) -> str:
        self.ensure_started()
        assert self._state is not None

        future = asyncio.run_coroutine_threadsafe(
            self._state.orchestrator.process_request(user_input), self._loop
        )
        return future.result()

    def close(self) -> None:
        with self._lock:
            if self._state is None and not self._loop.is_running():
                return

            future = asyncio.run_coroutine_threadsafe(self._async_close(), self._loop)
            future.result()
            self._state = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=1)

    async def _async_close(self) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None

        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None


__all__ = ["MCPOrchestratorService"]
