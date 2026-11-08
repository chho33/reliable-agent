"""ToolClient interface and its two implementations.

* ``InProcessToolClient``: direct calls into a ``Toolbox`` (default for evals).
* ``MCPToolClient``: talks MCP via the SDK's ``Client``. It runs the async client on a
  dedicated event-loop thread so the (synchronous) agent can use it. ``target`` is
  anything ``mcp.Client`` accepts: ``StdioServerParameters`` for a subprocess, or an
  ``MCPServer`` instance for an in-memory connection (used in tests).
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any, Protocol

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from tools.registry import TOOL_DEFS, ToolDef
from tools.toolbox import Toolbox, err


class ToolClient(Protocol):
    def list_tools(self) -> list[ToolDef]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class InProcessToolClient:
    def __init__(self, toolbox: Toolbox) -> None:
        self.toolbox = toolbox

    def list_tools(self) -> list[ToolDef]:
        return list(TOOL_DEFS)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.toolbox.call(name, dict(arguments))

    def close(self) -> None:
        pass


def stdio_server_params() -> StdioServerParameters:
    """Launch ``python -m tools.mcp_server`` with the current interpreter."""
    return StdioServerParameters(command=sys.executable, args=["-m", "tools.mcp_server"])


class MCPToolClient:
    def __init__(self, target: Any = None, *, timeout: float = 30.0) -> None:
        self._target = target if target is not None else stdio_server_params()
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._client: Client | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="mcp-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("MCP client did not connect in time")
        if self._error is not None:
            raise RuntimeError(f"MCP client failed to connect: {self._error!r}") from self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            # Enter and exit the client in this single task (anyio cancel-scope rule).
            async with Client(self._target) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # surface connection failures to the constructor
            self._error = exc
            self._ready.set()

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self._timeout)

    def list_tools(self) -> list[ToolDef]:
        assert self._client is not None
        result = self._submit(self._client.list_tools())
        tools = []
        for t in result.tools:
            meta = t.meta or {}
            tools.append(ToolDef(
                name=t.name, description=t.description or "", input_schema=t.input_schema,
                access=meta.get("access", "read"), requires_confirmation=bool(meta.get("requires_confirmation", False)),
            ))
        return tools

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        result = self._submit(self._client.call_tool(name, dict(arguments)))
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            return err("tool_error", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return err("bad_tool_output", text[:200])

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(self._timeout)

    def __enter__(self) -> MCPToolClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
