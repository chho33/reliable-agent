"""MCP server (stdio) exposing the support tools. Built for MCP Python SDK 2.x.

Run: ``python -m tools.mcp_server``. Each server process owns one fresh seeded
backend/session, so one stdio connection == one episode's world.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from backends.mock import MockBackend
from tools.registry import TOOLS_BY_NAME, ToolDef
from tools.toolbox import Toolbox


def build_server(toolbox: Toolbox) -> MCPServer:
    server = MCPServer("reliable-agent-tools")

    def register(defn: ToolDef):
        # Output is returned as JSON text (structured_output=False) so the in-process and
        # MCP clients can share one parsing path and one result shape.
        return server.tool(name=defn.name, description=defn.description, meta=defn.meta(), structured_output=False)

    @register(TOOLS_BY_NAME["verify_identity"])
    def verify_identity(email: str, order_id: str) -> dict[str, Any]:
        return toolbox.call("verify_identity", {"email": email, "order_id": order_id})

    @register(TOOLS_BY_NAME["get_order"])
    def get_order(order_id: str) -> dict[str, Any]:
        return toolbox.call("get_order", {"order_id": order_id})

    @register(TOOLS_BY_NAME["list_orders"])
    def list_orders(customer_email: str) -> dict[str, Any]:
        return toolbox.call("list_orders", {"customer_email": customer_email})

    @register(TOOLS_BY_NAME["update_address"])
    def update_address(order_id: str, new_address: str) -> dict[str, Any]:
        return toolbox.call("update_address", {"order_id": order_id, "new_address": new_address})

    @register(TOOLS_BY_NAME["refund_order"])
    def refund_order(order_id: str, amount: float, reason: str) -> dict[str, Any]:
        return toolbox.call("refund_order", {"order_id": order_id, "amount": amount, "reason": reason})

    @register(TOOLS_BY_NAME["log_crm_note"])
    def log_crm_note(note: str) -> dict[str, Any]:
        return toolbox.call("log_crm_note", {"note": note})

    @register(TOOLS_BY_NAME["transfer_to_human"])
    def transfer_to_human(reason: str, priority: str) -> dict[str, Any]:
        return toolbox.call("transfer_to_human", {"reason": reason, "priority": priority})

    return server


def main() -> None:
    build_server(Toolbox(MockBackend())).run("stdio")


if __name__ == "__main__":
    main()
