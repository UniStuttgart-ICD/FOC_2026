from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from vizor_mcp.attention import GazeAttentionTracker
from vizor_mcp.ros_client import RoslibpyVizorRosTransport, VizorSensorTransport
from vizor_mcp.tools import VizorMcpTools

MaxAgeSeconds = Annotated[
    float,
    Field(description="Maximum sensor age in seconds before a field is marked stale."),
]
IncludeRaw = Annotated[
    bool,
    Field(description="Include raw ROS/Vizor payload details useful for diagnostics."),
]
AttentionWindowSeconds = Annotated[
    float,
    Field(description="Rolling gaze window in seconds for attention ranking."),
]


def build_tools(
    *,
    transport: VizorSensorTransport | None = None,
    host: str = "localhost",
    port: int = 9090,
    enable_holo1_tracking_on_startup: bool = False,
    holo1_tracking_keepalive_s: float = 10.0,
    attention_window_s: float = 8.0,
) -> VizorMcpTools:
    attention = GazeAttentionTracker(window_s=attention_window_s)
    if transport is not None:
        return VizorMcpTools.with_transport(transport, attention=attention)

    real_transport = RoslibpyVizorRosTransport(host=host, port=port)
    real_transport.connect()
    if enable_holo1_tracking_on_startup:
        real_transport.start_holo1_tracking_keepalive(interval_s=holo1_tracking_keepalive_s)
    return VizorMcpTools.with_transport(real_transport, attention=attention)


def build_mcp(*, tools: VizorMcpTools, host: str = "127.0.0.1", port: int = 8001) -> FastMCP:
    mcp = FastMCP("VizorSensingServer", host=host, port=port)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    def vizor_get_sensor_context(
        max_age_s: MaxAgeSeconds = 2.0,
        include_raw: IncludeRaw = False,
        attention_window_s: AttentionWindowSeconds = 8.0,
    ) -> dict[str, Any]:
        """Read the latest Vizor/HoloLens user sensing context from ROSBridge.

        Returns gaze target, user pose, and manual target as one coherent snapshot with
        per-field freshness. This is read-only and does not publish robot commands.
        """
        return tools.get_sensor_context(
            max_age_s=max_age_s,
            include_raw=include_raw,
            attention_window_s=attention_window_s,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    def vizor_get_status() -> dict[str, Any]:
        """Read Vizor MCP ROSBridge connection status and configured sensing topics."""
        return tools.get_status()

    return mcp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vizor ROS 1 user-sensing MCP server")
    parser.add_argument("--rosbridge-host", default="localhost")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8001)
    parser.add_argument("--attention-window-s", type=float, default=8.0)
    parser.add_argument("--enable-holo1-tracking-on-startup", action="store_true")
    parser.add_argument("--holo1-tracking-keepalive-s", type=float, default=10.0)
    args = parser.parse_args()

    tools = build_tools(
        host=args.rosbridge_host,
        port=args.rosbridge_port,
        enable_holo1_tracking_on_startup=args.enable_holo1_tracking_on_startup,
        holo1_tracking_keepalive_s=args.holo1_tracking_keepalive_s,
        attention_window_s=args.attention_window_s,
    )
    mcp = build_mcp(tools=tools, host=args.http_host, port=args.http_port)
    mcp.run(transport=args.transport)
