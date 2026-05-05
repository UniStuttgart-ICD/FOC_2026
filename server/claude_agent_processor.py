"""Claude Agent SDK backend Adapter for Agent Turn processing."""

from __future__ import annotations

import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from loguru import logger

from prompts import SYSTEM_PROMPT
from voice_runtime.agent_turn import AgentTurnInput


class ClaudeAgentProcessor:
    """Runs Agent Turns through a persistent ClaudeSDKClient.

    Direct Claude MCP access is prompt-only Robot Safety coverage; this Adapter does
    not locally enforce robot_safety before SDK-managed MCP calls.
    """

    def __init__(self, mcp_server_url: str, model: str | None = None):
        self._mcp_server_url = mcp_server_url
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        self._client: ClaudeSDKClient | None = None
        self._model_logged = False

    async def connect(self) -> None:
        if self._client:
            return
        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=SYSTEM_PROMPT,
            include_partial_messages=True,
            max_turns=3,
            effort="low",
            mcp_servers={
                "robot": {
                    "type": "http",
                    "url": self._mcp_server_url,
                }
            },
            allowed_tools=["mcp__robot__*"],
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        logger.info("ClaudeSDKClient connected")

        try:
            status = await self._client.get_mcp_status()
            for server in status.get("mcpServers", []):
                if server.get("name") == "robot":
                    if server.get("status") == "connected":
                        logger.info("Robot MCP server connected and ready")
                    else:
                        logger.warning(f"Robot MCP server not ready: {server.get('status')}")
        except Exception as exc:
            logger.warning(f"Could not verify MCP status at startup: {exc}")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None
            logger.info("ClaudeSDKClient disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        user_text = turn.user_text
        logger.info(f"User said: {user_text}")
        if not self._client:
            yield "Agent not connected."
            return

        if not await self._ensure_mcp_connected():
            yield "I can't reach the robot control server right now."
            return

        await self._client.query(user_text)

        has_text = False
        try:
            async for message in self._client.receive_response():
                event = getattr(message, "event", None)
                if isinstance(event, dict):
                    delta = event.get("delta", {})
                    if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        text = delta.get("text")
                        if text:
                            has_text = True
                            yield text

                elif isinstance(message, AssistantMessage):
                    if not self._model_logged and message.model:
                        logger.info(f"Claude model: {message.model}")
                        self._model_logged = True
                    if not has_text:
                        for block in message.content:
                            text = getattr(block, "text", None)
                            if getattr(block, "type", None) == "text" and isinstance(text, str) and text:
                                has_text = True
                                yield text

                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        logger.error("Claude Agent SDK execution error")
                        yield "I hit an error while talking to the robot."
                        return
                    if not has_text and message.result:
                        has_text = True
                        yield str(message.result)
        except Exception as exc:
            logger.error(f"Claude Agent SDK error: {exc}")
            yield "I encountered an error. Please try again."
            return

    async def _ensure_mcp_connected(self) -> bool:
        if not self._client:
            return False
        try:
            status = await self._client.get_mcp_status()
            for server in status.get("mcpServers", []):
                if server.get("name") == "robot" and server.get("status") != "connected":
                    logger.warning("Robot MCP disconnected, reconnecting...")
                    await self._client.reconnect_mcp_server("robot")
                    return True
            return True
        except Exception as exc:
            logger.error(f"MCP status check failed: {exc}")
            return False
