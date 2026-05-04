"""Pipecat processor that runs Claude Agent SDK against the robot MCP server."""

import os
from collections.abc import Mapping

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from prompts import SYSTEM_PROMPT


class ClaudeAgentProcessor(FrameProcessor):
    """Routes user turns through a persistent ClaudeSDKClient with robot MCP tools.

    Maintains a single Claude session for the lifetime of the WebRTC connection.
    The SDK handles conversation history natively.
    """

    def __init__(self, mcp_server_url: str, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        self._client: ClaudeSDKClient | None = None
        self._model_logged = False

    async def connect(self):
        """Initialize the persistent Claude SDK client."""
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

        # Pre-warm: verify robot MCP is reachable
        try:
            status = await self._client.get_mcp_status()
            for server in status.get("mcpServers", []):
                if server.get("name") == "robot":
                    if server.get("status") == "connected":
                        logger.info("Robot MCP server connected and ready")
                    else:
                        logger.warning(f"Robot MCP server not ready: {server.get('status')}")
        except Exception as e:
            logger.warning(f"Could not verify MCP status at startup: {e}")

    async def disconnect(self):
        """Shut down the Claude SDK client."""
        if self._client:
            await self._client.disconnect()
            self._client = None
            logger.info("ClaudeSDKClient disconnected")

    async def _ensure_mcp_connected(self) -> bool:
        """Check MCP status and reconnect if needed."""
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
        except Exception as e:
            logger.error(f"MCP status check failed: {e}")
            return False

    async def _process_with_agent(self, user_text: str):
        """Send user text to Claude and stream response frames directly."""
        if not self._client:
            await self.push_frame(LLMTextFrame(text="Agent not connected."))
            return

        if not await self._ensure_mcp_connected():
            await self.push_frame(LLMTextFrame(text="I can't reach the robot control server right now."))
            return

        await self._client.query(user_text)

        has_text = False
        try:
            async for message in self._client.receive_response():
                event = getattr(message, "event", None)
                if isinstance(event, dict):
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            has_text = True
                            await self.push_frame(LLMTextFrame(text=delta["text"]))

                elif isinstance(message, AssistantMessage):
                    if not self._model_logged and message.model:
                        logger.info(f"Claude model: {message.model}")
                        self._model_logged = True
                    # Fallback: use AssistantMessage text only if no streaming happened
                    if not has_text:
                        for block in message.content:
                            text = getattr(block, "text", None)
                            if getattr(block, "type", None) == "text" and isinstance(text, str) and text:
                                has_text = True
                                await self.push_frame(LLMTextFrame(text=text))

                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        logger.error("Claude Agent SDK execution error")
                        await self.push_frame(LLMTextFrame(text="I hit an error while talking to the robot."))
                        return
                    if not has_text and message.result:
                        await self.push_frame(LLMTextFrame(text=str(message.result)))

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            await self.push_frame(LLMTextFrame(text="I encountered an error. Please try again."))
            return

        if not has_text:
            await self.push_frame(LLMTextFrame(text="I completed the action but have nothing to report."))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame):
            user_text = _latest_user_text(frame)

            if user_text:
                logger.info(f"User said: {user_text}")
                await self.push_frame(LLMFullResponseStartFrame())
                await self._process_with_agent(user_text)
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


def _latest_user_text(frame: LLMContextFrame) -> str | None:
    messages = frame.context.messages if frame.context else []
    for msg in reversed(messages):
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    continue
                text = part.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None
