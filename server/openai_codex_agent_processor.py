"""Pipecat processor that runs OpenAI Codex OAuth against the robot MCP server."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

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

from codex_auth import CodexAuthError, PiCodexCredentialStore
from codex_backend_client import CodexBackendClient, CodexBackendError, CodexResponseResult
from prompts import SYSTEM_PROMPT
from robot_mcp_bridge import RobotMCPBridge, RobotMCPError

MAX_CODEX_TOOL_TURNS = 3


class OpenAICodexAgentProcessor(FrameProcessor):
    """Routes user turns through ChatGPT's Codex backend with Pi Codex OAuth credentials."""

    def __init__(
        self,
        mcp_server_url: str,
        model: str,
        *,
        credential_store: PiCodexCredentialStore | None = None,
        backend_client: CodexBackendClient | None = None,
        tool_bridge: RobotMCPBridge | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._credential_store = credential_store or PiCodexCredentialStore()
        self._backend_client = backend_client
        self._tool_bridge = tool_bridge
        self._owns_backend_client = backend_client is None
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False

    async def connect(self) -> None:
        """Initialize Codex backend and robot MCP clients lazily."""
        await self._ensure_connected()

    async def disconnect(self) -> None:
        """Shut down Codex backend and MCP resources."""
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        if self._backend_client is not None and (self._connected or not self._owns_backend_client):
            await self._backend_client.close()
        self._backend_client = None
        self._tool_bridge = None
        self._connected = False
        logger.info("OpenAI Codex backend agent disconnected")

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._backend_client is None:
            self._backend_client = CodexBackendClient()
        if self._tool_bridge is None:
            self._tool_bridge = RobotMCPBridge(self._mcp_server_url)
        await self._tool_bridge.connect()
        self._connected = True
        logger.info("OpenAI Codex backend agent connected")

    async def _process_with_agent(self, user_text: str, input_items: list[dict[str, Any]]) -> None:
        try:
            await self._ensure_connected()
            credentials = self._credential_store.get_credentials()
        except CodexAuthError as exc:
            logger.error(f"OpenAI Codex OAuth error: {exc}")
            await self.push_frame(LLMTextFrame(text=str(exc)))
            return
        except Exception as exc:
            logger.error(f"OpenAI Codex agent connection error: {exc}")
            await self.push_frame(LLMTextFrame(text="I can't reach the robot control server right now."))
            return

        backend_client = self._backend_client
        tool_bridge = self._tool_bridge
        if backend_client is None or tool_bridge is None:
            await self.push_frame(LLMTextFrame(text="I can't reach the robot control server right now."))
            return

        if not self._model_logged:
            logger.info(f"OpenAI Codex model: {self._model}")
            self._model_logged = True

        if not input_items:
            input_items = [_user_input_item(user_text)]
        tools = tool_bridge.function_tools()

        try:
            result = await backend_client.create_response(
                credentials,
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input_items=input_items,
                tools=tools,
            )
            result = await self._run_tool_loop(
                result=result,
                input_items=input_items,
                credentials=credentials,
                backend_client=backend_client,
                tool_bridge=tool_bridge,
                tools=tools,
            )
            await self.push_frame(
                LLMTextFrame(text=result.text or "I completed the action but have nothing to report.")
            )
        except CodexBackendError as exc:
            logger.error(f"OpenAI Codex backend error: {exc}")
            await self.push_frame(LLMTextFrame(text="I encountered an error. Please try again."))
        except Exception as exc:
            logger.error(f"OpenAI Codex agent error: {exc}")
            await self.push_frame(LLMTextFrame(text="I encountered an error. Please try again."))

    async def _run_tool_loop(
        self,
        *,
        result: CodexResponseResult,
        input_items: list[dict[str, Any]],
        credentials: Any,
        backend_client: CodexBackendClient,
        tool_bridge: RobotMCPBridge,
        tools: list[dict[str, Any]],
    ) -> CodexResponseResult:
        turns = 0
        while result.tool_calls and turns < MAX_CODEX_TOOL_TURNS:
            turns += 1
            input_items.extend(result.output_items)
            for tool_call in result.tool_calls:
                try:
                    output = await tool_bridge.call_tool(tool_call.name, tool_call.arguments)
                except RobotMCPError as exc:
                    output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": output,
                    }
                )
            result = await backend_client.create_response(
                credentials,
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input_items=input_items,
                tools=tools,
            )
        return result

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
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
                await self._process_with_agent(user_text, _input_items_from_context(frame))
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


def _input_items_from_context(frame: LLMContextFrame) -> list[dict[str, Any]]:
    messages = frame.context.messages if frame.context else []
    items: list[dict[str, Any]] = []
    assistant_index = 0
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(msg)
        if not text:
            continue
        if role == "user":
            items.append(_user_input_item(text))
        else:
            assistant_index += 1
            items.append(_assistant_output_item(text, assistant_index))
    return items


def _user_input_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant_output_item(text: str, index: int) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "status": "completed",
        "id": f"history-assistant-{index}",
    }


def _message_text(msg: Mapping[str, Any]) -> str | None:
    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts) if parts else None
    return None


def _latest_user_text(frame: LLMContextFrame) -> str | None:
    messages = frame.context.messages if frame.context else []
    for msg in reversed(messages):
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        text = _message_text(msg)
        if text:
            return text
    return None
