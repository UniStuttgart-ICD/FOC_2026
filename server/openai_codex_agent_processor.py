"""Pipecat processor that runs OpenAI Codex OAuth agent against the robot MCP server."""

from __future__ import annotations

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
from prompts import SYSTEM_PROMPT


class OpenAICodexAgentProcessor(FrameProcessor):
    """Routes user turns through OpenAI Agents SDK with Pi Codex OAuth credentials."""

    def __init__(self, mcp_server_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._credential_store = PiCodexCredentialStore()
        self._agent = None
        self._mcp_server = None
        self._run_config = None
        self._previous_response_id: str | None = None
        self._model_logged = False

    async def connect(self):
        """Initialize the OpenAI agent lazily."""
        await self._ensure_agent()

    async def disconnect(self):
        """Shut down the MCP connection."""
        if self._mcp_server is not None:
            await self._mcp_server.cleanup()
            self._mcp_server = None
        self._agent = None
        self._run_config = None
        self._previous_response_id = None
        logger.info("OpenAI Codex agent disconnected")

    async def _ensure_agent(self) -> None:
        if self._agent is not None:
            return

        from agents import Agent, OpenAIProvider, RunConfig
        from agents.mcp import MCPServerStreamableHttp
        from openai import AsyncOpenAI

        credentials = self._credential_store.get_credentials()
        headers = {}
        if credentials.account_id:
            headers["ChatGPT-Account-ID"] = credentials.account_id

        openai_client = AsyncOpenAI(
            api_key=credentials.access,
            default_headers=headers or None,
        )
        provider = OpenAIProvider(openai_client=openai_client)
        self._mcp_server = MCPServerStreamableHttp(
            {"url": self._mcp_server_url},
            name="robot",
        )
        await self._mcp_server.connect()
        self._agent = Agent(
            name="Pi robot voice agent",
            instructions=SYSTEM_PROMPT,
            model=self._model,
            mcp_servers=[self._mcp_server],
        )
        self._run_config = RunConfig(model_provider=provider, tracing_disabled=True)
        logger.info("OpenAI Codex agent connected")

    async def _process_with_agent(self, user_text: str):
        try:
            await self._ensure_agent()
        except CodexAuthError as exc:
            logger.error(f"OpenAI Codex OAuth error: {exc}")
            await self.push_frame(LLMTextFrame(text=str(exc)))
            return
        except Exception as exc:
            logger.error(f"OpenAI Codex agent connection error: {exc}")
            await self.push_frame(LLMTextFrame(text="I can't reach the robot control server right now."))
            return

        from agents import Runner

        try:
            result = await Runner.run(
                self._agent,
                user_text,
                max_turns=3,
                run_config=self._run_config,
                previous_response_id=self._previous_response_id,
            )
            response_id = getattr(result, "last_response_id", None)
            if isinstance(response_id, str):
                self._previous_response_id = response_id

            if not self._model_logged:
                logger.info(f"OpenAI Codex model: {self._model}")
                self._model_logged = True

            text = str(getattr(result, "final_output", "") or "").strip()
            await self.push_frame(LLMTextFrame(text=text or "I completed the action but have nothing to report."))
        except Exception as exc:
            logger.error(f"OpenAI Codex agent error: {exc}")
            await self.push_frame(LLMTextFrame(text="I encountered an error. Please try again."))

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
            return

        await self.push_frame(frame, direction)


def _latest_user_text(frame: LLMContextFrame) -> str | None:
    messages = frame.context.messages if frame.context else []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    return None
