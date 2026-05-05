"""OpenAI Codex OAuth backend Adapter for Agent Turn processing."""

from __future__ import annotations

from typing import Any

from loguru import logger

from codex_auth import CodexAuthError, PiCodexCredentialStore
from codex_backend_client import CodexBackendClient, CodexBackendError
from langgraph_robot_agent import LangGraphRobotAgent
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.agent_turn import AgentTurnInput


class OpenAICodexAgentProcessor:
    """Runs Agent Turns through ChatGPT's Codex backend with Pi OAuth credentials."""

    def __init__(
        self,
        mcp_server_url: str,
        model: str,
        *,
        credential_store: Any | None = None,
        backend_client: Any | None = None,
        tool_bridge: Any | None = None,
        reasoning_effort: str | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._credential_store = credential_store or PiCodexCredentialStore()
        self._backend_client = backend_client
        self._tool_bridge = tool_bridge
        self._owns_backend_client = backend_client is None
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()
        self._thread_id = f"openai-codex-agent-{id(self)}"
        self._graph_agent: LangGraphRobotAgent | None = None
        self._graph_backend_client: Any | None = None
        self._graph_tool_bridge: Any | None = None

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        if self._backend_client is not None and (self._connected or not self._owns_backend_client):
            await self._backend_client.close()
        self._backend_client = None
        self._tool_bridge = None
        self._graph_agent = None
        self._graph_backend_client = None
        self._graph_tool_bridge = None
        self._connected = False
        logger.info("OpenAI Codex backend agent disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        logger.info(f"User said: {turn.user_text}")
        try:
            await self._ensure_connected()
            credentials = self._credential_store.get_credentials()
        except CodexAuthError as exc:
            logger.error(f"OpenAI Codex OAuth error: {exc}")
            yield str(exc)
            return
        except Exception as exc:
            logger.error(f"OpenAI Codex agent connection error: {exc}")
            yield "I can't reach the robot control server right now."
            return

        backend_client = self._backend_client
        tool_bridge = self._tool_bridge
        if backend_client is None or tool_bridge is None:
            yield "I can't reach the robot control server right now."
            return

        if not self._model_logged:
            logger.info(f"OpenAI Codex model: {self._model}")
            self._model_logged = True

        graph = self._graph_agent_for(backend_client, tool_bridge)
        try:
            yield await graph.run_turn(turn, credentials=credentials)
        except CodexAuthError as exc:
            logger.error(f"OpenAI Codex OAuth error: {exc}")
            yield str(exc)
        except CodexBackendError as exc:
            logger.error(f"OpenAI Codex backend error: {exc}")
            yield "I encountered an error. Please try again."
        except Exception as exc:
            logger.error(f"OpenAI Codex agent error: {exc}")
            yield "I encountered an error. Please try again."

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

    def _graph_agent_for(self, backend_client: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_backend_client is not backend_client
            or self._graph_tool_bridge is not tool_bridge
        ):
            self._graph_agent = LangGraphRobotAgent(
                model=self._model,
                credential_store=self._credential_store,
                backend_client=backend_client,
                tool_bridge=tool_bridge,
                robot_context=self._robot_context,
                thread_id=self._thread_id,
                reasoning_effort=self._reasoning_effort,
            )
            self._graph_backend_client = backend_client
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent
