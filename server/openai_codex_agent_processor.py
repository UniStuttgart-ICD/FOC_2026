"""OpenAI Codex OAuth backend Adapter for Agent Turn processing."""

from __future__ import annotations

from typing import Any

from codex_oauth.exceptions import CodexOAuthError
from langchain_codex_oauth import ChatCodexOAuth
from loguru import logger

from codex_auth import CodexAuthError, PiCodexCredentialStore
from codex_langchain_auth import PiLangChainCodexAuthStore
from codex_streaming_model import StreamingAinvokeChatModel
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
        chat_model: Any | None = None,
        tool_bridge: Any | None = None,
        reasoning_effort: str | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._credential_store = credential_store or PiCodexCredentialStore()
        self._backend_client = backend_client
        self._chat_model = chat_model
        self._tool_bridge = tool_bridge
        self._owns_backend_client = backend_client is None
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()
        self._thread_id = f"openai-codex-agent-{id(self)}"
        self._graph_agent: LangGraphRobotAgent | None = None
        self._graph_chat_model: Any | None = None
        self._graph_tool_bridge: Any | None = None

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        if self._backend_client is not None and (self._connected or not self._owns_backend_client):
            await self._backend_client.close()
        self._backend_client = None
        self._chat_model = None
        self._tool_bridge = None
        self._graph_agent = None
        self._graph_chat_model = None
        self._graph_tool_bridge = None
        self._connected = False
        logger.info("OpenAI Codex backend agent disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        logger.info("User said: {}", turn.user_text)
        try:
            await self._ensure_connected()
            self._credential_store.get_credentials()
        except CodexAuthError as exc:
            logger.error("OpenAI Codex OAuth error: {}", exc)
            yield str(exc)
            return
        except Exception as exc:
            logger.error("OpenAI Codex agent connection error: {}", exc)
            yield "I can't reach the robot control server right now."
            return

        tool_bridge = self._tool_bridge
        if tool_bridge is None:
            yield "I can't reach the robot control server right now."
            return

        if not self._model_logged:
            logger.info("OpenAI Codex model: {}", self._model)
            self._model_logged = True

        chat_model = self._chat_model_for_turn()
        graph = self._graph_agent_for(chat_model, tool_bridge)
        try:
            yield await graph.run_turn(turn)
        except (CodexAuthError, CodexOAuthError) as exc:
            logger.error("OpenAI Codex OAuth error: {}", exc)
            yield str(exc)
        except Exception as exc:
            logger.error("OpenAI Codex agent error: {}", exc)
            yield "I encountered an error. Please try again."

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._tool_bridge is None:
            self._tool_bridge = RobotMCPBridge(self._mcp_server_url)
        await self._tool_bridge.connect()
        self._connected = True
        logger.info("OpenAI Codex backend agent connected")

    def _chat_model_for_turn(self) -> Any:
        if self._chat_model is not None:
            return self._chat_model
        self._chat_model = StreamingAinvokeChatModel(
            ChatCodexOAuth(
                model=self._model,
                auth_store=PiLangChainCodexAuthStore(),
                reasoning_effort=self._reasoning_effort,
                text_verbosity="low",
                system_prompt_mode="strict",
            )
        )
        return self._chat_model

    def _graph_agent_for(self, chat_model: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_chat_model is not chat_model
            or self._graph_tool_bridge is not tool_bridge
        ):
            self._graph_agent = LangGraphRobotAgent(
                model=chat_model,
                tool_bridge=tool_bridge,
                robot_context=self._robot_context,
                thread_id=self._thread_id,
            )
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent
