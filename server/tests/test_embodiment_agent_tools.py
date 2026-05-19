from __future__ import annotations

from unittest.mock import Mock

from agent_control.langgraph_robot_agent import LangGraphRobotAgent
from embodiment.animations import EmbodimentAnimationController, FakeAnimationRosTransport
from process_trace import NoopProcessTracer
from robot_control.context import RobotContextStore
from voice_runtime.profiles import EmbodimentProfile


class FakeToolBridge:
    def function_tools(self) -> list[dict[str, object]]:
        return []


def test_embodiment_tools_are_model_visible_when_controller_is_enabled() -> None:
    controller = EmbodimentAnimationController(
        EmbodimentProfile(enabled=True),
        transport=FakeAnimationRosTransport(),
    )
    agent = LangGraphRobotAgent(
        model=Mock(),
        tool_bridge=FakeToolBridge(),
        robot_context=RobotContextStore(),
        embodiment_controller=controller,
        tracer=NoopProcessTracer(),
    )

    tool_names = {tool["name"] for tool in agent._model_visible_tools([])}

    assert "embodiment_set_animation" in tool_names
    assert "embodiment_fake_death" in tool_names
