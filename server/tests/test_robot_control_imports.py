import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
ROBOT_CONTROL_DIR = SERVER_DIR / "robot_control"

PURE_ROBOT_CONTROL_MODULES = {"call_validation.py", "context.py", "task_policy.py"}
ROBOT_CONTROL_FORBIDDEN_ROOTS = {
    "agent_control",
    "pipecat",
    "voice_runtime",
}
PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS = ROBOT_CONTROL_FORBIDDEN_ROOTS | {
    "agents",
    "codex_auth",
    "codex_backend_client",
    "langgraph",
    "mcp",
    "openai",
    "pipeline_builder",
    "providers",
    "robot_mcp_bridge",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_robot_control_modules_do_not_import_voice_runtime_or_agent_control() -> None:
    for path in ROBOT_CONTROL_DIR.glob("*.py"):
        imported = _import_roots(path)
        forbidden = imported & ROBOT_CONTROL_FORBIDDEN_ROOTS
        assert not forbidden, f"{path.name} imports forbidden module(s): {sorted(forbidden)}"


def test_pure_robot_control_modules_do_not_import_runtime_adapters() -> None:
    for name in PURE_ROBOT_CONTROL_MODULES:
        path = ROBOT_CONTROL_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports forbidden module(s): {sorted(forbidden)}"
