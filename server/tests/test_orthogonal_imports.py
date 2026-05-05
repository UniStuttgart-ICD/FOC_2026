import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
VOICE_RUNTIME_DIR = SERVER_DIR / "voice_runtime"

APP_MODULE_ROOTS = {
    "agent_processor_factory",
    "bot",
    "claude_agent_processor",
    "codex_auth",
    "codex_backend_client",
    "config",
    "metrics",
    "openai_codex_agent_processor",
    "pipeline_builder",
    "prompts",
    "providers",
    "robot_mcp_bridge",
    "wake",
}

PURE_MODULES = {
    "contracts.py",
    "profiles.py",
    "robot_safety.py",
    "voice_metrics.py",
    "assembly.py",
}
PURE_MODULE_FORBIDDEN_ROOTS = {
    "agents",
    "claude_agent_sdk",
    "dotenv",
    "mcp",
    "openai",
    "pipecat",
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


def test_voice_runtime_modules_do_not_import_app_modules():
    for path in VOICE_RUNTIME_DIR.glob("*.py"):
        imported = _import_roots(path)
        forbidden = imported & APP_MODULE_ROOTS
        assert not forbidden, f"{path.name} imports app-specific module(s): {sorted(forbidden)}"


def test_pure_voice_runtime_modules_do_not_import_runtime_adapters():
    for name in PURE_MODULES:
        path = VOICE_RUNTIME_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PURE_MODULE_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports adapter-specific module(s): {sorted(forbidden)}"
