import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
VOICE_RUNTIME_DIR = SERVER_DIR / "voice_runtime"
PROCESS_TRACE_DIR = SERVER_DIR / "process_trace"

APP_MODULE_ROOTS = {
    "agent_processor_factory",
    "bot",
    "config",
    "metrics",
    "pipeline_builder",
    "prompts",
    "robot_control",
    "wake",
}

PURE_MODULES = {
    "contracts.py",
    "profiles.py",
    "voice_metrics.py",
    "assembly.py",
}

DELETED_LEGACY_ROBOT_MODULES = {
    SERVER_DIR / "robot_mcp_bridge.py",
    VOICE_RUNTIME_DIR / "robot_context.py",
    VOICE_RUNTIME_DIR / "robot_safety.py",
}
DELETED_LEGACY_AGENT_MODULES = {
    SERVER_DIR / "codex_auth.py",
    SERVER_DIR / "codex_backend_client.py",
    SERVER_DIR / "codex_langchain_auth.py",
    SERVER_DIR / "codex_streaming_model.py",
    SERVER_DIR / "openai_codex_agent_processor.py",
}
DELETED_LEGACY_VOICE_PROVIDER_MODULES = {
    SERVER_DIR / "providers.py",
}
PURE_MODULE_FORBIDDEN_ROOTS = {
    "agents",
    "dotenv",
    "mcp",
    "openai",
    "pipecat",
}
PROCESS_TRACE_FORBIDDEN_ROOTS = PURE_MODULE_FORBIDDEN_ROOTS | {
    "agent_control",
    "langchain",
    "langchain_anthropic",
    "langchain_core",
    "langchain_google_genai",
    "langchain_openai",
    "langgraph",
    "langgraph_robot_agent",
    "langchain_agent_processor",
    "robot_control",
    "voice_runtime",
}
PURE_PROCESS_TRACE_MODULES = {
    "__init__.py",
    "context.py",
    "jsonl.py",
    "records.py",
    "trace.py",
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


def test_process_trace_core_modules_do_not_import_runtime_layers():
    for name in PURE_PROCESS_TRACE_MODULES:
        path = PROCESS_TRACE_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PROCESS_TRACE_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports runtime layer module(s): {sorted(forbidden)}"


def test_legacy_robot_modules_are_not_left_in_old_locations():
    for path in DELETED_LEGACY_ROBOT_MODULES:
        assert not path.exists(), f"Legacy robot module still exists: {path}"


def test_legacy_codex_oauth_modules_are_removed():
    for path in DELETED_LEGACY_AGENT_MODULES:
        assert not path.exists(), f"Legacy Codex OAuth module still exists: {path}"


def test_legacy_voice_provider_module_is_not_left_at_app_root():
    for path in DELETED_LEGACY_VOICE_PROVIDER_MODULES:
        assert not path.exists(), f"Legacy Voice Providers module still exists: {path}"
