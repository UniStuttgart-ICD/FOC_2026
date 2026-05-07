import ast
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
AGENT_CONTROL_DIR = SERVER_DIR / "agent_control"
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
REQUIRED_AGENT_CONTROL_MODULES = {
    "__init__.py",
    "factory.py",
    "langchain_agent_processor.py",
    "langgraph_robot_agent.py",
    "model_factory.py",
    "prompts.py",
}
DELETED_LEGACY_AGENT_CONTROL_ROOTS = {
    SERVER_DIR / "agent_model_factory.py",
    SERVER_DIR / "agent_processor_factory.py",
    SERVER_DIR / "langchain_agent_processor.py",
    SERVER_DIR / "langgraph_robot_agent.py",
    SERVER_DIR / "prompts.py",
}
DELETED_LEGACY_WAKE_WRAPPERS = {
    SERVER_DIR / "wake" / "transcript_cleanup.py",
    SERVER_DIR / "wake" / "wake_gate.py",
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
AGENT_CONTROL_FORBIDDEN_ROOTS = {
    "bot",
    "config",
    "metrics",
    "pipeline_builder",
    "providers",
    "wake",
    "wake_tuning",
}
AGENT_CONTROL_ALLOWED_VOICE_RUNTIME_MODULES = {
    "voice_runtime.agent_providers",
    "voice_runtime.agent_turn",
    "voice_runtime.profiles",
    "voice_runtime.timing",
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


def _import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _agent_control_python_files() -> list[Path]:
    return sorted(
        path
        for path in AGENT_CONTROL_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_agent_control_python_files_include_nested_modules(
    tmp_path: Path, monkeypatch
) -> None:
    package_dir = tmp_path / "agent_control"
    nested_module = package_dir / "prompt_parts" / "dynamic.py"
    root_module = package_dir / "factory.py"
    nested_module.parent.mkdir(parents=True)
    root_module.write_text("", encoding="utf-8")
    nested_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "AGENT_CONTROL_DIR", package_dir)

    assert set(_agent_control_python_files()) == {root_module, nested_module}


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


def test_agent_control_modules_do_not_import_app_or_voice_runtime_adapters() -> None:
    for path in _agent_control_python_files():
        imported = _import_roots(path)
        forbidden = imported & AGENT_CONTROL_FORBIDDEN_ROOTS

        assert not forbidden, f"{path.name} imports forbidden module(s): {sorted(forbidden)}"


def test_agent_control_voice_runtime_imports_stay_on_explicit_seams() -> None:
    for path in _agent_control_python_files():
        imported = _import_modules(path)
        voice_runtime_imports = {
            module
            for module in imported
            if module == "voice_runtime" or module.startswith("voice_runtime.")
        }
        forbidden = voice_runtime_imports - AGENT_CONTROL_ALLOWED_VOICE_RUNTIME_MODULES

        assert not forbidden, f"{path.name} imports forbidden Voice Runtime seam(s): {sorted(forbidden)}"


def test_legacy_robot_modules_are_not_left_in_old_locations():
    for path in DELETED_LEGACY_ROBOT_MODULES:
        assert not path.exists(), f"Legacy robot module still exists: {path}"


def test_legacy_codex_oauth_modules_are_removed():
    for path in DELETED_LEGACY_AGENT_MODULES:
        assert not path.exists(), f"Legacy Codex OAuth module still exists: {path}"


def test_legacy_voice_provider_module_is_not_left_at_app_root():
    for path in DELETED_LEGACY_VOICE_PROVIDER_MODULES:
        assert not path.exists(), f"Legacy Voice Providers module still exists: {path}"


def test_legacy_voice_command_wrappers_are_not_left_in_wake_adapters():
    for path in DELETED_LEGACY_WAKE_WRAPPERS:
        assert not path.exists(), f"Legacy Voice Command wrapper still exists: {path}"


def test_agent_control_package_contains_agent_control_modules() -> None:
    missing = [
        name
        for name in sorted(REQUIRED_AGENT_CONTROL_MODULES)
        if not (AGENT_CONTROL_DIR / name).exists()
    ]

    assert not missing, f"agent_control is missing module file(s): {missing}"


def test_legacy_agent_control_root_modules_are_deleted() -> None:
    remaining = [
        str(path.relative_to(SERVER_DIR))
        for path in DELETED_LEGACY_AGENT_CONTROL_ROOTS
        if path.exists()
    ]

    assert not remaining, f"legacy root Agent Control module(s) still exist: {remaining}"
