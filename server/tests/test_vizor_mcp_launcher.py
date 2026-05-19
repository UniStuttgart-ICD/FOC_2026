import vizor_mcp.__main__ as module_entrypoint
from vizor_mcp.server import main


def test_vizor_mcp_module_entrypoint_uses_server_main() -> None:
    assert module_entrypoint.main is main
