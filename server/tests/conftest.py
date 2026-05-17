from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--vizor-integration",
        action="store_true",
        default=False,
        help="Run tests against a live Vizor/MoveIt stack",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: manual tests that require running local integration services",
    )


def pytest_generate_tests(metafunc) -> None:
    if "vizor_integration" in metafunc.fixturenames:
        metafunc.parametrize(
            "vizor_integration",
            [metafunc.config.getoption("--vizor-integration")],
        )
