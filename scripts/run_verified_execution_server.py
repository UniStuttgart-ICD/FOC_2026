from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from verified_execution_server.server import create_default_app


def main() -> None:
    uvicorn.run(
        create_default_app(),
        host=os.getenv("VERIFIED_EXECUTION_HOST", "127.0.0.1"),
        port=int(os.getenv("VERIFIED_EXECUTION_PORT", "8770")),
    )


if __name__ == "__main__":
    main()
