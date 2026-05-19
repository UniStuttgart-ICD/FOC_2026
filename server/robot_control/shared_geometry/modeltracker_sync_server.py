from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from robot_control.shared_geometry.modeltracker_sync import ModelTrackerSyncSession
from robot_control.shared_geometry.world_context import DEFAULT_HOLOGRAM_MODEL_PATH

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
DEFAULT_LOG_PATH = Path(__file__).parents[2] / "logs" / "modeltracker_sync_events.jsonl"


def make_handler(
    *,
    model_path: str | Path = DEFAULT_HOLOGRAM_MODEL_PATH,
    log_path: str | Path | None = DEFAULT_LOG_PATH,
) -> type[BaseHTTPRequestHandler]:
    target_model_path = Path(model_path)
    target_log_path = Path(log_path) if log_path is not None else None
    session = ModelTrackerSyncSession(model_path=target_model_path)

    class ModelTrackerSyncHandler(BaseHTTPRequestHandler):
        server_version = "ModelTrackerSync/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json({"ok": False, "error": "not found"}, status=404)
                return
            self._send_json({"ok": True, "status": "ready"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/modeltracker-event":
                self._send_json({"ok": False, "error": "not found"}, status=404)
                return

            event = self._read_json_body()
            if not isinstance(event, dict):
                self._send_json({"ok": False, "error": "request body must be a JSON object"}, status=400)
                return

            result = session.handle_event(event)
            if target_log_path is not None:
                _append_event_log(target_log_path, event, result)
            self._send_json(result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json_body(self) -> Any:
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "0")
            except ValueError:
                return None
            try:
                raw = self.rfile.read(length)
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ModelTrackerSyncHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Grasshopper ModelTracker hologram sync.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_HOLOGRAM_MODEL_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args(argv)

    handler = make_handler(model_path=args.model_path, log_path=args.log_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"ModelTracker sync server listening on http://{args.host}:{args.port}")
    print(f"Writing hologram model: {args.model_path}")
    print(f"Logging events: {args.log_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _append_event_log(path: Path, event: dict[str, Any], result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "result": result,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
