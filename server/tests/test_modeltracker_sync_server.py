import http.client
import json
import math
import threading
from http.server import ThreadingHTTPServer

from robot_control.shared_geometry.modeltracker_sync_server import make_handler


def test_modeltracker_sync_server_accepts_post_and_logs(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    log_path = tmp_path / "events.jsonl"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(model_path=model_path, log_path=log_path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {
                "names": ["dynamic_snappy-V44B80_box0"],
                "orient": [_rotation_z(math.pi / 2.0)],
                "transl": [_translation(0.2, 0.3, 0.4)],
                "mesh_centers": [[0.2, 0.3, 0.4]],
            }
        )
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request(
            "POST",
            "/modeltracker-event",
            body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["object_name"] == "dynamic_0"

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["bodies"][0]["pose"]["xyz"] == [-0.2, -0.3, 0.4]

    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert logs[0]["result"]["ok"] is True
    assert logs[0]["event"]["names"] == ["dynamic_snappy-V44B80_box0"]


def _model() -> dict:
    return {
        "schema": "pipecat.shared_geometry.common_model.v0.1",
        "version": "0.1.0",
        "name": "hologram_frame",
        "units": "meters",
        "bodies": [
            {
                "id": "dynamic_0",
                "solid": {
                    "type": "box",
                    "origin": "center",
                    "dimensions": {"x": 0.1, "y": 0.04, "z": 0.04},
                },
                "axis": {
                    "start_xyz": [0.0, -0.7, 0.0],
                    "end_xyz": [0.0, -0.7, 0.1],
                },
                "pose": {
                    "frame": "rhino_world",
                    "xyz": [0.0, -0.7, -0.045],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "features": {},
            }
        ],
    }


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation(x: float, y: float, z: float) -> list[list[float]]:
    matrix = _identity()
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    return matrix


def _rotation_z(angle: float) -> list[list[float]]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return [
        [cos_angle, -sin_angle, 0.0, 0.0],
        [sin_angle, cos_angle, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
