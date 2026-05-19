import json
import math

from robot_control.shared_geometry.modeltracker_sync import (
    ModelTrackerSyncSession,
    sync_modeltracker_event,
)


def test_updates_hologram_pose_from_modeltracker_event(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")

    result = sync_modeltracker_event(
        {
            "names": ["dynamic_snappy-V110B110_box0", "dynamic_snappy-V110B110_box1"],
            "orient": [_identity(), _rotation_z(math.pi / 2.0)],
            "transl": [_identity(), _translation(0.2, 0.3, 0.4)],
            "mesh_centers": [[0.7, 0.0, -0.045], [0.2, 0.3, 0.4]],
        },
        model_path=model_path,
    )

    assert result["ok"] is True
    assert result["object_name"] == "dynamic_1"
    assert result["event_index"] == 1

    data = json.loads(model_path.read_text(encoding="utf-8"))
    first_body, changed_body = data["bodies"]
    assert first_body["pose"]["xyz"] == [0.0, -0.7, -0.045]
    assert changed_body["pose"]["xyz"] == [-0.2, -0.3, 0.4]
    assert changed_body["pose"]["quat_xyzw"] == [0.0, 0.0, -0.707106781187, 0.707106781187]
    assert changed_body["axis"]["start_xyz"] == [-0.2, -0.25, 0.4]
    assert changed_body["axis"]["end_xyz"] == [-0.2, -0.35, 0.4]


def test_idle_modeltracker_event_does_not_write(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")

    result = sync_modeltracker_event(
        {
            "names": ["dynamic_snappy-V110B110_box0", "dynamic_snappy-V110B110_box1"],
            "orient": [],
            "transl": [],
            "mesh_centers": [[0.7, 0.0, -0.045], [0.2, 0.3, 0.4]],
        },
        model_path=model_path,
    )

    assert result == {
        "ok": True,
        "updated": False,
        "status": "Waiting for orient/transl/names/mesh_centers inputs.",
    }
    assert model_path.read_text(encoding="utf-8") == original


def test_rejects_ambiguous_single_transform_with_static_names(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")

    result = sync_modeltracker_event(
        {
            "names": ["dynamic_snappy-V110B110_box0", "dynamic_snappy-V110B110_box1"],
            "orient": [_rotation_z(math.pi / 2.0)],
            "transl": [_translation(0.2, 0.3, 0.4)],
            "mesh_centers": [[0.7, 0.0, -0.045], [0.2, 0.3, 0.4]],
        },
        model_path=model_path,
    )

    assert result["ok"] is False
    assert "Cannot map ModelTracker event" in result["error"]
    assert model_path.read_text(encoding="utf-8") == original


def test_rejects_unknown_changed_name(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")

    result = sync_modeltracker_event(
        {
            "names": ["dynamic_snappy-V110B110_box9"],
            "orient": [_rotation_z(math.pi / 2.0)],
            "transl": [_translation(0.2, 0.3, 0.4)],
            "mesh_centers": [[0.2, 0.3, 0.4]],
        },
        model_path=model_path,
    )

    assert result["ok"] is False
    assert "dynamic_9: not found" in result["error"]
    assert model_path.read_text(encoding="utf-8") == original


def test_accepts_snappy_v44b80_modeltracker_names(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")

    result = sync_modeltracker_event(
        {
            "names": ["dynamic_snappy-V44B80_box0"],
            "orient": [_rotation_z(math.pi / 2.0)],
            "transl": [_translation(0.2, 0.3, 0.4)],
            "mesh_centers": [[0.2, 0.3, 0.4]],
        },
        model_path=model_path,
    )

    assert result["ok"] is True
    assert result["object_name"] == "dynamic_0"

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["bodies"][0]["pose"]["xyz"] == [-0.2, -0.3, 0.4]


def test_session_uses_previous_snapshot_to_select_one_changed_index(tmp_path) -> None:
    model_path = tmp_path / "hologram_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")
    session = ModelTrackerSyncSession(model_path=model_path)
    baseline = {
        "names": ["dynamic_snappy-V110B110_box0", "dynamic_snappy-V110B110_box1"],
        "orient": [_identity(), _rotation_z(math.pi / 2.0)],
        "transl": [_translation(0.1, 0.0, 0.0), _translation(0.2, 0.3, 0.4)],
        "mesh_centers": [[0.7, 0.0, -0.045], [0.2, 0.3, 0.4]],
    }

    first = session.handle_event(baseline)

    assert first == {
        "ok": True,
        "updated": False,
        "status": "Captured ModelTracker baseline with 2 changed transforms.",
    }

    moved = json.loads(json.dumps(baseline))
    moved["mesh_centers"][1] = [0.25, 0.3, 0.4]

    second = session.handle_event(moved)

    assert second["ok"] is True
    assert second["updated"] is True
    assert second["object_name"] == "dynamic_1"
    assert second["event_index"] == 1
    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["bodies"][1]["pose"]["xyz"] == [-0.25, -0.3, 0.4]


def _model() -> dict:
    return {
        "schema": "pipecat.shared_geometry.common_model.v0.1",
        "version": "0.1.0",
        "name": "hologram_frame",
        "units": "meters",
        "bodies": [_body("dynamic_0"), _body("dynamic_1")],
        "operation_history": [],
    }


def _body(body_id: str) -> dict:
    return {
        "id": body_id,
        "label": body_id,
        "family": "A",
        "solid": {
            "type": "box",
            "origin": "center",
            "dimensions": {"x": 0.1, "y": 0.04, "z": 0.04},
        },
        "axis": {
            "start_label": "bottom",
            "end_label": "top",
            "start_xyz": [0.0, -0.7, 0.0],
            "end_xyz": [0.0, -0.7, 0.1],
        },
        "pose": {
            "frame": "rhino_world",
            "xyz": [0.0, -0.7, -0.045],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "features": {
            "end_start": {"alias": "end_neg_x", "world_xyz": [0.0, -0.7, 0.0]},
            "end_end": {"alias": "end_pos_x", "world_xyz": [0.0, -0.7, 0.1]},
            "face_bottom": {"center_xyz": [0.0, -0.7, 0.0], "normal_xyz": [0, 0, -1]},
            "face_top": {"center_xyz": [0.0, -0.7, 0.1], "normal_xyz": [0, 0, 1]},
        },
        "state": {"alignment": "vertical"},
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
