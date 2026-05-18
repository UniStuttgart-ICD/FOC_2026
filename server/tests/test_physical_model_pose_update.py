import json

from robot_control.shared_geometry.pose_update import update_physical_model_pose


def test_updates_pose_axis_feature_endpoints_and_feature_centers(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")
    evidence = _pose_evidence(
        "dynamic_0",
        position={"x": 0.0, "y": -0.8, "z": 0.1},
        orientation={"x": 0.5, "y": 0.5, "z": -0.5, "w": 0.5},
    )

    result = update_physical_model_pose(
        "dynamic_0",
        "verified_pick_place_release",
        evidence,
        model_path=model_path,
    )

    assert result == {
        "ok": True,
        "object_name": "dynamic_0",
        "reason": "verified_pick_place_release",
        "source": "moveit_get_object_context",
    }
    data = json.loads(model_path.read_text(encoding="utf-8"))
    body = data["bodies"][0]
    assert body["pose"]["xyz"] == [0.0, -0.8, 0.1]
    assert body["pose"]["quat_xyzw"] == [0.5, 0.5, -0.5, 0.5]
    assert body["axis"]["start_xyz"] == [0.0, -0.8, 0.05]
    assert body["axis"]["end_xyz"] == [0.0, -0.8, 0.15000000000000002]
    assert body["features"]["end_start"]["world_xyz"] == [0.0, -0.8, 0.05]
    assert body["features"]["end_end"]["world_xyz"] == [0.0, -0.8, 0.15000000000000002]
    assert body["features"]["face_bottom"]["center_xyz"] == [0.0, -0.78, 0.1]
    assert body["features"]["face_top"]["center_xyz"] == [0.0, -0.8200000000000001, 0.1]
    assert body["features"]["face_top"]["normal_xyz"] == [0, 0, 1]


def test_normalizes_padded_dynamic_name(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")

    result = update_physical_model_pose(
        "dynamic_00",
        "operator_sync",
        _pose_evidence("dynamic_00"),
        model_path=model_path,
    )

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["object_name"] == "dynamic_0"
    assert data["bodies"][0]["pose"]["xyz"] == [0.2, -0.6, 0.3]


def test_rejects_missing_quaternion(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")
    evidence = {
        "object_name": "dynamic_0",
        "source": "moveit_get_object_context",
        "pose": {"position": {"x": 0.0, "y": -0.8, "z": 0.1}},
    }

    result = update_physical_model_pose(
        "dynamic_0",
        "verified_place_release",
        evidence,
        model_path=model_path,
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert "quaternion" in str(result["error"])
    assert model_path.read_text(encoding="utf-8") == original


def test_rejects_bounds_only_evidence(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")
    evidence = {
        "object_name": "dynamic_0",
        "source": "moveit_get_object_context",
        "bounds": {"min": {"x": 0.0}, "max": {"x": 1.0}},
    }

    result = update_physical_model_pose(
        "dynamic_0",
        "verified_place_release",
        evidence,
        model_path=model_path,
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert "pose evidence" in str(result["correction"])
    assert model_path.read_text(encoding="utf-8") == original


def test_rejects_disallowed_reason(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    original = json.dumps(_model())
    model_path.write_text(original, encoding="utf-8")

    result = update_physical_model_pose(
        "dynamic_0",
        "unverified_guess",
        _pose_evidence("dynamic_0"),
        model_path=model_path,
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert "unsupported reason" in str(result["error"])
    assert model_path.read_text(encoding="utf-8") == original


def test_leaves_unrelated_bodies_untouched(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    model = _model()
    model["bodies"].append(_body("dynamic_1"))
    original_second_body = json.loads(json.dumps(model["bodies"][1]))
    model_path.write_text(json.dumps(model), encoding="utf-8")

    update_physical_model_pose(
        "dynamic_0",
        "operator_sync",
        _pose_evidence("dynamic_0"),
        model_path=model_path,
    )

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["bodies"][1] == original_second_body


def test_appends_compact_operation_history(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")
    evidence = _pose_evidence("dynamic_0")

    update_physical_model_pose(
        "dynamic_0",
        "operator_sync",
        evidence,
        model_path=model_path,
    )

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["operation_history"] == [
        {
            "op": "physical_model_pose_update",
            "status": "applied",
            "object_name": "dynamic_0",
            "reason": "operator_sync",
            "source": "moveit_get_object_context",
            "pose": evidence["pose"],
        }
    ]


def test_failed_update_does_not_corrupt_json(tmp_path) -> None:
    model_path = tmp_path / "physical_model.json"
    original_data = _model()
    model_path.write_text(json.dumps(original_data), encoding="utf-8")

    result = update_physical_model_pose(
        "dynamic_2",
        "operator_sync",
        _pose_evidence("dynamic_2"),
        model_path=model_path,
    )

    assert result["ok"] is False
    assert json.loads(model_path.read_text(encoding="utf-8")) == original_data


def _model() -> dict:
    return {
        "schema": "pipecat.shared_geometry.common_model.v0.1",
        "version": "0.1.0",
        "name": "physical_frame",
        "units": "meters",
        "bodies": [_body("dynamic_0")],
        "operation_history": [],
    }


def _body(body_id: str) -> dict:
    return {
        "id": body_id,
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
            "xyz": [0.0, -0.7, 0.05],
            "quat_xyzw": [0.5, 0.5, -0.5, 0.5],
        },
        "features": {
            "end_start": {
                "alias": "end_neg_x",
                "world_xyz": [0.0, -0.7, 0.0],
            },
            "end_end": {
                "alias": "end_pos_x",
                "world_xyz": [0.0, -0.7, 0.1],
            },
            "face_bottom": {
                "center_xyz": [0.0, -0.7, 0.0],
                "normal_xyz": [0, 0, -1],
            },
            "face_top": {
                "center_xyz": [0.0, -0.7, 0.1],
                "normal_xyz": [0, 0, 1],
            },
        },
    }


def _pose_evidence(
    object_name: str,
    *,
    position: dict[str, float] | None = None,
    orientation: dict[str, float] | None = None,
) -> dict:
    return {
        "object_name": object_name,
        "source": "moveit_get_object_context",
        "pose": {
            "position": position or {"x": 0.2, "y": -0.6, "z": 0.3},
            "orientation": orientation or {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
