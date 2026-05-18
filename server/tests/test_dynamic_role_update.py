import copy
import json

from robot_control.shared_geometry.role_update import update_dynamic_role


def test_update_dynamic_role_sets_unassigned(tmp_path) -> None:
    model_path = _write_model(tmp_path)

    result = update_dynamic_role(
        "dynamic_00",
        {"type": "unassigned"},
        "object is no longer part of the frame",
        model_path=model_path,
    )

    model = _read_model(model_path)
    assert result == {
        "ok": True,
        "object_name": "dynamic_0",
        "role": {"type": "unassigned"},
        "physical_model_updated": True,
    }
    assert _body(model, "dynamic_0")["state"]["role"] == {"type": "unassigned"}


def test_update_dynamic_role_sets_supporting_column(tmp_path) -> None:
    model_path = _write_model(tmp_path)

    result = update_dynamic_role(
        "dynamic_00",
        {"type": "supporting_column", "supports": ["dynamic_02"]},
        "column supports the horizontal span",
        model_path=model_path,
    )

    model = _read_model(model_path)
    role = {"type": "supporting_column", "supports": ["dynamic_2"]}
    assert result == {
        "ok": True,
        "object_name": "dynamic_0",
        "role": role,
        "physical_model_updated": True,
    }
    assert _body(model, "dynamic_0")["state"]["role"] == role


def test_update_dynamic_role_sets_beam_supported_by(tmp_path) -> None:
    model_path = _write_model(tmp_path)

    result = update_dynamic_role(
        "dynamic_2",
        {"type": "beam_supported_by", "supported_by": ["dynamic_00", "dynamic_01"]},
        "beam rests on two columns",
        model_path=model_path,
    )

    model = _read_model(model_path)
    role = {"type": "beam_supported_by", "supported_by": ["dynamic_0", "dynamic_1"]}
    assert result == {
        "ok": True,
        "object_name": "dynamic_2",
        "role": role,
        "physical_model_updated": True,
    }
    assert _body(model, "dynamic_2")["state"]["role"] == role


def test_update_dynamic_role_rejects_unknown_references(tmp_path) -> None:
    model_path = _write_model(tmp_path)
    before = _read_model(model_path)

    result = update_dynamic_role(
        "dynamic_0",
        {"type": "supporting_column", "supports": ["dynamic_99"]},
        "bad reference",
        model_path=model_path,
    )

    assert result["ok"] is False
    assert "unknown" in str(result["error"])
    assert result["retryable"] is True
    assert _read_model(model_path) == before


def test_update_dynamic_role_rejects_free_prose_and_view_dependent_roles(tmp_path) -> None:
    model_path = _write_model(tmp_path)
    before = _read_model(model_path)

    for role in (
        "left support",
        {"type": "left_support"},
        {"type": "right_support"},
        {"type": "inventory"},
        {"type": "built"},
        {"type": "supporting_column", "supports": []},
    ):
        result = update_dynamic_role(
            "dynamic_0",
            role,
            "invalid role",
            model_path=model_path,
        )

        assert result["ok"] is False
        assert result["retryable"] is True

    assert _read_model(model_path) == before


def test_update_dynamic_role_appends_history_and_preserves_other_fields(tmp_path) -> None:
    model_path = _write_model(tmp_path)
    before = _read_model(model_path)
    role = {"type": "beam_supported_by", "supported_by": ["dynamic_0", "dynamic_1"]}

    update_dynamic_role(
        "dynamic_2",
        role,
        "span is now supported by both columns",
        model_path=model_path,
    )

    model = _read_model(model_path)
    expected = copy.deepcopy(before)
    expected["bodies"][2]["state"]["role"] = role
    expected["operation_history"].append(
        {
            "op": "dynamic_role_update",
            "status": "applied",
            "object_name": "dynamic_2",
            "reason": "span is now supported by both columns",
            "role": role,
        }
    )
    assert model == expected


def _write_model(tmp_path):
    model_path = tmp_path / "physical_model.json"
    model_path.write_text(json.dumps(_model()), encoding="utf-8")
    return model_path


def _read_model(model_path):
    return json.loads(model_path.read_text(encoding="utf-8"))


def _model() -> dict:
    return {
        "schema": "pipecat.shared_geometry.common_model.v0.1",
        "version": "0.1.0",
        "name": "test_physical_model",
        "units": "meters",
        "bodies": [
            _model_body("dynamic_0", "left column"),
            _model_body("dynamic_1", "right column"),
            _model_body("dynamic_2", "span"),
        ],
        "operation_history": [
            {
                "op": "initialize_test_model",
                "status": "applied",
            }
        ],
        "untouched": {"kept": True},
    }


def _model_body(body_id: str, label: str) -> dict:
    return {
        "id": body_id,
        "label": label,
        "state": {
            "status": "built",
            "alignment": "vertical",
            "role": "old_role",
        },
    }


def _body(model: dict, body_id: str) -> dict:
    for body in model["bodies"]:
        if body["id"] == body_id:
            return body
    raise AssertionError(f"missing body {body_id}")
