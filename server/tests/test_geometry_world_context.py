import json

from robot_control.shared_geometry.world_context import (
    GeometryWorldContextStore,
    canonical_dynamic_name,
)


def test_geometry_world_context_renders_physical_role_and_hologram_target_pose(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(
            _model(
                "physical_frame",
                [
                    _body(
                        "dynamic_0",
                        label="B vertical at shared vertex",
                        family="B",
                        group="vertical",
                        status="built",
                        role={"type": "supporting_column", "supports": ["dynamic_2"]},
                        xyz=[9.0, 9.0, 9.0],
                        quat=[0.5, 0.5, -0.5, 0.5],
                    ),
                    _body("dynamic_2", role={"type": "unassigned"}),
                ],
            )
        ),
        encoding="utf-8",
    )
    hologram_path.write_text(
        json.dumps(
            _model(
                "hologram_frame",
                [
                    _body(
                        "dynamic_0",
                        label="B vertical hologram target",
                        family="B",
                        group="vertical",
                        status="preview",
                        xyz=[0.0, -0.8, 0.1],
                        quat=[0.5, 0.5, -0.5, 0.5],
                    ),
                    _body("dynamic_2", role={"type": "hologram_only"}),
                ],
            )
        ),
        encoding="utf-8",
    )
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "Geometry World Context" in text
    assert '"physical_model_name": "physical_frame"' in text
    assert '"hologram_model_name": "hologram_frame"' in text
    assert '"object_name": "dynamic_0"' in text
    assert '"label": "B vertical at shared vertex"' in text
    assert '"family": "B"' in text
    assert '"role": {"type": "supporting_column", "supports": ["dynamic_2"]}' in text
    assert '"position": {"x": 0.0, "y": -0.8, "z": 0.1}' in text
    assert '"orientation": {"x": 0.5, "y": 0.5, "z": -0.5, "w": 0.5}' in text
    assert '"group"' not in text
    assert '"status"' not in text
    assert '"position": {"x": 9.0, "y": 9.0, "z": 9.0}' not in text
    assert "desired object poses, not TCP poses" in text


def test_geometry_world_context_reloads_hologram_file_each_render(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(
            _model(
                "physical_frame",
                [
                    _body(
                        "dynamic_0",
                        role={"type": "unassigned"},
                        xyz=[0.0, 0.0, 0.1],
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    hologram_path.write_text(
        json.dumps(_model("hologram_frame", [_body("dynamic_0", xyz=[0.0, 0.0, 0.1])])),
        encoding="utf-8",
    )
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    assert '"position": {"x": 0.0, "y": 0.0, "z": 0.1}' in store.render_instruction_block()

    hologram_path.write_text(
        json.dumps(_model("hologram_frame", [_body("dynamic_0", xyz=[0.2, 0.3, 0.4])])),
        encoding="utf-8",
    )

    assert '"position": {"x": 0.2, "y": 0.3, "z": 0.4}' in store.render_instruction_block()


def test_geometry_world_context_blocks_missing_hologram_body(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(
            _model(
                "physical_frame",
                [_body("dynamic_0", role={"type": "unassigned"})],
            )
        ),
        encoding="utf-8",
    )
    hologram_path.write_text(json.dumps(_model("hologram_frame", [])), encoding="utf-8")
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "BLOCKED" in text
    assert "dynamic_0 is missing from hologram model" in text
    assert "must not infer a fallback target" in text
    assert '"target_pose"' not in text


def test_geometry_world_context_blocks_invalid_hologram_pose(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(
            _model(
                "physical_frame",
                [_body("dynamic_0", role={"type": "unassigned"})],
            )
        ),
        encoding="utf-8",
    )
    hologram_body = _body("dynamic_0")
    hologram_body["pose"] = {"frame": "rhino_world", "xyz": [0.0, 0.0]}
    hologram_path.write_text(
        json.dumps(_model("hologram_frame", [hologram_body])),
        encoding="utf-8",
    )
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "BLOCKED" in text
    assert "invalid hologram target pose for dynamic_0" in text
    assert "must not infer a fallback target" in text
    assert '"target_pose"' not in text


def test_geometry_world_context_blocks_invalid_physical_role_payload(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(_model("physical_frame", [_body("dynamic_0", role="left_support")])),
        encoding="utf-8",
    )
    hologram_path.write_text(json.dumps(_model("hologram_frame", [_body("dynamic_0")])), encoding="utf-8")
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "BLOCKED" in text
    assert "invalid physical role payload for dynamic_0" in text
    assert '"target_pose"' not in text


def test_geometry_world_context_blocks_unknown_physical_role_reference(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(
            _model(
                "physical_frame",
                [_body("dynamic_0", role={"type": "supporting_column", "supports": ["dynamic_99"]})],
            )
        ),
        encoding="utf-8",
    )
    hologram_path.write_text(json.dumps(_model("hologram_frame", [_body("dynamic_0")])), encoding="utf-8")
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "BLOCKED" in text
    assert "invalid physical role payload for dynamic_0" in text
    assert '"target_pose"' not in text


def test_geometry_world_context_ignores_hologram_role_payload(tmp_path) -> None:
    physical_path = tmp_path / "physical_model.json"
    hologram_path = tmp_path / "hologram_model.json"
    physical_path.write_text(
        json.dumps(_model("physical_frame", [_body("dynamic_0", role={"type": "unassigned"})])),
        encoding="utf-8",
    )
    hologram_path.write_text(
        json.dumps(
            _model(
                "hologram_frame",
                [
                    _body(
                        "dynamic_0",
                        role={"type": "hologram_only", "sentinel": "do_not_render"},
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    store = GeometryWorldContextStore(
        physical_model_path=physical_path,
        hologram_model_path=hologram_path,
    )

    text = store.render_instruction_block()

    assert "BLOCKED" not in text
    assert '"role": {"type": "unassigned"}' in text
    assert "hologram_only" not in text
    assert "do_not_render" not in text


def test_canonical_dynamic_name_normalizes_padded_dynamic_names() -> None:
    assert canonical_dynamic_name("dynamic_0") == "dynamic_0"
    assert canonical_dynamic_name("dynamic_00") == "dynamic_0"
    assert canonical_dynamic_name("beam_1") == "beam_1"


def _model(name: str, bodies: list[dict]) -> dict:
    return {
        "schema": "pipecat.shared_geometry.common_model.v0.1",
        "version": "0.1.0",
        "name": name,
        "units": "meters",
        "bodies": bodies,
    }


def _body(
    body_id: str,
    *,
    label: str = "B vertical",
    family: str = "B",
    group: str = "vertical",
    status: str | None = None,
    role: dict | str | None = None,
    xyz: list[float] | None = None,
    quat: list[float] | None = None,
) -> dict:
    body = {
        "id": body_id,
        "label": label,
        "family": family,
        "group": group,
        "pose": {
            "frame": "rhino_world",
            "xyz": xyz or [0.0, -0.8, 0.1],
            "quat_xyzw": quat or [0.5, 0.5, -0.5, 0.5],
        },
    }
    state = {}
    if status is not None:
        state["status"] = status
    if role is not None:
        state["role"] = role
    if state:
        body["state"] = state
    return body
