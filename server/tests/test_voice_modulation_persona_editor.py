from pathlib import Path

import pytest


def test_behavior_examples_are_included_after_canonical_examples() -> None:
    from agent_control import prompts

    canonical_index = next(
        (
            prompts.SYSTEM_PROMPT.index(heading)
            for heading in (
                "# Canonical motion examples",
                "# Canonical manipulation examples",
            )
            if heading in prompts.SYSTEM_PROMPT
        ),
        None,
    )
    assert canonical_index is not None
    behavior_index = prompts.SYSTEM_PROMPT.index("# Behavior examples")
    speech_tags_index = prompts.SYSTEM_PROMPT.index("# Speech tag examples")

    assert canonical_index < behavior_index < speech_tags_index


def test_persona_parts_expose_only_allowlisted_files(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import load_persona_parts

    prompt_dir = _prompt_parts_dir(tmp_path)
    (prompt_dir / "robot_contract.md").write_text("# Robot contract\n", encoding="utf-8")

    parts = load_persona_parts(prompt_dir)
    by_id = {part.id: part for part in parts}

    assert by_id["mave_embodiment"].editable is True
    assert by_id["speech_delivery_style"].editable is True
    assert by_id["behavior_examples"].editable is True
    assert by_id["canonical_motion_examples"].editable is False
    assert by_id["canonical_motion_examples"].filename == "examples.md"
    assert "robot_contract" not in by_id


def test_save_persona_part_writes_only_editable_allowlisted_part(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import save_persona_part

    prompt_dir = _prompt_parts_dir(tmp_path)

    part = save_persona_part(
        prompt_dir,
        "behavior_examples",
        "# Behavior examples\n- Keep replies brief.\n",
    )

    assert part.id == "behavior_examples"
    assert part.content.startswith("# Behavior examples")
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8").startswith(
        "# Behavior examples"
    )


def test_save_persona_part_rejects_readonly_unknown_and_empty_content(
    tmp_path: Path,
) -> None:
    from voice_modulation.persona_editor import PersonaValidationError, save_persona_part

    prompt_dir = _prompt_parts_dir(tmp_path)

    with pytest.raises(PersonaValidationError, match="read-only"):
        save_persona_part(prompt_dir, "canonical_motion_examples", "# Changed\n")
    with pytest.raises(PersonaValidationError, match="Unknown persona prompt part"):
        save_persona_part(prompt_dir, "robot_contract", "# Changed\n")
    with pytest.raises(PersonaValidationError, match="must not be empty"):
        save_persona_part(prompt_dir, "behavior_examples", "  \n")


def test_persona_templates_report_available_known_templates(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import list_persona_templates

    _template_dir(tmp_path)
    _template_dir(tmp_path, "robot_embodied_agent")

    templates = list_persona_templates(tmp_path)
    by_id = {template["id"]: template for template in templates}

    assert by_id["independent_agent"] == {
        "id": "independent_agent",
        "label": "Independent agent",
        "available": True,
    }
    assert by_id["robot_embodied_agent"] == {
        "id": "robot_embodied_agent",
        "label": "Robot embodied agent",
        "available": True,
    }


def test_load_persona_template_copies_editable_parts(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import load_persona_template

    prompt_dir = _prompt_parts_dir(tmp_path)
    template_dir = _template_dir(tmp_path)
    (template_dir / "behavior_examples.md").write_text(
        "# Behavior examples\n- Template text.\n",
        encoding="utf-8",
    )

    parts = load_persona_template(tmp_path, "independent_agent")
    by_id = {part.id: part for part in parts}

    assert by_id["behavior_examples"].content == "# Behavior examples\n- Template text.\n"
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8") == (
        "# Behavior examples\n- Template text.\n"
    )


def test_load_robot_embodied_template_copies_editable_parts(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import load_persona_template

    prompt_dir = _prompt_parts_dir(tmp_path)
    template_dir = _template_dir(tmp_path, "robot_embodied_agent")
    (template_dir / "mave_embodiment.md").write_text(
        "# MAVE embodiment\nRobot body template.\n",
        encoding="utf-8",
    )
    (template_dir / "speech_delivery_style.md").write_text(
        "# Speech delivery style\nRobot body voice.\n",
        encoding="utf-8",
    )

    parts = load_persona_template(tmp_path, "robot_embodied_agent")
    by_id = {part.id: part for part in parts}

    assert by_id["mave_embodiment"].content == (
        "# MAVE embodiment\nRobot body template.\n"
    )
    assert by_id["speech_delivery_style"].content == (
        "# Speech delivery style\nRobot body voice.\n"
    )
    assert (prompt_dir / "mave_embodiment.md").read_text(encoding="utf-8") == (
        "# MAVE embodiment\nRobot body template.\n"
    )


def test_load_persona_template_rejects_unavailable_template(tmp_path: Path) -> None:
    from voice_modulation.persona_editor import PersonaValidationError, load_persona_template

    with pytest.raises(PersonaValidationError, match="Unavailable persona template"):
        load_persona_template(tmp_path, "robot_embodied_agent")


def _prompt_parts_dir(root: Path) -> Path:
    prompt_dir = root / "agent_control" / "prompt_parts"
    prompt_dir.mkdir(parents=True)
    for filename in (
        "mave_embodiment.md",
        "reasoning_agent_persona.md",
        "response_style.md",
        "speech_delivery_style.md",
        "speech_tag_examples.md",
        "behavior_examples.md",
        "examples.md",
    ):
        (prompt_dir / filename).write_text(f"# {filename}\n", encoding="utf-8")
    return prompt_dir


def _template_dir(root: Path, template_id: str = "independent_agent") -> Path:
    template_dir = root / "agent_control" / "persona_templates" / template_id
    template_dir.mkdir(parents=True)
    for filename in (
        "mave_embodiment.md",
        "reasoning_agent_persona.md",
        "response_style.md",
        "speech_delivery_style.md",
        "speech_tag_examples.md",
        "behavior_examples.md",
    ):
        (template_dir / filename).write_text(f"# template {filename}\n", encoding="utf-8")
    return template_dir
