from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaPart:
    id: str
    filename: str
    title: str
    editable: bool
    content: str
    restart_required: bool = True


class PersonaValidationError(ValueError):
    """Raised when a persona edit targets a disallowed prompt part."""


@dataclass(frozen=True)
class _PartSpec:
    id: str
    filename: str
    title: str
    editable: bool


_EDITABLE_PARTS: tuple[_PartSpec, ...] = (
    _PartSpec("mave_embodiment", "mave_embodiment.md", "MAVE embodiment", True),
    _PartSpec(
        "reasoning_agent_persona",
        "reasoning_agent_persona.md",
        "Reasoning agent persona",
        True,
    ),
    _PartSpec(
        "speech_delivery_style",
        "speech_delivery_style.md",
        "Speech delivery style",
        True,
    ),
    _PartSpec(
        "speech_tag_examples",
        "speech_tag_examples.md",
        "Speech tag examples",
        True,
    ),
    _PartSpec("behavior_examples", "behavior_examples.md", "Behavior examples", True),
)
_READ_ONLY_PARTS: tuple[_PartSpec, ...] = (
    _PartSpec(
        "canonical_motion_examples",
        "examples.md",
        "Canonical motion examples",
        False,
    ),
)
_PARTS = _EDITABLE_PARTS + _READ_ONLY_PARTS
_PARTS_BY_ID = {part.id: part for part in _PARTS}


def load_persona_parts(prompt_parts_dir: str | Path) -> list[PersonaPart]:
    root = Path(prompt_parts_dir)
    return [_load_part(root, spec) for spec in _PARTS]


def save_persona_part(
    prompt_parts_dir: str | Path,
    part_id: str,
    content: str,
) -> PersonaPart:
    spec = _part_spec(part_id)
    if not spec.editable:
        raise PersonaValidationError(f"Persona prompt part is read-only: {part_id}")
    _validate_content(content)

    root = Path(prompt_parts_dir)
    path = root / spec.filename
    path.write_text(content, encoding="utf-8")
    return _part_from_content(spec, content)


def save_persona_template_part(
    server_dir: str | Path,
    template_id: str,
    part_id: str,
    content: str,
) -> PersonaPart:
    spec = _part_spec(part_id)
    if not spec.editable:
        raise PersonaValidationError(f"Persona prompt part is read-only: {part_id}")
    _validate_content(content)

    root = Path(server_dir)
    template_dir = _template_dir(root, template_id)
    if template_dir is None or not template_dir.is_dir():
        raise PersonaValidationError(f"Unavailable persona template: {template_id}")

    path = template_dir / spec.filename
    if not path.is_file():
        raise PersonaValidationError(f"Persona template is missing prompt part: {spec.filename}")
    path.write_text(content, encoding="utf-8")
    return _part_from_content(spec, content)


def list_persona_templates(server_dir: str | Path) -> list[dict[str, object]]:
    root = Path(server_dir)
    templates_dir = _templates_dir(root)
    if not templates_dir.is_dir():
        return []

    templates = []
    for template_dir in sorted(
        templates_dir.iterdir(),
        key=lambda path: _template_label(path.name),
    ):
        if not template_dir.is_dir():
            continue
        missing_parts = [
            spec.filename
            for spec in _EDITABLE_PARTS
            if not (template_dir / spec.filename).is_file()
        ]
        templates.append(
            {
                "id": template_dir.name,
                "label": _template_label(template_dir.name),
                "available": not missing_parts,
                "missing_parts": missing_parts,
            }
        )
    return templates


def load_persona_template(server_dir: str | Path, template_id: str) -> list[PersonaPart]:
    root = Path(server_dir)
    template_dir = _template_dir(root, template_id)
    if template_dir is None or not template_dir.is_dir():
        raise PersonaValidationError(f"Unavailable persona template: {template_id}")

    prompt_parts_dir = root / "agent_control" / "prompt_parts"
    prompt_parts_dir.mkdir(parents=True, exist_ok=True)
    changed_parts: list[PersonaPart] = []
    for spec in _EDITABLE_PARTS:
        source = template_dir / spec.filename
        if not source.is_file():
            raise PersonaValidationError(
                f"Persona template is missing prompt part: {spec.filename}"
            )
        content = source.read_text(encoding="utf-8")
        _validate_content(content)
        (prompt_parts_dir / spec.filename).write_text(content, encoding="utf-8")
        changed_parts.append(_part_from_content(spec, content))
    return changed_parts


def _templates_dir(server_dir: Path) -> Path:
    return server_dir / "agent_control" / "persona_templates"


def _template_dir(server_dir: Path, template_id: str) -> Path | None:
    if not template_id.strip():
        return None
    templates_dir = _templates_dir(server_dir).resolve()
    candidate = (templates_dir / template_id).resolve()
    try:
        candidate.relative_to(templates_dir)
    except ValueError:
        return None
    return candidate


def _template_label(template_id: str) -> str:
    label = template_id.replace("_", " ").replace("-", " ").strip()
    return " ".join(_title_word(word) for word in label.split()) or template_id


def _title_word(word: str) -> str:
    if word.isupper():
        return word
    return word[:1].upper() + word[1:]


def _load_part(prompt_parts_dir: Path, spec: _PartSpec) -> PersonaPart:
    content = (prompt_parts_dir / spec.filename).read_text(encoding="utf-8")
    return _part_from_content(spec, content)


def _part_from_content(spec: _PartSpec, content: str) -> PersonaPart:
    return PersonaPart(
        id=spec.id,
        filename=spec.filename,
        title=spec.title,
        editable=spec.editable,
        content=content,
    )


def _part_spec(part_id: str) -> _PartSpec:
    try:
        return _PARTS_BY_ID[part_id]
    except KeyError as exc:
        raise PersonaValidationError(
            f"Unknown persona prompt part: {part_id}"
        ) from exc


def _validate_content(content: str) -> None:
    if not content.strip():
        raise PersonaValidationError("Persona prompt part content must not be empty")
