from pathlib import Path

import pytest

from model_eval.config import EvalRunConfig, load_model_matrix


def test_load_model_matrix_reads_candidates(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
        """
[[candidates]]
label = "gpt-5.4-mini-medium"
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "sonnet-4.6-low"
provider = "anthropic_api"
model = "claude-sonnet-4-6"
reasoning_effort = "low"
api_key_env = "ANTHROPIC_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    candidates = load_model_matrix(matrix_path)

    assert [candidate.label for candidate in candidates] == [
        "gpt-5.4-mini-medium",
        "sonnet-4.6-low",
    ]
    assert candidates[0].provider == "openai_api"
    assert candidates[0].model == "gpt-5.4-mini"
    assert candidates[0].reasoning_effort == "medium"


def test_load_model_matrix_rejects_empty_candidates(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text("candidates = []", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one candidate"):
        load_model_matrix(matrix_path)


def test_load_model_matrix_rejects_duplicate_labels(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
        """
[[candidates]]
label = "same"
provider = "openai_api"
model = "gpt-5.4-mini"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "same"
provider = "anthropic_api"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate candidate labels"):
        load_model_matrix(matrix_path)


def test_load_model_matrix_rejects_unknown_provider(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
        """
[[candidates]]
label = "bad"
provider = "unknown_api"
model = "some-model"
api_key_env = "BAD_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported candidate provider"):
        load_model_matrix(matrix_path)


def test_load_model_matrix_rejects_unknown_reasoning_effort(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
        """
[[candidates]]
label = "bad"
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "extreme"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported reasoning_effort"):
        load_model_matrix(matrix_path)


def test_eval_run_config_defaults_to_simulated_adapter(tmp_path: Path) -> None:
    config = EvalRunConfig(
        matrix_path=tmp_path / "matrix.toml",
        pack_name="core_robot_commands",
    )

    assert config.adapter == "simulated"
    assert config.mcp_url == "http://127.0.0.1:8765/mcp"
    assert config.samples == 1
    assert config.evidence_root == Path("evidence/model_eval")


def test_eval_run_config_rejects_non_positive_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="samples must be at least 1"):
        EvalRunConfig(
            matrix_path=tmp_path / "matrix.toml",
            pack_name="core_robot_commands",
            samples=0,
        )
