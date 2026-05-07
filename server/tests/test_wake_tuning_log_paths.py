from pathlib import Path

from wake_tuning.log_paths import default_log_dir, log_paths


def test_default_log_dir_lives_under_server_logs_wake_tuning(tmp_path: Path) -> None:
    assert default_log_dir(tmp_path) == tmp_path / "logs" / "wake_tuning"


def test_log_paths_use_safe_label_and_out_err_suffixes(tmp_path: Path) -> None:
    paths = log_paths("manual run", server_dir=tmp_path)

    assert paths.stdout == tmp_path / "logs" / "wake_tuning" / "wake_tuning_manual_run.out.log"
    assert paths.stderr == tmp_path / "logs" / "wake_tuning" / "wake_tuning_manual_run.err.log"


def test_log_paths_default_to_server_label(tmp_path: Path) -> None:
    paths = log_paths(server_dir=tmp_path)

    assert paths.stdout.name == "wake_tuning_server.out.log"
    assert paths.stderr.name == "wake_tuning_server.err.log"


def test_readme_routes_wake_tuning_logs_under_server_logs() -> None:
    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "logs/wake_tuning" in text
    assert "wake_tuning_server.out.log" in text
    assert "wake_tuning_server.err.log" in text
    assert "local override" in text
    assert "does not edit `server/runtime_profiles.toml`" in text
