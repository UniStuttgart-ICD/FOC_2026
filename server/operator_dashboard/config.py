from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from operator_dashboard.models import AUXILIARY_SERVICE_IDS, DashboardConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_dashboard_config(path: str | Path) -> DashboardConfig:
    config_path = Path(path).expanduser().resolve()
    data: dict[str, Any] = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _apply_service_defaults(data)
    _resolve_service_cwds(data, _config_base_dir(config_path))
    return DashboardConfig.model_validate(data)


def _apply_service_defaults(data: dict[str, Any]) -> None:
    services = data.get("services")
    if not isinstance(services, dict):
        return

    for service_id in AUXILIARY_SERVICE_IDS:
        service = services.get(service_id)
        if isinstance(service, dict):
            service["include_in_global_actions"] = False


def _resolve_service_cwds(data: dict[str, Any], base_dir: Path) -> None:
    services = data.get("services")
    if not isinstance(services, dict):
        return

    for service in services.values():
        if not isinstance(service, dict):
            continue
        cwd = service.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            continue
        cwd_path = Path(cwd).expanduser()
        if cwd_path.is_absolute():
            service["cwd"] = str(cwd_path)
        else:
            service["cwd"] = str((base_dir / cwd_path).resolve())


def _config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def default_config_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    local_path = root / "configs" / "operator_dashboard.local.toml"
    if local_path.exists():
        return local_path
    return root / "configs" / "operator_dashboard.example.toml"
