from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AUXILIARY_SERVICE_IDS = frozenset({"wake_tuning", "voice_modulation"})


class ServiceState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    FAILED = "failed"


class CheckType(str, Enum):
    TCP = "tcp"
    HTTP = "http"
    PROCESS = "process"
    LOG_PATTERN = "log_pattern"


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def require_localhost(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost"}:
            raise ValueError("dashboard host must be 127.0.0.1 or localhost")
        return value


class LinkConfig(BaseModel):
    label: str
    url: str


class ReadyCheckConfig(BaseModel):
    type: CheckType
    label: str | None = None
    required: bool = True
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    url: str | None = None
    pattern: str | None = None
    timeout_s: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_shape(self) -> "ReadyCheckConfig":
        if self.type is CheckType.TCP and (not self.host or self.port is None):
            raise ValueError("tcp checks require host and port")
        if self.type is CheckType.HTTP and not self.url:
            raise ValueError("http checks require url")
        if self.type is CheckType.LOG_PATTERN and not self.pattern:
            raise ValueError("log_pattern checks require pattern")
        return self


class ServiceConfig(BaseModel):
    label: str
    cwd: str
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    include_in_global_actions: bool = True
    require_running_process: bool = True
    stop_command: list[str] | None = None
    ready_checks: list[ReadyCheckConfig] = Field(default_factory=list)
    ready_patterns: list[str] = Field(default_factory=list)
    links: list[LinkConfig] = Field(default_factory=list)
    startup_timeout_s: float = Field(default=120.0, gt=0)

    @field_validator("command")
    @classmethod
    def require_command(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("command must contain at least one item")
        return value


class DashboardConfig(BaseModel):
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    services: dict[str, ServiceConfig]

    @field_validator("services")
    @classmethod
    def require_services(
        cls, value: dict[str, ServiceConfig]
    ) -> dict[str, ServiceConfig]:
        if not value:
            raise ValueError("at least one service is required")
        return value


class ReadyCheckStatus(BaseModel):
    type: CheckType
    label: str
    required: bool = True
    ok: bool
    detail: str


class ServiceStatus(BaseModel):
    id: str
    label: str
    state: ServiceState
    pid: int | None = None
    last_exit_code: int | None = None
    command: list[str]
    ready_checks: list[ReadyCheckStatus] = Field(default_factory=list)
    links: list[LinkConfig] = Field(default_factory=list)
    recent_logs: list[str] = Field(default_factory=list)
    last_error: str | None = None
    detected_urls: list[str] = Field(default_factory=list)
    include_in_global_actions: bool = True


class DashboardStatus(BaseModel):
    services: dict[str, ServiceStatus]


ActionName = Literal["start", "stop", "restart"]
