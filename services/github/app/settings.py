import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

GITHUB_LABEL = "agent-team"
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_PAGE_SIZE = 100
GITHUB_MAX_PAGES = 10
GITHUB_POLL_OVERLAP_S = 600
GITHUB_MAX_START_ATTEMPTS = 5
GITHUB_LEASE_TTL_S = 300
GITHUB_RATE_LIMIT_FLOOR = 50
GITHUB_SECONDARY_LIMIT_WAIT_CAP_S = 120
GITHUB_CONNECT_TIMEOUT_S = 5
GITHUB_READ_TIMEOUT_S = 20
WORKFLOW_START_TIMEOUT_S = 10
GITHUB_APP_JWT_BACKDATE_S = 60
GITHUB_APP_JWT_TTL_S = 540
GITHUB_APP_TOKEN_REFRESH_BUFFER_S = 300
FORBIDDEN_UPSTREAM_REPOS = frozenset(
    {
        "andialbrecht/sqlparse",
        "marshmallow-code/marshmallow",
    }
)

_TRUTHY_FALSY = {
    "true": True,
    "1": True,
    "yes": True,
    "false": False,
    "0": False,
    "no": False,
    "": False,
}


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class GithubSettings:
    token: str = field(default="", repr=False)
    app_id: str = ""
    app_private_key: str = field(default="", repr=False)
    app_private_key_path: str = ""
    app_installation_id: int | None = None
    approvers: tuple[str, ...] = ()
    repo: str = ""
    workflow_url: str = "http://workflow:8002"
    workflow_version: Literal["v1", "v2"] = "v1"
    sqlite_path: str = "/var/lib/agent-team/github/github.sqlite"
    poll_interval_s: float = 60
    max_starts_per_cycle: int = 1
    max_starts_per_day: int = 10
    body_max_chars: int = 8000
    process_own_issues: bool = False

    @property
    def auth_mode(self) -> Literal["app", "pat", "none"]:
        has_app = bool(
            self.app_id.strip()
            or self.app_private_key.strip()
            or self.app_private_key_path.strip()
            or self.app_installation_id is not None
        )
        if has_app:
            return "app"
        if self.token.strip():
            return "pat"
        return "none"

    def load_private_key_pem(self) -> str:
        if self.app_private_key.strip():
            return self.app_private_key.replace("\\n", "\n").strip()
        if self.app_private_key_path.strip():
            try:
                return (
                    Path(self.app_private_key_path).read_text(encoding="utf-8").strip()
                )
            except OSError as error:
                raise SettingsError(
                    "GITHUB_APP_PRIVATE_KEY_PATH could not be read"
                ) from error
        raise SettingsError(
            "GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_PATH must be set"
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GithubSettings":
        source = os.environ if env is None else env
        try:
            poll_interval_s = float(source.get("GITHUB_POLL_INTERVAL_S", "60"))
        except ValueError as error:
            raise SettingsError("GITHUB_POLL_INTERVAL_S must be a number") from error
        try:
            max_starts_per_cycle = int(source.get("GITHUB_MAX_STARTS_PER_CYCLE", "1"))
        except ValueError as error:
            raise SettingsError(
                "GITHUB_MAX_STARTS_PER_CYCLE must be an integer"
            ) from error
        try:
            max_starts_per_day = int(source.get("GITHUB_MAX_STARTS_PER_DAY", "10"))
        except ValueError as error:
            raise SettingsError(
                "GITHUB_MAX_STARTS_PER_DAY must be an integer"
            ) from error
        try:
            body_max_chars = int(source.get("GITHUB_BODY_MAX_CHARS", "8000"))
        except ValueError as error:
            raise SettingsError("GITHUB_BODY_MAX_CHARS must be an integer") from error
        raw_own_issues = source.get("GITHUB_PROCESS_OWN_ISSUES", "false")
        process_own_issues = _TRUTHY_FALSY.get(raw_own_issues.strip().lower())
        if process_own_issues is None:
            raise SettingsError("GITHUB_PROCESS_OWN_ISSUES must be true or false")

        raw_install_id = source.get("GITHUB_APP_INSTALLATION_ID", "").strip()
        if raw_install_id:
            try:
                app_installation_id: int | None = int(raw_install_id)
            except ValueError as error:
                raise SettingsError(
                    "GITHUB_APP_INSTALLATION_ID must be an integer"
                ) from error
        else:
            app_installation_id = None

        app_id = source.get("GITHUB_APP_ID", "").strip()
        app_private_key = source.get("GITHUB_APP_PRIVATE_KEY", "")
        app_private_key_path = source.get("GITHUB_APP_PRIVATE_KEY_PATH", "").strip()
        has_app = bool(
            app_id
            or app_private_key.strip()
            or app_private_key_path
            or app_installation_id is not None
        )
        default_wf_version = "v2" if has_app else "v1"
        raw_wf_version = (
            source.get("GITHUB_WORKFLOW_VERSION", default_wf_version).strip().lower()
        )
        if raw_wf_version not in {"v1", "v2"}:
            raise SettingsError("GITHUB_WORKFLOW_VERSION must be v1 or v2")
        workflow_version: Literal["v1", "v2"] = "v2" if raw_wf_version == "v2" else "v1"
        approvers = tuple(
            login.strip()
            for login in source.get("GITHUB_APPROVERS", "").split(",")
            if login.strip()
        )

        return cls(
            token=source.get("GITHUB_TOKEN", ""),
            app_id=app_id,
            app_private_key=app_private_key,
            app_private_key_path=app_private_key_path,
            app_installation_id=app_installation_id,
            approvers=approvers,
            repo=source.get("GITHUB_REPO", ""),
            workflow_url=source.get("WORKFLOW_URL", "http://workflow:8002").rstrip("/"),
            workflow_version=workflow_version,
            sqlite_path=source.get(
                "GITHUB_SQLITE_PATH", "/var/lib/agent-team/github/github.sqlite"
            ),
            poll_interval_s=poll_interval_s,
            max_starts_per_cycle=max_starts_per_cycle,
            max_starts_per_day=max_starts_per_day,
            body_max_chars=body_max_chars,
            process_own_issues=process_own_issues,
        )

    def validate_for_start(self) -> None:
        has_app = bool(
            self.app_id.strip()
            or self.app_private_key.strip()
            or self.app_private_key_path.strip()
            or self.app_installation_id is not None
        )
        if has_app and self.token.strip():
            raise SettingsError(
                "configure GitHub App credentials (GITHUB_APP_ID) or GITHUB_TOKEN, not both"
            )
        if has_app:
            if not self.app_id.strip():
                raise SettingsError("GITHUB_APP_ID must be set")
            if self.app_installation_id is not None and self.app_installation_id < 1:
                raise SettingsError("GITHUB_APP_INSTALLATION_ID must be positive")
            pem = self.load_private_key_pem()
            try:
                loaded = serialization.load_pem_private_key(
                    pem.encode("utf-8"),
                    password=None,
                )
            except (TypeError, ValueError) as error:
                raise SettingsError(
                    "GITHUB_APP_PRIVATE_KEY is not a valid unencrypted PEM key"
                ) from error
            if not isinstance(loaded, rsa.RSAPrivateKey):
                raise SettingsError("GITHUB_APP_PRIVATE_KEY must be an RSA private key")
        elif not self.token.strip():
            raise SettingsError("GITHUB_APP_ID or GITHUB_TOKEN must be set")

        if not self.repo.strip():
            raise SettingsError("GITHUB_REPO must be set")
        if re.fullmatch(r"[^/\s]+/[^/\s]+", self.repo) is None:
            raise SettingsError("GITHUB_REPO must be in owner/name form")
        if self.repo.strip().casefold() in FORBIDDEN_UPSTREAM_REPOS:
            raise SettingsError(
                "GITHUB_REPO must be a fork, never an upstream repository"
            )
        if self.workflow_version not in {"v1", "v2"}:
            raise SettingsError("GITHUB_WORKFLOW_VERSION must be v1 or v2")
        if not math.isfinite(self.poll_interval_s) or self.poll_interval_s < 30:
            raise SettingsError("GITHUB_POLL_INTERVAL_S must be at least 30")
        if self.max_starts_per_cycle < 1:
            raise SettingsError("GITHUB_MAX_STARTS_PER_CYCLE must be positive")
        if self.max_starts_per_day < self.max_starts_per_cycle:
            raise SettingsError(
                "GITHUB_MAX_STARTS_PER_DAY must be at least GITHUB_MAX_STARTS_PER_CYCLE"
            )
        if not 100 <= self.body_max_chars <= 100_000:
            raise SettingsError("GITHUB_BODY_MAX_CHARS must be between 100 and 100000")
