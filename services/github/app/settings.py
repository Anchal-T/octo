import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

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


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class GithubSettings:
    token: str = field(default="", repr=False)
    repo: str = ""
    workflow_url: str = "http://workflow:8002"
    sqlite_path: str = "/var/lib/agent-team/github/github.sqlite"
    poll_interval_s: float = 60
    max_starts_per_cycle: int = 1
    max_starts_per_day: int = 10
    body_max_chars: int = 8000

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

        return cls(
            token=source.get("GITHUB_TOKEN", ""),
            repo=source.get("GITHUB_REPO", ""),
            workflow_url=source.get("WORKFLOW_URL", "http://workflow:8002").rstrip("/"),
            sqlite_path=source.get(
                "GITHUB_SQLITE_PATH", "/var/lib/agent-team/github/github.sqlite"
            ),
            poll_interval_s=poll_interval_s,
            max_starts_per_cycle=max_starts_per_cycle,
            max_starts_per_day=max_starts_per_day,
            body_max_chars=body_max_chars,
        )

    def validate_for_start(self) -> None:
        if not self.token.strip():
            raise SettingsError("GITHUB_TOKEN must be set")
        if not self.repo.strip():
            raise SettingsError("GITHUB_REPO must be set")
        if re.fullmatch(r"[^/\s]+/[^/\s]+", self.repo) is None:
            raise SettingsError("GITHUB_REPO must be in owner/name form")
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
