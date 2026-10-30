import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class Settings:
    xtdb_host: str = "localhost"
    xtdb_port: int = 5432
    xtdb_database: str = "xtdb"
    xtdb_user: str = "xtdb"
    xtdb_password: str = ""
    memory_url: str = "http://127.0.0.1:8001"
    restate_ingress: str = "http://127.0.0.1:8080"
    scip_repo_path: str | None = None
    scip_project_name: str | None = None
    scip_cache_dir: str | None = None
    triager_model: str = "openai/gpt-oss-20b"
    triager_max_tokens: int = 1024
    triager_prompt_token_cap: int = 6000
    triager_timeout_s: float = 60
    triager_expected_generation_s: float = 15
    triager_wait_cap_s: float = 7200
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = field(default="", repr=False)
    groq_reset_tz: str = "UTC"
    groq_rpm: int = 30
    groq_tpm: int = 8000
    groq_rpd: int = 1000
    groq_tpd: int = 200000
    models_free_hosts: tuple[str, ...] = ("https://api.groq.com",)
    models_sqlite_path: str = "/var/lib/agent-team/models/models.sqlite"
    models_busy_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            xtdb_host=os.getenv("XTDB_HOST", "localhost"),
            xtdb_port=int(os.getenv("XTDB_PORT", "5432")),
            xtdb_database=os.getenv("XTDB_DATABASE", "xtdb"),
            xtdb_user=os.getenv("XTDB_USER", "xtdb"),
            xtdb_password=os.getenv("XTDB_PASSWORD", ""),
            memory_url=os.getenv("MEMORY_URL", "http://127.0.0.1:8001"),
            restate_ingress=os.getenv("RESTATE_INGRESS", "http://127.0.0.1:8080"),
            scip_repo_path=os.getenv("SCIP_REPO_PATH") or None,
            scip_project_name=os.getenv("SCIP_PROJECT_NAME") or None,
            scip_cache_dir=os.getenv("SCIP_CACHE_DIR") or None,
            triager_model=os.getenv("TRIAGER_MODEL", "openai/gpt-oss-20b"),
            triager_max_tokens=int(os.getenv("TRIAGER_MAX_TOKENS", "1024")),
            triager_prompt_token_cap=int(os.getenv("TRIAGER_PROMPT_TOKEN_CAP", "6000")),
            triager_timeout_s=float(os.getenv("TRIAGER_TIMEOUT_S", "60")),
            triager_expected_generation_s=float(
                os.getenv("TRIAGER_EXPECTED_GENERATION_S", "15")
            ),
            triager_wait_cap_s=float(os.getenv("TRIAGER_WAIT_CAP_S", "7200")),
            groq_base_url=os.getenv(
                "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            ).rstrip("/"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_reset_tz=os.getenv("GROQ_RESET_TZ", "UTC"),
            groq_rpm=int(os.getenv("GROQ_RPM", "30")),
            groq_tpm=int(os.getenv("GROQ_TPM", "8000")),
            groq_rpd=int(os.getenv("GROQ_RPD", "1000")),
            groq_tpd=int(os.getenv("GROQ_TPD", "200000")),
            models_free_hosts=tuple(
                host.strip().rstrip("/")
                for host in os.getenv(
                    "MODELS_FREE_HOSTS", "https://api.groq.com"
                ).split(",")
                if host.strip()
            ),
            models_sqlite_path=os.getenv(
                "MODELS_SQLITE_PATH", "/var/lib/agent-team/models/models.sqlite"
            ),
            models_busy_timeout_ms=int(os.getenv("MODELS_BUSY_TIMEOUT_MS", "5000")),
        )
        settings.validate_model_config()
        return settings

    def validate_model_config(self) -> None:
        base = urlsplit(self.groq_base_url)
        origin = f"{base.scheme}://{base.netloc}"
        if base.scheme != "https" or origin not in self.models_free_hosts:
            raise ValueError("GROQ_BASE_URL must be an HTTPS host in MODELS_FREE_HOSTS")
        if not self.models_free_hosts or not base.netloc or base.query or base.fragment:
            raise ValueError(
                "model provider settings must define a base URL and free host"
            )
        if not self.triager_model.strip():
            raise ValueError("TRIAGER_MODEL must be set to one pinned model id")
        if (
            min(
                self.triager_max_tokens,
                self.triager_prompt_token_cap,
                self.groq_rpm,
                self.groq_tpm,
                self.groq_rpd,
                self.groq_tpd,
            )
            <= 0
        ):
            raise ValueError("model token caps and Groq quota limits must be positive")
        if self.triager_prompt_token_cap + self.triager_max_tokens > self.groq_tpm:
            raise ValueError("triager prompt cap plus max tokens exceeds GROQ_TPM")
        if (
            self.triager_timeout_s <= 0
            or self.triager_expected_generation_s < 0
            or self.triager_expected_generation_s > self.triager_timeout_s
            or self.triager_wait_cap_s <= 0
        ):
            raise ValueError("triager timeout and generation settings are invalid")
        if not self.models_sqlite_path:
            raise ValueError("MODELS_SQLITE_PATH must not be empty")
        if self.models_busy_timeout_ms <= 0:
            raise ValueError("MODELS_BUSY_TIMEOUT_MS must be positive")
        try:
            ZoneInfo(self.groq_reset_tz)
        except ZoneInfoNotFoundError as error:
            raise ValueError("GROQ_RESET_TZ must be a valid timezone") from error
