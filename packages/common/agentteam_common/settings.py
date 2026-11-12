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
    github_url: str = "http://127.0.0.1:8003"
    restate_ingress: str = "http://127.0.0.1:8080"
    scip_repo_path: str | None = None
    scip_project_name: str | None = None
    scip_cache_dir: str | None = None
    sandbox_image: str = "python:3.12-slim"
    sandbox_repo_path: str | None = None
    sandbox_network: str = "none"
    sandbox_cmd_timeout_s: float = 60
    triager_model: str = "openai/gpt-oss-20b"
    triager_max_tokens: int = 1024
    triager_prompt_token_cap: int = 6000
    triager_timeout_s: float = 60
    triager_expected_generation_s: float = 15
    triager_wait_cap_s: float = 7200
    reproducer_model: str = "openai/gpt-oss-20b"
    reproducer_max_tokens: int = 1200
    reproducer_prompt_token_cap: int = 5500
    reproducer_timeout_s: float = 60
    reproducer_expected_generation_s: float = 15
    reproducer_turn_cap: int = 6
    fixer_model: str = "openai/gpt-oss-20b"
    fixer_max_tokens: int = 1500
    fixer_prompt_token_cap: int = 5500
    fixer_timeout_s: float = 60
    fixer_expected_generation_s: float = 15
    fixer_turn_cap: int = 8
    fixer_max_attempts: int = 2
    tester_model: str = "openai/gpt-oss-20b"
    tester_max_tokens: int = 1024
    tester_prompt_token_cap: int = 5500
    tester_timeout_s: float = 60
    tester_expected_generation_s: float = 15
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
            memory_url=os.getenv("MEMORY_URL", "http://127.0.0.1:8001").rstrip("/"),
            github_url=os.getenv("GITHUB_URL", "http://127.0.0.1:8003").rstrip("/"),
            restate_ingress=os.getenv("RESTATE_INGRESS", "http://127.0.0.1:8080"),
            scip_repo_path=os.getenv("SCIP_REPO_PATH") or None,
            scip_project_name=os.getenv("SCIP_PROJECT_NAME") or None,
            scip_cache_dir=os.getenv("SCIP_CACHE_DIR") or None,
            sandbox_image=os.getenv("SANDBOX_IMAGE", "python:3.12-slim"),
            sandbox_repo_path=os.getenv("SANDBOX_REPO_PATH") or None,
            sandbox_network=os.getenv("SANDBOX_NETWORK", "none"),
            sandbox_cmd_timeout_s=float(os.getenv("SANDBOX_CMD_TIMEOUT_S", "60")),
            triager_model=os.getenv("TRIAGER_MODEL", "openai/gpt-oss-20b"),
            triager_max_tokens=int(os.getenv("TRIAGER_MAX_TOKENS", "1024")),
            triager_prompt_token_cap=int(os.getenv("TRIAGER_PROMPT_TOKEN_CAP", "6000")),
            triager_timeout_s=float(os.getenv("TRIAGER_TIMEOUT_S", "60")),
            triager_expected_generation_s=float(
                os.getenv("TRIAGER_EXPECTED_GENERATION_S", "15")
            ),
            triager_wait_cap_s=float(os.getenv("TRIAGER_WAIT_CAP_S", "7200")),
            reproducer_model=os.getenv("REPRODUCER_MODEL", "openai/gpt-oss-20b"),
            reproducer_max_tokens=int(os.getenv("REPRODUCER_MAX_TOKENS", "1200")),
            reproducer_prompt_token_cap=int(
                os.getenv("REPRODUCER_PROMPT_TOKEN_CAP", "5500")
            ),
            reproducer_timeout_s=float(os.getenv("REPRODUCER_TIMEOUT_S", "60")),
            reproducer_expected_generation_s=float(
                os.getenv("REPRODUCER_EXPECTED_GENERATION_S", "15")
            ),
            reproducer_turn_cap=int(os.getenv("REPRODUCER_TURN_CAP", "6")),
            fixer_model=os.getenv("FIXER_MODEL", "openai/gpt-oss-20b"),
            fixer_max_tokens=int(os.getenv("FIXER_MAX_TOKENS", "1500")),
            fixer_prompt_token_cap=int(os.getenv("FIXER_PROMPT_TOKEN_CAP", "5500")),
            fixer_timeout_s=float(os.getenv("FIXER_TIMEOUT_S", "60")),
            fixer_expected_generation_s=float(
                os.getenv("FIXER_EXPECTED_GENERATION_S", "15")
            ),
            fixer_turn_cap=int(os.getenv("FIXER_TURN_CAP", "8")),
            fixer_max_attempts=int(os.getenv("FIXER_MAX_ATTEMPTS", "2")),
            tester_model=os.getenv("TESTER_MODEL", "openai/gpt-oss-20b"),
            tester_max_tokens=int(os.getenv("TESTER_MAX_TOKENS", "1024")),
            tester_prompt_token_cap=int(os.getenv("TESTER_PROMPT_TOKEN_CAP", "5500")),
            tester_timeout_s=float(os.getenv("TESTER_TIMEOUT_S", "60")),
            tester_expected_generation_s=float(
                os.getenv("TESTER_EXPECTED_GENERATION_S", "15")
            ),
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
        settings.validate_model_config(strict_all_roles=True)
        return settings

    def validate_model_config(self, *, strict_all_roles: bool = False) -> None:
        base = urlsplit(self.groq_base_url)
        origin = f"{base.scheme}://{base.netloc}"
        if base.scheme != "https" or origin not in self.models_free_hosts:
            raise ValueError("GROQ_BASE_URL must be an HTTPS host in MODELS_FREE_HOSTS")
        if not self.models_free_hosts or not base.netloc or base.query or base.fragment:
            raise ValueError(
                "model provider settings must define a base URL and free host"
            )
        role_configs = (
            (
                "triager",
                self.triager_model,
                self.triager_max_tokens,
                self.triager_prompt_token_cap,
                self.triager_timeout_s,
                self.triager_expected_generation_s,
            ),
            (
                "reproducer",
                self.reproducer_model,
                self.reproducer_max_tokens,
                self.reproducer_prompt_token_cap,
                self.reproducer_timeout_s,
                self.reproducer_expected_generation_s,
            ),
            (
                "fixer",
                self.fixer_model,
                self.fixer_max_tokens,
                self.fixer_prompt_token_cap,
                self.fixer_timeout_s,
                self.fixer_expected_generation_s,
            ),
            (
                "tester",
                self.tester_model,
                self.tester_max_tokens,
                self.tester_prompt_token_cap,
                self.tester_timeout_s,
                self.tester_expected_generation_s,
            ),
        )
        if min(self.groq_rpm, self.groq_tpm, self.groq_rpd, self.groq_tpd) <= 0:
            raise ValueError("model token caps and Groq quota limits must be positive")
        if self.triager_wait_cap_s <= 0:
            raise ValueError("triager timeout and generation settings are invalid")
        for (
            role_name,
            model_id,
            max_tokens,
            prompt_cap,
            timeout_s,
            expected_gen_s,
        ) in role_configs:
            if not model_id.strip():
                raise ValueError(
                    f"{role_name.upper()}_MODEL must be set to one pinned model id"
                )
            if min(max_tokens, prompt_cap) <= 0:
                raise ValueError(
                    "model token caps and Groq quota limits must be positive"
                )
            if (
                role_name == "triager" or strict_all_roles
            ) and prompt_cap + max_tokens > self.groq_tpm:
                raise ValueError(
                    f"{role_name} prompt cap plus max tokens exceeds GROQ_TPM"
                )
            if timeout_s <= 0 or expected_gen_s < 0 or expected_gen_s > timeout_s:
                raise ValueError(
                    f"{role_name} timeout and generation settings are invalid"
                )
        if (
            self.reproducer_turn_cap < 1
            or self.fixer_turn_cap < 1
            or not 1 <= self.fixer_max_attempts <= 2
            or self.sandbox_cmd_timeout_s <= 0
            or not self.sandbox_image.strip()
        ):
            raise ValueError("role turn caps and sandbox settings must be positive")
        if not self.models_sqlite_path:
            raise ValueError("MODELS_SQLITE_PATH must not be empty")
        if self.models_busy_timeout_ms <= 0:
            raise ValueError("MODELS_BUSY_TIMEOUT_MS must be positive")
        try:
            ZoneInfo(self.groq_reset_tz)
        except ZoneInfoNotFoundError as error:
            raise ValueError("GROQ_RESET_TZ must be a valid timezone") from error
