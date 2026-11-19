import os
from dataclasses import dataclass, field

from .model_authority import (
    FIXER,
    GROQ_DEFAULTS,
    GROQ_ORIGIN,
    REPRODUCER,
    TESTER,
    TRIAGER,
    ModelAuthority,
    model_settings_from_env,
)


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
    fact_user_report_ttl_commits: int = 50
    fact_user_report_half_life_days: float = 30.0
    fact_min_effective_confidence: float = 0.6
    fact_user_report_initial_confidence: float = 0.8
    sandbox_image: str = "python:3.12-slim"
    sandbox_repo_path: str | None = None
    sandbox_network: str = "none"
    sandbox_cmd_timeout_s: float = 60
    triager_model: str = TRIAGER.model
    triager_max_tokens: int = TRIAGER.max_tokens
    triager_prompt_token_cap: int = TRIAGER.prompt_token_cap
    triager_timeout_s: float = TRIAGER.timeout_s
    triager_expected_generation_s: float = TRIAGER.expected_generation_s
    triager_wait_cap_s: float = 7200
    reproducer_model: str = REPRODUCER.model
    reproducer_max_tokens: int = REPRODUCER.max_tokens
    reproducer_prompt_token_cap: int = REPRODUCER.prompt_token_cap
    reproducer_timeout_s: float = REPRODUCER.timeout_s
    reproducer_expected_generation_s: float = REPRODUCER.expected_generation_s
    reproducer_turn_cap: int = 6
    fixer_model: str = FIXER.model
    fixer_max_tokens: int = FIXER.max_tokens
    fixer_prompt_token_cap: int = FIXER.prompt_token_cap
    fixer_timeout_s: float = FIXER.timeout_s
    fixer_expected_generation_s: float = FIXER.expected_generation_s
    fixer_turn_cap: int = 8
    fixer_max_attempts: int = 2
    tester_model: str = TESTER.model
    tester_max_tokens: int = TESTER.max_tokens
    tester_prompt_token_cap: int = TESTER.prompt_token_cap
    tester_timeout_s: float = TESTER.timeout_s
    tester_expected_generation_s: float = TESTER.expected_generation_s
    groq_base_url: str = GROQ_DEFAULTS.base_url
    groq_api_key: str = field(default="", repr=False)
    groq_reset_tz: str = GROQ_DEFAULTS.reset_tz
    groq_rpm: int = GROQ_DEFAULTS.rpm
    groq_tpm: int = GROQ_DEFAULTS.tpm
    groq_rpd: int = GROQ_DEFAULTS.rpd
    groq_tpd: int = GROQ_DEFAULTS.tpd
    models_free_hosts: tuple[str, ...] = (GROQ_ORIGIN,)
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
            fact_user_report_ttl_commits=int(
                os.getenv("FACT_USER_REPORT_TTL_COMMITS", "50")
            ),
            fact_user_report_half_life_days=float(
                os.getenv("FACT_USER_REPORT_HALF_LIFE_DAYS", "30")
            ),
            fact_min_effective_confidence=float(
                os.getenv("FACT_MIN_EFFECTIVE_CONFIDENCE", "0.6")
            ),
            fact_user_report_initial_confidence=float(
                os.getenv("FACT_USER_REPORT_INITIAL_CONFIDENCE", "0.8")
            ),
            sandbox_image=os.getenv("SANDBOX_IMAGE", "python:3.12-slim"),
            sandbox_repo_path=os.getenv("SANDBOX_REPO_PATH") or None,
            sandbox_network=os.getenv("SANDBOX_NETWORK", "none"),
            sandbox_cmd_timeout_s=float(os.getenv("SANDBOX_CMD_TIMEOUT_S", "60")),
            triager_wait_cap_s=float(os.getenv("TRIAGER_WAIT_CAP_S", "7200")),
            reproducer_turn_cap=int(os.getenv("REPRODUCER_TURN_CAP", "6")),
            fixer_turn_cap=int(os.getenv("FIXER_TURN_CAP", "8")),
            fixer_max_attempts=int(os.getenv("FIXER_MAX_ATTEMPTS", "2")),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            **model_settings_from_env(),
            models_sqlite_path=os.getenv(
                "MODELS_SQLITE_PATH", "/var/lib/agent-team/models/models.sqlite"
            ),
            models_busy_timeout_ms=int(os.getenv("MODELS_BUSY_TIMEOUT_MS", "5000")),
        )
        settings.validate_model_config(strict_all_roles=True)
        return settings

    def validate_model_config(
        self, *, strict_all_roles: bool = False
    ) -> ModelAuthority:
        authority = ModelAuthority(self, strict_all_roles=strict_all_roles)
        if self.triager_wait_cap_s <= 0:
            raise ValueError("triager timeout and generation settings are invalid")
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
        return authority
