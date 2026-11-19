import os
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from .settings import Settings


@dataclass(frozen=True)
class Pool:
    pool: str
    base_url: str
    reset_tz: str
    rpm: int
    tpm: int
    rpd: int
    tpd: int


@dataclass(frozen=True)
class Pin:
    role: str
    model: str
    pool: str
    max_tokens: int
    prompt_token_cap: int
    timeout_s: float
    expected_generation_s: float


GROQ_ORIGIN = "https://api.groq.com"
GROQ_DEFAULTS = Pool("groq", f"{GROQ_ORIGIN}/openai/v1", "UTC", 30, 8000, 1000, 200000)
GPT_OSS_20B = "openai/gpt-oss-20b"
QWEN_27B = "qwen/qwen3.8-27b"
FREE_MODELS = frozenset({GPT_OSS_20B, QWEN_27B})

TRIAGER = Pin("triager", GPT_OSS_20B, "groq", 1024, 6000, 60.0, 15.0)
REPRODUCER = Pin("reproducer", GPT_OSS_20B, "groq", 1200, 5500, 60.0, 15.0)
FIXER = Pin("fixer", GPT_OSS_20B, "groq", 1500, 5500, 60.0, 15.0)
TESTER = Pin("tester", GPT_OSS_20B, "groq", 1024, 5500, 60.0, 15.0)

_ROLE_PROFILES = {
    "local": (TRIAGER, REPRODUCER, FIXER, TESTER),
    "compose": (
        TRIAGER,
        replace(REPRODUCER, model=QWEN_27B),
        replace(FIXER, model=QWEN_27B),
        replace(TESTER, model=QWEN_27B),
    ),
}
_ROLE_FIELDS = tuple(
    field.name for field in fields(Pin) if field.name not in {"role", "pool"}
)
_POOL_FIELDS = tuple(field.name for field in fields(Pool) if field.name != "pool")


def _env_value(
    name: str, default: str | int | float, *, empty_uses_default: bool
) -> str | int | float:
    value = os.getenv(name.upper())
    if value is None or (empty_uses_default and not value):
        return default
    return type(default)(value)


def model_settings_from_env() -> dict:
    profile = os.getenv("MODEL_AUTHORITY_PROFILE", "local")
    if profile not in _ROLE_PROFILES:
        raise ValueError("MODEL_AUTHORITY_PROFILE must be local or compose")
    empty_uses_default = profile == "compose"
    overrides = {}
    for pin in _ROLE_PROFILES[profile]:
        for field in _ROLE_FIELDS:
            name = f"{pin.role}_{field}"
            overrides[name] = _env_value(
                name, getattr(pin, field), empty_uses_default=empty_uses_default
            )
    for field in _POOL_FIELDS:
        name = f"groq_{field}"
        overrides[name] = _env_value(
            name, getattr(GROQ_DEFAULTS, field), empty_uses_default=empty_uses_default
        )
    overrides["groq_base_url"] = overrides["groq_base_url"].rstrip("/")
    free_hosts = _env_value(
        "models_free_hosts", GROQ_ORIGIN, empty_uses_default=empty_uses_default
    )
    overrides["models_free_hosts"] = tuple(
        host.strip().rstrip("/") for host in free_hosts.split(",") if host.strip()
    )
    return overrides


class ModelAuthority:
    def __init__(self, settings: "Settings", *, strict_all_roles: bool = False) -> None:
        self.pool = replace(
            GROQ_DEFAULTS,
            **{field: getattr(settings, f"groq_{field}") for field in _POOL_FIELDS},
        )
        self._pins = {
            pin.role: replace(
                pin,
                **{
                    field: getattr(settings, f"{pin.role}_{field}")
                    for field in _ROLE_FIELDS
                },
            )
            for pin in _ROLE_PROFILES["local"]
        }
        base = urlsplit(self.pool.base_url)
        origin = f"{base.scheme}://{base.netloc}"
        if (
            origin != GROQ_ORIGIN
            or origin not in settings.models_free_hosts
            or base.query
            or base.fragment
        ):
            raise ValueError(
                "GROQ_BASE_URL must be a Groq HTTPS host in MODELS_FREE_HOSTS"
            )
        if min(self.pool.rpm, self.pool.tpm, self.pool.rpd, self.pool.tpd) <= 0:
            raise ValueError("model token caps and Groq quota limits must be positive")
        try:
            ZoneInfo(self.pool.reset_tz)
        except ZoneInfoNotFoundError as error:
            raise ValueError("GROQ_RESET_TZ must be a valid timezone") from error
        for pin in self._pins.values():
            if pin.model not in FREE_MODELS:
                raise ValueError(
                    f"{pin.role.upper()}_MODEL must be a configured free Groq model"
                )
            if min(pin.max_tokens, pin.prompt_token_cap) <= 0:
                raise ValueError("model token caps and Groq quota limits must be positive")
            if not 0 <= pin.expected_generation_s <= pin.timeout_s or pin.timeout_s <= 0:
                raise ValueError(f"{pin.role} timeout and generation settings are invalid")
            if pin.role == "triager" or strict_all_roles:
                self._validate_budget(pin)

    def pin_for(self, role: str, *, model: str | None = None) -> Pin:
        pin = self._pins.get(role)
        if pin is None:
            raise ValueError(f"unsupported model role: {role}")
        if model is not None and model != pin.model:
            raise ValueError(f"requested model is not authorized for role: {role}")
        self._validate_budget(pin)
        return pin

    def _validate_budget(self, pin: Pin) -> None:
        if pin.prompt_token_cap + pin.max_tokens > self.pool.tpm:
            raise ValueError(f"{pin.role} prompt cap plus max tokens exceeds GROQ_TPM")
