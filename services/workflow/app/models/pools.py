from dataclasses import dataclass

from agentteam_common import Settings


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


def pin_for(role: str, settings: Settings) -> Pin:
    pins = {
        "triager": Pin(
            role="triager",
            model=settings.triager_model,
            pool="groq",
            max_tokens=settings.triager_max_tokens,
            prompt_token_cap=settings.triager_prompt_token_cap,
            timeout_s=settings.triager_timeout_s,
            expected_generation_s=settings.triager_expected_generation_s,
        ),
        "reproducer": Pin(
            role="reproducer",
            model=settings.reproducer_model,
            pool="groq",
            max_tokens=settings.reproducer_max_tokens,
            prompt_token_cap=settings.reproducer_prompt_token_cap,
            timeout_s=settings.reproducer_timeout_s,
            expected_generation_s=settings.reproducer_expected_generation_s,
        ),
        "fixer": Pin(
            role="fixer",
            model=settings.fixer_model,
            pool="groq",
            max_tokens=settings.fixer_max_tokens,
            prompt_token_cap=settings.fixer_prompt_token_cap,
            timeout_s=settings.fixer_timeout_s,
            expected_generation_s=settings.fixer_expected_generation_s,
        ),
        "tester": Pin(
            role="tester",
            model=settings.tester_model,
            pool="groq",
            max_tokens=settings.tester_max_tokens,
            prompt_token_cap=settings.tester_prompt_token_cap,
            timeout_s=settings.tester_timeout_s,
            expected_generation_s=settings.tester_expected_generation_s,
        ),
    }
    if role not in pins:
        raise ValueError(f"unsupported model role: {role}")
    pin = pins[role]
    if pin.prompt_token_cap + pin.max_tokens > settings.groq_tpm:
        raise ValueError(f"{role} prompt cap plus max tokens exceeds GROQ_TPM")
    return pin


def pool_for(pin: Pin, settings: Settings) -> Pool:
    if pin.pool != "groq":
        raise ValueError(f"unsupported quota pool: {pin.pool}")
    return Pool(
        pool="groq",
        base_url=settings.groq_base_url,
        reset_tz=settings.groq_reset_tz,
        rpm=settings.groq_rpm,
        tpm=settings.groq_tpm,
        rpd=settings.groq_rpd,
        tpd=settings.groq_tpd,
    )
