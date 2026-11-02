from datetime import datetime
from typing import Literal, TypedDict


class WaitDecision(TypedDict):
    action: Literal["sleep", "return"]
    wait_s: float | None


def decide_wait(
    *, retry_at: datetime, decided_at: datetime, cap_s: float
) -> WaitDecision:
    return decide_delay(delay_s=(retry_at - decided_at).total_seconds(), cap_s=cap_s)


def decide_delay(*, delay_s: float, cap_s: float) -> WaitDecision:
    delay_s = max(0.0, delay_s)
    if delay_s <= cap_s:
        return {"action": "sleep", "wait_s": delay_s}
    return {"action": "return", "wait_s": None}
