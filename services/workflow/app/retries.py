from datetime import datetime
from typing import Literal


def decide_wait(
    *, retry_at: datetime, decided_at: datetime, cap_s: float
) -> dict[str, float | Literal["sleep", "return"] | None]:
    wait_s = max(0.0, (retry_at - decided_at).total_seconds())
    if wait_s <= cap_s:
        return {"action": "sleep", "wait_s": wait_s}
    return {"action": "return", "wait_s": None}
