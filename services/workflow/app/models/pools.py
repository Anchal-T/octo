from agentteam_common import Settings
from agentteam_common.model_authority import ModelAuthority, Pin, Pool

__all__ = ["Pin", "Pool", "pin_for"]


def pin_for(role: str, settings: Settings) -> Pin:
    return ModelAuthority(settings).pin_for(role)
