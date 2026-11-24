from collections.abc import Sequence
from typing import Any


def prompt_tokens_for_bytes(size: int, messages: int, *, strict: bool = False) -> int:
    if strict:
        # Allow one token per serialized byte plus text template headroom.
        return size + 32 * messages + 1024
    return (size + 3) // 4


def supports_strict_prompt(
    messages: Sequence[dict[str, Any]], *, tools: Any, tool_choice: Any
) -> bool:
    return tools is None and tool_choice is None and all(
        message.keys() == {"role", "content"}
        and message["role"] in ("system", "user", "assistant")
        and isinstance(message["content"], str)
        for message in messages
    )
