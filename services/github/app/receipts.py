import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from agentteam_common import stable_id
from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from .client import GithubFailure
from .schemas import Clock
from .store import Store, WriteConflictError

T = TypeVar("T")


def request_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        to_jsonable_python(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GithubReceipts:
    def __init__(self, store: Store, repo: str, clock: Clock) -> None:
        self.store = store
        self.repo = repo
        self.clock = clock

    async def run(
        self,
        operation_key: str,
        action: str,
        target: str,
        result_type: type[T],
        effect: Callable[..., Awaitable[T]],
        **payload: Any,
    ) -> T:
        key = stable_id("gh-write", json.dumps([self.repo, operation_key, action]))
        target = f"{self.repo}/{target}"
        digest = request_hash(payload)
        adapter = TypeAdapter(result_type)
        async with self.store.receipt_lock(key):
            receipt = self.store.get_side_effect_receipt(key)
            if receipt is not None:
                if receipt["target"] != target or receipt["request_sha256"] != digest:
                    raise WriteConflictError(
                        "idempotency_key was already used with a different request"
                    )
                if receipt["outcome"] == "ok":
                    return adapter.validate_json(receipt["response_json"])

            try:
                result = await effect(**payload)
            except GithubFailure as error:
                self.store.record_side_effect_receipt(
                    key,
                    action=action,
                    target=target,
                    request_sha256=digest,
                    outcome=error.outcome.outcome,
                    response_json=error.outcome.model_dump_json(),
                    decided_at=error.outcome.decided_at,
                )
                raise

            self.store.record_side_effect_receipt(
                key,
                action=action,
                target=target,
                request_sha256=digest,
                outcome="ok",
                response_json=adapter.dump_json(result).decode("utf-8"),
                decided_at=self.clock(),
            )
            return result
