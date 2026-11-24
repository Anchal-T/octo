import logging
import os
import re
from datetime import timedelta
from typing import Literal

import httpx
import restate
from agentteam_contracts import EventWrite
from pydantic import BaseModel

from .issue_run_v2 import post_memory

logger = logging.getLogger(__name__)
reconciler = restate.VirtualObject("WorkflowFailureReconcilerV1")
_RECONCILER_KEY = "issue-runs"
_BATCH_SIZE = 25
_INTERVAL = timedelta(minutes=1)
_INVOCATION_ID = re.compile(r"inv_[A-Za-z0-9_]+")


class InvocationFailure(BaseModel):
    id: str
    created_at: str
    target_service_key: str
    status: Literal["paused", "completed"]
    journal_commands_size: int | None = None
    retry_count: int | None = None
    last_failure_error_code: str | None = None
    last_failure_related_command_index: int | None = None

    def event(self) -> EventWrite:
        return EventWrite(
            idempotency_key=f"workflow-failure:{self.id}:{self.created_at}:{self.status}",
            source="tool",
            kind="failure_record",
            taint=0,
            payload={
                "invocation_id": self.id,
                "workflow_key": self.target_service_key,
                "status": self.status,
                "recovery_action": "resume"
                if self.status == "paused"
                else "new_attempt",
                "journal_commands_size": self.journal_commands_size,
                "retry_count": self.retry_count,
                "error_code": self.last_failure_error_code,
                "failed_command_index": self.last_failure_related_command_index,
            },
        )


async def reconcile_page(cursor: str) -> str:
    # A cyclic keyset scan lets older retained failures coexist with new ones.
    # The cursor is a Restate-generated ID from the previous page.
    if cursor and not _INVOCATION_ID.fullmatch(cursor):
        raise ValueError("invalid Restate invocation cursor")
    query = f"""
        SELECT id, CAST(created_at AS VARCHAR) AS created_at, target_service_key,
               status, journal_commands_size,
               retry_count, last_failure_error_code, last_failure_related_command_index
        FROM sys_invocation
        WHERE target_service_name = 'IssueRunV2' AND target_handler_name = 'run'
          AND (status = 'paused' OR (status = 'completed' AND completion_result = 'failure'))
          AND id > '{cursor}'
        ORDER BY id
        LIMIT {_BATCH_SIZE}
    """
    admin_url = os.environ.get("RESTATE_ADMIN_URL", "http://127.0.0.1:9070")
    async with httpx.AsyncClient(
        base_url=admin_url, timeout=5, trust_env=False
    ) as client:
        response = await client.post(
            "/query", json={"query": query}, headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        failures = [
            InvocationFailure.model_validate(row) for row in response.json()["rows"]
        ]

    for failure in failures:
        try:
            await post_memory("/v1/events", failure.event().model_dump(mode="json"))
        except httpx.HTTPStatusError as error:
            # Memory returns 409 when a prior observation of this invocation
            # already recorded different transient metadata under the same key.
            if error.response.status_code != 409:
                raise
    return failures[-1].id if len(failures) == _BATCH_SIZE else ""


@reconciler.handler()
async def start(ctx: restate.ObjectContext, _: None) -> None:
    if ctx.key() != _RECONCILER_KEY:
        raise restate.TerminalError(
            "Use the issue-runs reconciler key", status_code=400
        )
    if await ctx.get("next_slot", type_hint=int) is None:
        ctx.object_send(tick, key=ctx.key(), arg=0, idempotency_key=f"{ctx.key()}:0")
        ctx.set("next_slot", 0)


@reconciler.handler(ingress_private=True)
async def tick(ctx: restate.ObjectContext, slot: int) -> None:
    if await ctx.get("next_slot", type_hint=int) != slot:
        return
    next_slot = slot + 1
    ctx.object_send(
        tick,
        key=ctx.key(),
        arg=next_slot,
        send_delay=_INTERVAL,
        idempotency_key=f"{ctx.key()}:{next_slot}",
    )
    ctx.set("next_slot", next_slot)

    cursor = await ctx.get("cursor", type_hint=str) or ""
    try:
        next_cursor = await ctx.run_typed(
            "reconcile-failures",
            reconcile_page,
            restate.RunOptions(type_hint=str, max_attempts=2),
            cursor=cursor,
        )
    except restate.TerminalError:
        logger.warning("Workflow failure reconciliation failed for slot %s", slot)
    else:
        ctx.set("cursor", next_cursor)
