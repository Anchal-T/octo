"""Synthetic IssueRunV2 and Memory endpoint for the isolated recovery test."""

import json
import os
from datetime import timedelta
from pathlib import Path

import restate
from agentteam_common import workflow_key
from agentteam_contracts import IssueRunRequestV2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.workflow.app.failure_reconciliation import reconciler
from services.workflow.app.issue_run_v2 import verify_recovery

app = FastAPI()
issue_run = restate.Workflow("IssueRunV2")
evidence = Path(os.environ["RECOVERY_PROBE_EVIDENCE"])


def fail_fake_provider() -> None:
    with evidence.joinpath("attempts").open("a") as stream:
        stream.write("attempt\n")
    raise RuntimeError("synthetic provider failure")


def record_checkpoint() -> str:
    with evidence.joinpath("checkpoint").open("a") as stream:
        stream.write("checkpoint\n")
    return "saved"


@issue_run.main(name="run")
async def run(ctx: restate.WorkflowContext, request: dict | None) -> dict | None:
    if request and request.get("title") == "Paused example":
        await ctx.run_typed("checkpoint", record_checkpoint)
        await ctx.sleep(timedelta(seconds=5), name="resume-check")
        return {"resumed": True}
    if request and request.get("recovery_from"):
        await ctx.run_typed(
            "verify-recovery",
            verify_recovery,
            restate.RunOptions(max_attempts=2),
            request=IssueRunRequestV2.model_validate(request),
        )
        return request["recovery_from"]
    await ctx.run_typed(
        "fake-provider", fail_fake_provider, restate.RunOptions(max_attempts=2)
    )


@app.post("/v2/issues/{issue_id}/start", status_code=202)
async def start_attempt(issue_id: str, request: IssueRunRequestV2):
    assert request.issue_id == issue_id
    async with restate.create_client(os.environ["RESTATE_INGRESS"]) as client:
        handle = await client.workflow_send(
            run,
            key=workflow_key(issue_id, request.attempt),
            arg=request.model_dump(mode="json"),
        )
    return {
        "issue_id": issue_id,
        "attempt": request.attempt,
        "invocation_id": handle.invocation_id,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/events")
async def record(request: Request):
    event = await request.json()
    try:
        with evidence.joinpath("event.json").open("x") as stream:
            json.dump(event, stream)
    except FileExistsError:
        return JSONResponse({}, status_code=409)
    return {"id": "fake-event"}


app.mount("/restate/v1", restate.app(services=[issue_run, reconciler]))
