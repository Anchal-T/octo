"""Synthetic IssueRunV2 and Memory endpoint for the isolated recovery test."""

import json
import os
from pathlib import Path

import restate
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.workflow.app.failure_reconciliation import reconciler

app = FastAPI()
issue_run = restate.Workflow("IssueRunV2")
evidence = Path(os.environ["RECOVERY_PROBE_EVIDENCE"])


def fail_fake_provider() -> None:
    with evidence.joinpath("attempts").open("a") as stream:
        stream.write("attempt\n")
    raise RuntimeError("synthetic provider failure")


@issue_run.main(name="run")
async def run(ctx: restate.WorkflowContext, _: None) -> None:
    await ctx.run_typed(
        "fake-provider", fail_fake_provider, restate.RunOptions(max_attempts=2)
    )


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
