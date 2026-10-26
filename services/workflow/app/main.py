from datetime import UTC, datetime
from typing import Any

import httpx
import restate
from agentteam_common import Settings
from agentteam_contracts import IssueRunRequest, IssueRunResult
from fastapi import FastAPI, HTTPException
from restate.client_types import HttpError as RestateHttpError

from .plan import build_issue_plan

settings = Settings.from_env()
issue_run = restate.Workflow("IssueRun")
app = FastAPI(title="Agent Team Workflow")


def current_basis() -> str:
    return datetime.now(UTC).isoformat()


async def post_memory(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=settings.memory_url, timeout=10) as client:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


@issue_run.main()
async def run_issue(
    ctx: restate.WorkflowContext, request: IssueRunRequest
) -> IssueRunResult:
    if request.read_basis is None:
        basis = await ctx.run_typed("read-basis", current_basis)
        request = request.model_copy(
            update={"read_basis": datetime.fromisoformat(basis)}
        )
    plan = build_issue_plan(request)

    await ctx.run_typed(
        "create-round",
        post_memory,
        path="/v1/rounds",
        payload=plan.round.model_dump(mode="json"),
    )
    await ctx.run_typed(
        "record-issue",
        post_memory,
        path="/v1/events",
        payload=plan.event.model_dump(mode="json"),
    )
    await ctx.run_typed(
        "record-triage-run",
        post_memory,
        path="/v1/runs",
        payload=plan.run.model_dump(mode="json"),
    )
    await ctx.run_typed(
        "write-stub-proposal",
        post_memory,
        path=f"/v1/rounds/{plan.round_id}/proposals",
        payload=plan.proposal.model_dump(mode="json"),
    )
    await ctx.run_typed(
        "commit-round",
        post_memory,
        path=f"/v1/rounds/{plan.round_id}/commit",
        payload={},
    )

    return IssueRunResult(
        issue_id=request.issue_id,
        round_id=plan.round_id,
        event_id=plan.event_id,
        run_id=plan.run_id,
        proposal_id=plan.proposal_id,
        status="committed",
    )


app.mount("/restate/v1", restate.app(services=[issue_run]))


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/issues/{issue_id}/run", response_model=IssueRunResult)
async def start_issue(issue_id: str, request: IssueRunRequest) -> IssueRunResult:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            return await client.workflow_call(run_issue, key=issue_id, arg=request)
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error
