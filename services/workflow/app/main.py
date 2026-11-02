from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import restate
from agentteam_common import Settings
from agentteam_contracts import (
    IssueRunRequest,
    IssueRunResult,
    ModelCallRequest,
    ModelOutcome,
    Sampling,
)
from fastapi import FastAPI, HTTPException
from restate.client_types import HttpError as RestateHttpError

from .models import Ledger, ModelClient, Store
from .plan import (
    build_issue_plan,
    build_issue_plan_v1,
    build_triage_proposal_v1,
    map_triage_return_v1,
    map_triage_run_v1,
)
from .retries import WaitDecision, decide_delay, decide_wait

settings = Settings.from_env()
issue_run = restate.Workflow("IssueRun")
issue_run_v1 = restate.Workflow("IssueRunV1")
_model_client: ModelClient | None = None
app = FastAPI(title="Agent Team Workflow")


def current_basis() -> str:
    return datetime.now(UTC).isoformat()


def call_model(request: ModelCallRequest) -> ModelOutcome:
    global _model_client
    if _model_client is None:
        store = Store(settings)
        _model_client = ModelClient(
            settings=settings,
            ledger=Ledger(settings, store),
            store=store,
            http=httpx.Client(),
            now=lambda: datetime.now(UTC),
        )
    return _model_client.call(request)


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


@issue_run_v1.main()
async def run_issue_v1(
    ctx: restate.WorkflowContext, request: IssueRunRequest
) -> IssueRunResult:
    if request.read_basis is None:
        basis = await ctx.run_typed("read-basis", current_basis)
        request = request.model_copy(
            update={"read_basis": datetime.fromisoformat(basis)}
        )
    plan = build_issue_plan_v1(request)

    await ctx.run_typed(
        "open-round",
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

    model_request = ModelCallRequest(
        run_id=plan.run_id,
        step="triage",
        role="triager",
        messages=[
            {
                "role": "user",
                "content": f"GitHub issue title: {request.title}\n\n{request.body}",
            }
        ],
        sampling=Sampling(),
        taint=2,
        source="github_issue",
    )
    for call_number in (1, 2):
        outcome = await ctx.run_typed(
            f"triage-call:{call_number}",
            call_model,
            restate.RunOptions(type_hint=ModelOutcome, max_attempts=2),
            request=model_request,
        )
        if outcome.outcome == "completed":
            break
        if outcome.outcome in ("failed", "ledger_unavailable"):
            raise restate.TerminalError(
                f"triager {outcome.outcome}: {outcome.error_code or 'unspecified'}"
            )
        if outcome.outcome == "throttled":
            if outcome.retry_in is None:
                raise restate.TerminalError("triager throttle is missing retry_in")
            if call_number == 2:
                return map_triage_return_v1(plan, outcome)
            wait = await ctx.run_typed(
                f"triage-throttle-wait:{call_number}",
                decide_delay,
                restate.RunOptions(type_hint=WaitDecision),
                delay_s=outcome.retry_in,
                cap_s=settings.triager_wait_cap_s,
            )
            if wait["action"] == "return":
                return map_triage_return_v1(plan, outcome)
            wait_s = wait["wait_s"]
            if wait_s is None:
                raise restate.TerminalError("triager wait decision is missing wait_s")
            await ctx.sleep(
                timedelta(seconds=wait_s),
                name=f"triage-throttle:{call_number}",
            )
            continue

        if outcome.retry_at is None:
            raise restate.TerminalError("triager quota stop is missing retry_at")
        wait = await ctx.run_typed(
            f"triage-wait:{call_number}",
            decide_wait,
            restate.RunOptions(type_hint=WaitDecision),
            retry_at=outcome.retry_at,
            decided_at=outcome.decided_at,
            cap_s=settings.triager_wait_cap_s,
        )
        if wait["action"] == "return" or call_number == 2:
            return map_triage_return_v1(plan, outcome)
        wait_s = wait["wait_s"]
        if wait_s is None:
            raise restate.TerminalError("triager wait decision is missing wait_s")
        await ctx.sleep(
            timedelta(seconds=wait_s),
            name=f"triage-quota:{call_number}",
        )

    else:
        raise restate.TerminalError("triager did not complete within two calls")

    run = map_triage_run_v1(plan, outcome)
    proposal = build_triage_proposal_v1(plan, outcome)
    await ctx.run_typed(
        "record-triager-run",
        post_memory,
        path="/v1/runs",
        payload=run.model_dump(mode="json"),
    )
    await ctx.run_typed(
        "write-route-proposal",
        post_memory,
        path=f"/v1/rounds/{plan.round_id}/proposals",
        payload=proposal.model_dump(mode="json"),
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


app.mount("/restate/v1", restate.app(services=[issue_run, issue_run_v1]))


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/issues/{issue_id}/run", response_model=IssueRunResult)
async def start_issue(issue_id: str, request: IssueRunRequest) -> IssueRunResult:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            return await client.workflow_call(run_issue_v1, key=issue_id, arg=request)
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error
