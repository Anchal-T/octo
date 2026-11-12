from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import restate
from agentteam_common import Settings, load_agentteam_config, workflow_key
from agentteam_contracts import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
    IssueRunRequest,
    IssueRunRequestV2,
    IssueRunResult,
    IssueRunResultV2,
    ModelCallRequest,
    ModelOutcome,
    RunStartAccepted,
    RunStatus,
    RunStatusV2,
    Sampling,
)
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from restate.client_types import HttpError as RestateHttpError

from .issue_run_v2 import (
    approve_issue_v2,
    info_issue_v2,
    issue_run_v2,
    run_issue_v2,
)
from .models import Ledger, ModelClient, Store
from .models.pools import pin_for
from .plan import (
    build_issue_plan,
    build_issue_plan_v1,
    build_triage_proposal_v1,
    map_triage_commit_v1,
    map_triage_return_v1,
    map_triage_run_v1,
)
from .retries import WaitDecision, decide_delay, decide_wait
from .triage import parse_triage, triage_messages

settings = Settings.from_env()
agentteam_config = load_agentteam_config()
issue_run = restate.Workflow("IssueRun")
issue_run_v1 = restate.Workflow("IssueRunV1")
_model_client: ModelClient | None = None
app = FastAPI(title="Agent Team Workflow")


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, error: RequestValidationError
):
    details = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, error: HTTPException):
    if isinstance(error.__cause__, RecursionError):
        return JSONResponse(
            status_code=422, content={"detail": "request body is nested too deeply"}
        )
    return await http_exception_handler(request, error)


@app.exception_handler(RecursionError)
async def handle_recursion_error(request: Request, error: RecursionError):
    return JSONResponse(
        status_code=422, content={"detail": "request body is nested too deeply"}
    )


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
    try:
        allowed_areas = agentteam_config.repo(request.repo).allowed_area_names
    except ValueError as error:
        raise restate.TerminalError("repository is not configured") from error

    if request.read_basis is None:
        basis = await ctx.run_typed("read-basis", current_basis)
        request = request.model_copy(
            update={"read_basis": datetime.fromisoformat(basis)}
        )
    try:
        plan = build_issue_plan_v1(request)
    except (ValidationError, ValueError) as error:
        raise restate.TerminalError("invalid triager issue plan") from error

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

    triager_provider = pin_for("triager", settings).pool
    model_request = ModelCallRequest(
        run_id=plan.run_id,
        step="triage",
        role="triager",
        messages=triage_messages(
            request.title, request.body, allowed_areas=allowed_areas
        ),
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
                return map_triage_return_v1(plan, outcome, provider=triager_provider)
            wait = await ctx.run_typed(
                f"triage-throttle-wait:{call_number}",
                decide_delay,
                restate.RunOptions(type_hint=WaitDecision),
                delay_s=outcome.retry_in,
                cap_s=settings.triager_wait_cap_s,
            )
            if wait["action"] == "return":
                return map_triage_return_v1(plan, outcome, provider=triager_provider)
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
            return map_triage_return_v1(plan, outcome, provider=triager_provider)
        wait_s = wait["wait_s"]
        if wait_s is None:
            raise restate.TerminalError("triager wait decision is missing wait_s")
        await ctx.sleep(
            timedelta(seconds=wait_s),
            name=f"triage-quota:{call_number}",
        )

    else:
        raise restate.TerminalError("triager did not complete within two calls")

    triage = parse_triage(outcome.text, allowed_areas=allowed_areas)
    try:
        run = map_triage_run_v1(plan, outcome)
        proposal = build_triage_proposal_v1(plan, outcome, triage)
        result = map_triage_commit_v1(plan, outcome, triage, provider=triager_provider)
    except (ValidationError, ValueError) as error:
        raise restate.TerminalError("invalid triage result") from error
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
    return result


app.mount(
    "/restate/v1",
    restate.app(services=[issue_run, issue_run_v1, issue_run_v2]),
)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/issues/{issue_id}/run", response_model=IssueRunResult)
async def start_issue(issue_id: str, request: IssueRunRequest) -> IssueRunResult:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    key = workflow_key(issue_id, request.attempt)
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            return await client.workflow_call(run_issue_v1, key=key, arg=request)
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error


@app.post(
    "/v1/issues/{issue_id}/start",
    response_model=RunStartAccepted,
    status_code=202,
)
async def start_issue_async(
    issue_id: str, request: IssueRunRequest
) -> RunStartAccepted:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    key = workflow_key(issue_id, request.attempt)
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            handle = await client.workflow_send(
                run_issue_v1,
                key=key,
                arg=request,
            )
        return RunStartAccepted(
            issue_id=issue_id,
            invocation_id=handle.invocation_id,
            attempt=request.attempt,
        )
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error


@app.get(
    "/v1/issues/{issue_id}/run",
    response_model=IssueRunResult | RunStatus,
)
async def issue_run_status(
    issue_id: str,
    attempt: int = Query(default=1, ge=1, le=99),
) -> IssueRunResult | RunStatus | JSONResponse:
    key = workflow_key(issue_id, attempt)
    try:
        async with httpx.AsyncClient(
            base_url=settings.restate_ingress, timeout=5
        ) as client:
            response = await client.post(
                "/restate/output",
                json={
                    "target": "workflow",
                    "workflowName": "IssueRunV1",
                    "workflowKey": key,
                },
            )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error

    try:
        output = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Invalid Restate output") from error

    if response.status_code == 470 or (
        isinstance(output, dict) and output.get("message") == "not ready"
    ):
        status = RunStatus(issue_id=issue_id, attempt=attempt)
        return JSONResponse(
            status_code=202,
            content=status.model_dump(mode="json"),
        )
    if response.status_code == 404 or (
        isinstance(output, dict) and output.get("message") == "not found"
    ):
        raise HTTPException(status_code=404, detail="Workflow run not found")
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=output.get("message", "Restate request failed")
            if isinstance(output, dict)
            else "Restate request failed",
        )
    try:
        return IssueRunResult.model_validate(output)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Invalid Restate output") from error


@app.post("/v2/issues/{issue_id}/run", response_model=IssueRunResultV2)
async def start_issue_v2(issue_id: str, request: IssueRunRequestV2) -> IssueRunResultV2:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    key = workflow_key(issue_id, request.attempt)
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            return await client.workflow_call(run_issue_v2, key=key, arg=request)
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error


@app.post(
    "/v2/issues/{issue_id}/start",
    response_model=RunStartAccepted,
    status_code=202,
)
async def start_issue_v2_async(
    issue_id: str, request: IssueRunRequestV2
) -> RunStartAccepted:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    key = workflow_key(issue_id, request.attempt)
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            handle = await client.workflow_send(
                run_issue_v2,
                key=key,
                arg=request,
            )
        return RunStartAccepted(
            issue_id=issue_id,
            invocation_id=handle.invocation_id,
            attempt=request.attempt,
        )
    except RestateHttpError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error


@app.get(
    "/v2/issues/{issue_id}/run",
    response_model=IssueRunResultV2 | RunStatusV2,
)
async def issue_run_v2_status(
    issue_id: str,
    attempt: int = Query(default=1, ge=1, le=99),
) -> IssueRunResultV2 | RunStatusV2 | JSONResponse:
    key = workflow_key(issue_id, attempt)
    try:
        async with httpx.AsyncClient(
            base_url=settings.restate_ingress, timeout=5
        ) as http:
            response = await http.post(
                "/restate/output",
                json={
                    "target": "workflow",
                    "workflowName": "IssueRunV2",
                    "workflowKey": key,
                },
            )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error

    try:
        output = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Invalid Restate output") from error

    if response.status_code == 470 or (
        isinstance(output, dict) and output.get("message") == "not ready"
    ):
        status_v2 = RunStatusV2(issue_id=issue_id, attempt=attempt)
        try:
            async with restate.create_client(settings.restate_ingress) as client:
                status_v2 = await client.workflow_call(info_issue_v2, key=key, arg=None)
        except (RestateHttpError, httpx.HTTPError):
            status_v2 = RunStatusV2(issue_id=issue_id, attempt=attempt)
        return JSONResponse(
            status_code=202,
            content=status_v2.model_dump(mode="json"),
        )
    if response.status_code == 404 or (
        isinstance(output, dict) and output.get("message") == "not found"
    ):
        raise HTTPException(status_code=404, detail="Workflow run not found")
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=output.get("message", "Restate request failed")
            if isinstance(output, dict)
            else "Restate request failed",
        )
    try:
        return IssueRunResultV2.model_validate(output)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Invalid Restate output") from error


@app.post(
    "/v2/issues/{issue_id}/approve",
    response_model=ApprovalResult,
)
async def approve_issue_v2_route(
    issue_id: str, request: ApprovalRequest
) -> ApprovalResult:
    if issue_id != request.issue_id:
        raise HTTPException(status_code=422, detail="issue_id does not match the path")
    key = workflow_key(issue_id, request.attempt)
    decision = ApprovalDecision(
        approved=request.approved,
        by=request.by,
        reason=request.reason,
        head_sha=request.head_sha,
        review_id=request.review_id,
        source=request.source,
    )
    try:
        async with restate.create_client(settings.restate_ingress) as client:
            return await client.workflow_call(
                approve_issue_v2,
                key=key,
                arg=decision,
            )
    except RestateHttpError as error:
        if error.status_code == 409:
            return ApprovalResult(
                issue_id=issue_id,
                attempt=request.attempt,
                status="already_decided",
            )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.body or error.message,
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Restate is unavailable") from error
