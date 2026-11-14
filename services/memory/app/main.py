from typing import Any

from agentteam_contracts import (
    CloseByIssueRequest,
    CloseByIssueResponse,
    CommitRound,
    EventRecord,
    EventWrite,
    FactQuery,
    FactQueryResponse,
    FactView,
    ProposalRecord,
    ProposalWrite,
    RoundRecord,
    RoundWrite,
    RunRecord,
    RunWrite,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .dag import UnknownCommit
from .fact_query import FactExpiryConfig, filter_valid
from .store import IdempotencyConflict, MemoryStore, RoundConflict

app = FastAPI(title="Agent Team Memory")
store = MemoryStore()


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


def _event(row: dict[str, Any]) -> EventRecord:
    return EventRecord(
        id=row["_id"],
        source=row["source"],
        kind=row["kind"],
        taint=row["taint"],
        taint_label=row["taint_label"],
        payload=row["payload"],
    )


def _round(row: dict[str, Any]) -> RoundRecord:
    return RoundRecord(
        id=row["_id"],
        seq=row["seq"],
        status=row["status"],
        base_commit=row.get("base_commit"),
        read_basis=row["read_basis"],
        committed_proposals=list(row.get("committed_proposals") or []),
    )


def _run(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        id=row["_id"],
        agent_id=row["agent_id"],
        role=row["role"],
        round_id=row["round_id"],
        workflow_id=row["workflow_id"],
        step=row["step"],
        model=row.get("model"),
        base_commit=row.get("base_commit"),
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cost_usd=row["cost_usd"],
        outcome=row["outcome"],
        event_ids=list(row.get("event_ids") or []),
        started_at=row["started_at"],
        ended_at=row.get("ended_at"),
    )


def _proposal(row: dict[str, Any]) -> ProposalRecord:
    return ProposalRecord(
        id=row["_id"],
        round_id=row["round_id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        op=row["op"],
        status=row["status"],
        reason=row.get("reason"),
        evidence_event_ids=list(row.get("evidence_event_ids") or []),
        payload=row["payload"],
    )


@app.get("/healthz")
def health() -> dict[str, str]:
    try:
        store.ping()
    except Exception as error:
        raise HTTPException(status_code=503, detail="XTDB is unavailable") from error
    return {"status": "ok"}


@app.post("/v1/events", response_model=EventRecord)
def write_event(request: EventWrite) -> EventRecord:
    try:
        return _event(store.create_event(request))
    except IdempotencyConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/v1/events/{event_id}", response_model=EventRecord)
def read_event(event_id: str) -> EventRecord:
    row = store.get_event(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="event not found")
    return _event(row)


@app.post("/v1/rounds", response_model=RoundRecord)
def write_round(request: RoundWrite) -> RoundRecord:
    try:
        return _round(store.create_round(request))
    except IdempotencyConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/v1/rounds/{round_id}", response_model=RoundRecord)
def read_round(round_id: str) -> RoundRecord:
    row = store.get_round(round_id)
    if row is None:
        raise HTTPException(status_code=404, detail="round not found")
    return _round(row)


@app.get("/v1/runs/{run_id}", response_model=RunRecord)
def read_run(run_id: str) -> RunRecord:
    row = store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run(row)


@app.post("/v1/runs", status_code=201)
def write_run(request: RunWrite) -> dict[str, str]:
    try:
        row = store.create_run(request)
    except (IdempotencyConflict, RoundConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"id": row["_id"]}


@app.post("/v1/rounds/{round_id}/proposals", response_model=ProposalRecord)
def write_proposal(round_id: str, request: ProposalWrite) -> ProposalRecord:
    if round_id != request.round_id:
        raise HTTPException(status_code=422, detail="round_id does not match the path")
    try:
        return _proposal(store.create_proposal(request))
    except (IdempotencyConflict, RoundConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/v1/proposals/{proposal_id}", response_model=ProposalRecord)
def read_proposal(proposal_id: str) -> ProposalRecord:
    row = store.get_proposal(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    return _proposal(row)


@app.post("/v1/rounds/{round_id}/commit", response_model=RoundRecord)
def commit_round(round_id: str, request: CommitRound) -> RoundRecord:
    try:
        return _round(store.commit_round(round_id))
    except RoundConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/facts/query", response_model=FactQueryResponse)
def query_facts(request: FactQuery) -> FactQueryResponse:
    try:
        dag = store.get_dag()
    except Exception as error:
        raise HTTPException(status_code=503, detail="commit DAG unavailable") from error

    if request.base_commit not in dag.commits:
        raise HTTPException(status_code=404, detail="unknown base_commit")

    try:
        raw_facts = store.facts_for_symbols(request.symbols)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    fact_ids = [f["_id"] for f in raw_facts if "_id" in f or "id" in f]
    closures_rows = store.closures_for_facts(fact_ids)
    closures: dict[str, set[str]] = {fid: set() for fid in fact_ids}
    for row in closures_rows:
        closures[row["fact_id"]].add(row["closing_commit"])

    cfg = FactExpiryConfig(
        user_report_ttl_commits=store.settings.fact_user_report_ttl_commits,
        user_report_half_life_days=store.settings.fact_user_report_half_life_days,
        min_effective_confidence=store.settings.fact_min_effective_confidence,
        user_report_initial_confidence=store.settings.fact_user_report_initial_confidence,
    )
    base_time = dag.commits[request.base_commit].committed_at
    try:
        valid_facts, dropped = filter_valid(
            raw_facts,
            closures,
            dag,
            request.base_commit,
            ancestry_filter=request.ancestry_filter,
            cfg=cfg,
            base_time=base_time,
        )
    except (UnknownCommit, KeyError) as error:
        raise HTTPException(status_code=404, detail="unknown base_commit") from error

    if request.limit is not None and request.limit > 0:
        valid_facts = valid_facts[: request.limit]

    fact_views = [
        FactView(
            id=f.get("_id") or f.get("id"),
            claim=f.get("claim") or "",
            kind=f.get("kind") or "",
            area=f.get("area") or "",
            anchors=f.get("anchors") or [],
            valid_from_commit=f["valid_from_commit"],
            closing_commits=sorted(
                closures.get(f.get("_id") or f.get("id"))
                or f.get("closing_commits")
                or []
            ),
            confidence=float(f.get("confidence") if f.get("confidence") is not None else 1.0),
            decay_half_life_days=float(
                f.get("decay_half_life_days")
                if f.get("decay_half_life_days") is not None
                else 90.0
            ),
            last_confirmed_at=f.get("last_confirmed_at"),
            taint=int(f.get("taint", 0)),
            evidence_test=f.get("evidence_test"),
            source_issue=f.get("source_issue"),
            supersedes=f.get("supersedes"),
        )
        for f in valid_facts
    ]

    return FactQueryResponse(
        base_commit=request.base_commit,
        facts=fact_views,
        dropped=dropped,
    )


@app.post("/v1/facts/close-by-issue", response_model=CloseByIssueResponse)
def close_by_issue(request: CloseByIssueRequest) -> CloseByIssueResponse:
    try:
        closed_fact_ids = store.close_facts_by_issue(
            issue_ref=request.issue_ref,
            closing_commit=request.closing_commit,
            reason=request.reason,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return CloseByIssueResponse(closed_fact_ids=closed_fact_ids)
