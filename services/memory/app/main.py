from typing import Any

from agentteam_contracts import (
    CommitRound,
    EventRecord,
    EventWrite,
    ProposalRecord,
    ProposalWrite,
    RoundRecord,
    RoundWrite,
    RunRecord,
    RunWrite,
)
from fastapi import FastAPI, HTTPException

from .store import IdempotencyConflict, MemoryStore, RoundConflict

app = FastAPI(title="Agent Team Memory")
store = MemoryStore()


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
