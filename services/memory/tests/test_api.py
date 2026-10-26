from datetime import UTC, datetime

import httpx
import pytest

from services.memory.app.main import app, store


@pytest.mark.anyio
async def test_event_write_returns_the_stored_record(monkeypatch):
    request = {
        "idempotency_key": "workflow-1:issue-event",
        "source": "github_issue",
        "kind": "issue",
        "taint": 2,
        "payload": {"title": "Example issue", "body": "Example body"},
    }
    expected = {
        "_id": "ev-example",
        "source": "github_issue",
        "kind": "issue",
        "taint": 2,
        "taint_label": "untrusted",
        "payload": request["payload"],
    }
    monkeypatch.setattr(store, "create_event", lambda _: expected)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/events", json=request)

    assert response.status_code == 200
    assert response.json() == {
        "id": "ev-example",
        **{k: v for k, v in expected.items() if k != "_id"},
    }


@pytest.mark.anyio
async def test_invalid_taint_is_rejected():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/events",
            json={
                "idempotency_key": "bad-taint",
                "source": "github_issue",
                "kind": "issue",
                "taint": 4,
                "payload": {},
            },
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_round_write_and_commit_are_exposed(monkeypatch):
    read_basis = datetime(2026, 10, 26, tzinfo=UTC).isoformat()
    round_row = {
        "_id": "round-example",
        "seq": 1,
        "status": "committed",
        "base_commit": None,
        "read_basis": read_basis,
        "committed_proposals": ["prop-example"],
    }
    monkeypatch.setattr(
        store,
        "create_round",
        lambda _: {**round_row, "status": "open", "committed_proposals": []},
    )
    monkeypatch.setattr(store, "commit_round", lambda *_: round_row)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        opened = await client.post(
            "/v1/rounds",
            json={
                "idempotency_key": "issue-1:round",
                "seq": 1,
                "read_basis": read_basis,
            },
        )
        committed = await client.post(
            "/v1/rounds/round-example/commit", json={"proposal_ids": ["prop-example"]}
        )

    assert opened.status_code == 200
    assert opened.json()["status"] == "open"
    assert committed.status_code == 200
    assert committed.json()["status"] == "committed"


@pytest.mark.anyio
async def test_run_read_returns_the_stored_record(monkeypatch):
    started_at = "2026-10-26T00:00:00Z"
    row = {
        "_id": "run-example",
        "agent_id": "agent-example",
        "role": "triager",
        "round_id": "round-example",
        "workflow_id": "workflow-example",
        "step": "triage",
        "model": None,
        "base_commit": None,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0,
        "outcome": "completed",
        "event_ids": ["ev-example"],
        "started_at": started_at,
        "ended_at": None,
    }
    monkeypatch.setattr(store, "get_run", lambda _: row, raising=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/runs/run-example")

    assert response.status_code == 200
    assert response.json() == {
        "id": "run-example",
        **{k: v for k, v in row.items() if k != "_id"},
    }


@pytest.mark.anyio
async def test_proposal_read_returns_the_stored_record(monkeypatch):
    row = {
        "_id": "prop-example",
        "round_id": "round-example",
        "agent_id": "agent-example",
        "run_id": "run-example",
        "op": "route_issue",
        "status": "pending",
        "reason": None,
        "evidence_event_ids": ["ev-example"],
        "payload": {"stub": True},
    }
    monkeypatch.setattr(store, "get_proposal", lambda _: row, raising=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/proposals/prop-example")

    assert response.status_code == 200
    assert response.json() == {
        "id": "prop-example",
        **{k: v for k, v in row.items() if k != "_id"},
    }


@pytest.mark.anyio
async def test_missing_run_and_proposal_reads_return_404(monkeypatch):
    monkeypatch.setattr(store, "get_run", lambda _: None, raising=False)
    monkeypatch.setattr(store, "get_proposal", lambda _: None, raising=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        run_response = await client.get("/v1/runs/missing-run")
        proposal_response = await client.get("/v1/proposals/missing-proposal")

    assert run_response.status_code == 404
    assert proposal_response.status_code == 404
