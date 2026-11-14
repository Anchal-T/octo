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
@pytest.mark.parametrize(
    "body",
    [
        (
            b'{"idempotency_key":"\\ud800","source":"github_issue",'
            b'"kind":"issue","taint":0,"payload":{}}'
        ),
        (
            b'{"idempotency_key":"nan","source":"github_issue",'
            b'"kind":"issue","taint":0,"payload":{"value":NaN}}'
        ),
        (
            b'{"idempotency_key":"nested","source":"github_issue",'
            b'"kind":"issue","taint":0,"payload":'
            + b"[" * 100
            + b"0"
            + b"]" * 100
            + b"}"
        ),
        (
            b'{"idempotency_key":"deep","source":"github_issue",'
            b'"kind":"issue","taint":0,"payload":'
            + b"[" * 1200
            + b"0"
            + b"]" * 1200
            + b"}"
        ),
    ],
)
async def test_hostile_event_bodies_return_422_before_storage(monkeypatch, body):
    monkeypatch.setattr(
        store, "create_event", lambda _: pytest.fail("invalid body reached storage")
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/events",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


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
    monkeypatch.setattr(store, "commit_round", lambda _: round_row)

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
        committed = await client.post("/v1/rounds/round-example/commit", json={})
        caller_selected = await client.post(
            "/v1/rounds/round-example/commit",
            json={"proposal_ids": ["prop-example"]},
        )

    assert opened.status_code == 200
    assert opened.json()["status"] == "open"
    assert committed.status_code == 200
    assert committed.json()["status"] == "committed"
    assert caller_selected.status_code == 422


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
        "fact_ids_injected": [],
        "fact_ids_cited": [],
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


@pytest.mark.anyio
async def test_1_7_query_facts_unknown_base_returns_404(monkeypatch):
    from services.memory.app.dag import Commit, Dag

    c0 = Commit("known-sha", (), datetime(2026, 1, 1, tzinfo=UTC), "main", "test", ())
    dag = Dag.from_commits([c0])
    monkeypatch.setattr(store, "get_dag", lambda: dag)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/facts/query",
            json={"symbols": ["my_symbol"], "base_commit": "unknown-sha"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown base_commit"}


@pytest.mark.anyio
async def test_1_7_query_facts_nul_in_symbol_returns_422():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/facts/query",
            json={"symbols": ["bad\x00symbol"], "base_commit": "known-sha"},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_query_facts_success(monkeypatch):
    from services.memory.app.dag import Commit, Dag

    c0 = Commit("sha-0", (), datetime(2026, 1, 1, tzinfo=UTC), "main", "test", ())
    c1 = Commit(
        "sha-1", ("sha-0",), datetime(2026, 1, 2, tzinfo=UTC), "main", "test", ()
    )
    dag = Dag.from_commits([c0, c1])
    fact = {
        "_id": "fact-1a2b3c4d5e6f",
        "valid_from_commit": "sha-0",
        "claim": "Lexer tokenizes correctly",
        "kind": "bug_cause",
        "area": "lexer",
        "anchors": [{"symbol": "my_symbol"}],
        "confidence": 0.8,
        "decay_half_life_days": 90.0,
        "last_confirmed_at": "2026-11-10T12:00:00Z",
        "taint": 0,
        "evidence_test": "tests/test_x.py::test_y",
    }
    monkeypatch.setattr(store, "get_dag", lambda: dag)
    monkeypatch.setattr(store, "facts_for_symbols", lambda symbols, **kwargs: [fact])
    monkeypatch.setattr(store, "closures_for_facts", lambda ids: [])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/facts/query",
            json={
                "symbols": ["my_symbol"],
                "base_commit": "sha-1",
                "ancestry_filter": True,
                "limit": 50,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["base_commit"] == "sha-1"
    assert len(data["facts"]) == 1
    assert data["facts"][0]["id"] == "fact-1a2b3c4d5e6f"
    assert data["facts"][0]["claim"] == "Lexer tokenizes correctly"
    assert data["dropped"] == {"not_ancestor": 0, "closed": 0, "expired": 0}


@pytest.mark.anyio
async def test_resolve_symbols_uses_base_index_and_innermost_location(monkeypatch):
    from services.memory.app.scip import ScipDefinition, ScipIndex

    base = "a" * 40  # synthetic base
    symbol = "scip-python python sqlparse 0 `sqlparse.lexer`/Lexer#get_tokens()."
    class_symbol = "scip-python python sqlparse 0 `sqlparse.lexer`/Lexer#"
    index = ScipIndex(
        {
            symbol: (
                ScipDefinition(symbol, "sqlparse/lexer.py", (4, 0, 1), (4, 0, 8, 0)),
            ),
            class_symbol: (
                ScipDefinition(
                    class_symbol, "sqlparse/lexer.py", (0, 0, 1), (0, 0, 10, 0)
                ),
            ),
        },
        sha=base,
    )
    calls = []

    class Indexer:
        def load(self, sha):
            calls.append(sha)
            return index

    monkeypatch.setattr(store, "scip_indexer", Indexer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/symbols/resolve",
            json={
                "base_commit": base,
                "names": ["sqlparse.lexer.Lexer.get_tokens", "missing"],
                "locations": [{"file": "sqlparse/lexer.py", "line": 6}],
            },
        )
        hostile = await client.post(
            "/v1/symbols/resolve",
            json={
                "base_commit": base,
                "locations": [{"file": "../escape.py", "line": 1}],
            },
        )
    assert response.status_code == 200
    assert response.json() == {"symbols": [symbol]}
    assert hostile.status_code == 422
    assert calls == [base]


@pytest.mark.anyio
async def test_run_write_and_read_expose_injected_and_cited_fact_ids(monkeypatch):
    rows = {}

    def create(request):
        rows["run-example"] = {"_id": "run-example", **request.model_dump()}
        return rows["run-example"]

    monkeypatch.setattr(store, "create_run", create)
    monkeypatch.setattr(store, "get_run", rows.get)
    payload = {
        "idempotency_key": "example-run",
        "agent_id": "reproducer",
        "role": "reproducer",
        "round_id": "round-example",
        "workflow_id": "example-workflow",
        "step": "repro",
        "outcome": "reproduced",
        "started_at": "2026-11-15T00:00:00Z",
        "fact_ids_injected": ["fact-111111111111", "fact-222222222222"],
        "fact_ids_cited": ["fact-222222222222"],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        written = await client.post("/v1/runs", json=payload)
        read = await client.get("/v1/runs/run-example")
    assert written.status_code == 201
    assert read.status_code == 200
    assert read.json()["fact_ids_injected"] == payload["fact_ids_injected"]
    assert read.json()["fact_ids_cited"] == payload["fact_ids_cited"]


@pytest.mark.anyio
async def test_issue_scoped_reports_share_validity_filters_and_deduplicate(monkeypatch):
    from datetime import timedelta

    from services.memory.app.dag import Commit, Dag

    now = datetime(2026, 11, 15, tzinfo=UTC)
    shas = [f"{i:040x}" for i in range(51)]  # synthetic linear main history
    side = "f" * 40
    dag = Dag(
        [
            Commit(sha, () if i == 0 else (shas[i - 1],), now, "main", None, ())
            for i, sha in enumerate(shas)
        ]
        + [Commit(side, (shas[0],), now, "side", None, ())]
    )

    def report(fid, **overrides):
        return {
            "_id": fid,
            "claim": "Example user report",
            "kind": "user_report",
            "area": "lexer",
            "anchors": [],
            "valid_from_commit": shas[-1],
            "confidence": 0.8,
            "decay_half_life_days": 30,
            "last_confirmed_at": now,
            "taint": 2,
            "source_issue": "example/fork#7",
            **overrides,
        }

    reports = [
        report("fact-111111111111"),
        report("fact-222222222222", valid_from_commit=side),
        report("fact-333333333333"),
        report("fact-444444444444", valid_from_commit=shas[0]),
        report("fact-555555555555", last_confirmed_at=now - timedelta(days=13)),
    ]
    requested = []

    def for_issue(ref):
        requested.append(ref)
        return reports + [report("not-a-report", kind="bug_cause")]

    monkeypatch.setattr(store, "get_dag", lambda: dag)
    monkeypatch.setattr(
        store, "facts_for_symbols", lambda *_args, **_kwargs: [reports[0]]
    )
    monkeypatch.setattr(store, "facts_for_issue", for_issue)
    monkeypatch.setattr(
        store,
        "closures_for_facts",
        lambda _ids: [{"fact_id": reports[2]["_id"], "closing_commit": shas[-1]}],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/facts/query",
            json={"base_commit": shas[-1], "issue_refs": ["example/fork#7"]},
        )
    assert response.status_code == 200
    assert requested == ["example/fork#7"]
    assert [fact["id"] for fact in response.json()["facts"]] == [reports[0]["_id"]]
    assert response.json()["dropped"] == {"not_ancestor": 1, "closed": 1, "expired": 2}
