import json
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import restate
from agentteam_common import stable_id
from agentteam_common.cli import run_cli

from services.workflow.app import failure_reconciliation


class FakeObjectContext:
    def __init__(self, state, scheduled):
        self.state = state
        self.scheduled = scheduled

    def key(self):
        return "issue-runs"

    async def get(self, key, *, type_hint=None):
        return self.state.get(key)

    def set(self, key, value):
        self.state[key] = value

    def object_send(self, handler, *, key, arg, send_delay=None, idempotency_key=None):
        assert handler is failure_reconciliation.tick
        assert key == self.key()
        assert idempotency_key == f"issue-runs:{arg}"
        self.scheduled.append((arg, send_delay))

    async def run_typed(self, name, fn, options, **kwargs):
        assert options.max_attempts == 2
        return await fn(**kwargs)


@pytest.mark.anyio
async def test_reconciler_records_a_failed_invocation_after_worker_restart(monkeypatch):
    events = {}
    admin_queries = []
    state = {}
    scheduled = []
    failed_invocation = {
        "id": "inv_failed",
        "created_at": "2026-11-24T12:00:00Z",
        "target_service_key": "gh-1-8-r2",
        "status": "paused",
        "journal_commands_size": 12,
        "retry_count": 3,
        "last_failure_error_code": "500",
        "last_failure_related_command_index": 9,
    }

    def transport(request):
        if request.url.path == "/query":
            query = json.loads(request.content)["query"]
            assert request.headers["accept"] == "application/json"
            assert "target_service_name = 'IssueRunV2'" in query
            assert "target_handler_name = 'run'" in query
            assert (
                "status = 'paused'" in query
                and "completion_result = 'failure'" in query
            )
            admin_queries.append(query)
            return httpx.Response(200, json={"rows": [failed_invocation]})

        assert request.url.path == "/v1/events"
        event = json.loads(request.content)
        event_id = stable_id("ev", event["idempotency_key"])
        if event_id in events:
            return httpx.Response(409)
        events[event_id] = event
        return httpx.Response(200, json={"id": event_id})

    client_type = httpx.AsyncClient

    def mock_client(**kwargs):
        return client_type(transport=httpx.MockTransport(transport), **kwargs)

    monkeypatch.setattr(failure_reconciliation.httpx, "AsyncClient", mock_client)

    original_worker = FakeObjectContext(state, scheduled)
    await failure_reconciliation.start(original_worker, None)
    await failure_reconciliation.start(original_worker, None)
    assert scheduled == [(0, None)]
    del original_worker

    restarted_worker = FakeObjectContext(state, scheduled)
    await failure_reconciliation.tick(restarted_worker, 0)
    await failure_reconciliation.tick(restarted_worker, 0)
    await failure_reconciliation.tick(restarted_worker, 1)

    assert len(events) == 1
    event = next(iter(events.values()))
    assert event["kind"] == "failure_record"
    assert event["payload"] == {
        "invocation_id": "inv_failed",
        "workflow_key": "gh-1-8-r2",
        "status": "paused",
        "recovery_action": "resume",
        "journal_commands_size": 12,
        "retry_count": 3,
        "error_code": "500",
        "failed_command_index": 9,
    }
    assert scheduled == [
        (0, None),
        (1, timedelta(minutes=1)),
        (2, timedelta(minutes=1)),
    ]
    assert len(admin_queries) == 2


@pytest.mark.anyio
async def test_reconciler_schedules_next_tick_when_admin_query_fails():
    class UnavailableAdmin(FakeObjectContext):
        async def run_typed(self, name, fn, options, **kwargs):
            raise restate.TerminalError("admin unavailable")

    state = {}
    scheduled = []
    ctx = UnavailableAdmin(state, scheduled)
    await failure_reconciliation.start(ctx, None)
    await failure_reconciliation.tick(ctx, 0)
    assert scheduled == [(0, None), (1, timedelta(minutes=1))]


@pytest.mark.skipif(
    os.environ.get("AGENTTEAM_TEST_DURABILITY") != "1",
    reason="opt-in Restate 1.8.1 crash test",
)
def test_pinned_restate_recovers_failed_and_paused_workflows(tmp_path, capsys):
    from services.workflow.tests.test_durability_integration import (
        IsolatedRestate,
        _poll_until,
    )

    root = Path(__file__).resolve().parents[3]
    server = IsolatedRestate(tmp_path)
    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
    worker_url = f"http://127.0.0.1:{port}"
    worker = None

    def spawn_worker():
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "services.workflow.tests.failure_reconciliation_probe:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=root,
            env={
                **os.environ,
                "GROQ_API_KEY": "",
                "MEMORY_URL": worker_url,
                "RESTATE_ADMIN_URL": server.admin,
                "RESTATE_INGRESS": server.ingress,
                "RECOVERY_PROBE_EVIDENCE": str(tmp_path),
            },
        )

    def require_success(response):
        assert response.is_success, (response.status_code, response.text)
        return response.json()

    try:
        server.start()
        with httpx.Client(timeout=5, trust_env=False) as client:
            worker = spawn_worker()
            _poll_until(
                "fixture worker",
                lambda: client.get(f"{worker_url}/health").status_code,
                lambda status: status == 200,
            )
            require_success(
                client.post(
                    f"{server.admin}/deployments",
                    json={
                        "uri": f"http://host.docker.internal:{port}/restate/v1",
                        "use_http_11": True,
                    },
                )
            )
            require_success(
                client.post(
                    f"{server.ingress}/IssueRunV2/example-r2/run/send",
                    content="null",
                    headers={"Content-Type": "application/json"},
                )
            )

            def failed_invocation():
                query = (
                    "SELECT id FROM sys_invocation "
                    "WHERE target_service_name = 'IssueRunV2' AND status = 'completed' "
                    "AND completion_result = 'failure'"
                )
                return require_success(
                    client.post(
                        f"{server.admin}/query",
                        json={"query": query},
                        headers={"Accept": "application/json"},
                    )
                )["rows"]

            failed = _poll_until("failed invocation", failed_invocation, bool)[0]
            worker.kill()
            worker.wait(timeout=10)
            worker = None
            event_path = tmp_path / "event.json"
            assert not event_path.exists()

            require_success(
                client.post(
                    f"{server.ingress}/WorkflowFailureReconcilerV1/issue-runs/start/send",
                    content="null",
                    headers={"Content-Type": "application/json"},
                )
            )
            worker = spawn_worker()
            _poll_until(
                "failure record",
                lambda: event_path.is_file() and event_path.stat().st_size,
                bool,
            )
            event = json.loads(event_path.read_text())
            assert event["kind"] == "failure_record"
            assert event["payload"]["invocation_id"] == failed["id"]
            assert event["payload"]["workflow_key"] == "example-r2"
            assert event["payload"]["recovery_action"] == "new_attempt"
            assert (tmp_path / "attempts").read_text().splitlines() == [
                "attempt",
                "attempt",
            ]
            _verify_operator_recovery(
                client,
                server,
                worker_url,
                tmp_path,
                capsys,
                require_success,
                _poll_until,
            )
    finally:
        try:
            if worker is not None:
                worker.kill()
                worker.wait(timeout=10)
        finally:
            server.remove()


def _verify_operator_recovery(
    client, server, worker_url, evidence, capsys, require_success, poll_until
):
    request = {
        "issue_id": "gh-1-8",
        "repo": "Anchal-T/sqlparse",
        "title": "Terminal example",
        "base_commit": "a" * 40,
        "read_basis": "2026-11-24T12:00:00Z",
    }

    def submit(key, payload):
        return require_success(
            client.post(f"{server.ingress}/IssueRunV2/{key}/run/send", json=payload)
        )["invocationId"]

    def status(invocation_id):
        rows = require_success(
            client.post(
                f"{server.admin}/query",
                json={
                    "query": f"SELECT status FROM sys_invocation WHERE id = '{invocation_id}'"
                },
                headers={"Accept": "application/json"},
            )
        )["rows"]
        return rows[0]["status"] if rows else None

    def wait_for(invocation_id, expected):
        poll_until(
            expected, lambda: status(invocation_id), lambda value: value == expected
        )

    flags = [
        "--admin-url",
        server.admin,
        "--memory-url",
        worker_url,
        "--workflow-url",
        worker_url,
    ]
    failed = submit("gh-1-8", request)
    wait_for(failed, "completed")
    args = ["recover", failed, *flags, "--attempt", "2", "--apply"]
    assert run_cli(args) == 0
    linked = json.loads(capsys.readouterr().out)
    assert (
        linked["source_invocation_id"] == failed
        and linked["workflow_key"] == "gh-1-8-r2"
    )
    wait_for(linked["invocation_id"], "completed")
    rows = require_success(
        client.post(
            f"{server.admin}/query",
            json={
                "query": "SELECT completion_result FROM sys_invocation "
                f"WHERE id = '{linked['invocation_id']}'"
            },
            headers={"Accept": "application/json"},
        )
    )["rows"]
    assert rows == [{"completion_result": "success"}]
    assert run_cli(args) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["already_submitted"] is True
    assert repeated["invocation_id"] == linked["invocation_id"]

    paused = submit(
        "gh-1-9", {**request, "issue_id": "gh-1-9", "title": "Paused example"}
    )
    wait_for(paused, "suspended")
    client.patch(f"{server.admin}/invocations/{paused}/pause").raise_for_status()
    wait_for(paused, "paused")
    assert run_cli(["recover", paused, *flags, "--apply"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"action": "resume", "invocation_id": paused, "applied": True}
    wait_for(paused, "completed")
    assert (evidence / "checkpoint").read_text().splitlines() == ["checkpoint"]
