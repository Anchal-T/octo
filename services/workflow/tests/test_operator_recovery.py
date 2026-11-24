import json

import httpx
import pytest
from agentteam_common.cli import run_cli

ORIGIN = "inv_exampleOriginal"
NEW = "inv_exampleRetry"
OLD_KEY = "gh-1-8-r2"
NEW_KEY = "gh-1-8-r3"
SHA = "a" * 40
BASIS = "2026-11-24T12:00:00Z"
ARGS = [
    "recover",
    ORIGIN,
    "--admin-url",
    "http://admin",
    "--memory-url",
    "http://memory",
    "--workflow-url",
    "http://workflow",
]


class RecoveryFixture:
    def __init__(self, *, status="completed", pr_recorded=False):
        self.status = status
        self.pr_recorded = pr_recorded
        self.input = {
            "issue_id": "gh-1-8",
            "repo": "Anchal-T/sqlparse",
            "title": "Synthetic failure",
            "body": "Private fixture body",
            "attempt": 2,
            "fact_condition": "C2",
            "fact_snapshot": [{"claim": "Example frozen fact"}],
        }
        self.patches = []
        self.starts = []

    def transport(self, request):
        path = request.url.path
        match (request.method, request.url.host):
            case ("PATCH", "admin"):
                assert path == f"/invocations/{ORIGIN}/resume"
                self.patches.append(path)
                self.status = "running"
                return httpx.Response(200)
            case ("GET", "memory"):
                assert path.startswith("/v1/rounds/round-")
                return httpx.Response(
                    200,
                    json={
                        "id": path.split("/")[-1],
                        "seq": 1,
                        "status": "open",
                        "base_commit": SHA,
                        "read_basis": BASIS,
                        "committed_proposals": [],
                    },
                )
            case ("POST", "workflow"):
                assert path == "/v2/issues/gh-1-8/start"
                self.starts.append(json.loads(request.content))
                return httpx.Response(
                    202,
                    json={
                        "issue_id": "gh-1-8",
                        "attempt": 3,
                        "invocation_id": NEW,
                    },
                )
            case ("POST", "admin"):
                assert path == "/query"
                query = json.loads(request.content)["query"]
                if "sys_journal" in query:
                    if "open-pr:" in query:
                        rows = [{"name": "open-pr:1"}] if self.pr_recorded else []
                    else:
                        value = (
                            self.starts[0]
                            if NEW in query and self.starts
                            else self.input
                        )
                        entry = json.dumps(
                            {
                                "Command": {
                                    "Input": {
                                        "payload": list(
                                            json.dumps(value).encode("utf-8")
                                        ),
                                    }
                                }
                            }
                        )
                        rows = [{"entry_json": entry}]
                elif "sys_invocation" in query:
                    assert "target_service_name = 'IssueRunV2'" in query
                    assert "target_handler_name = 'run'" in query
                    if NEW_KEY in query:
                        rows = (
                            [
                                {
                                    "id": NEW,
                                    "target_service_key": NEW_KEY,
                                    "status": "running",
                                    "completion_result": None,
                                }
                            ]
                            if self.starts
                            else []
                        )
                    else:
                        assert ORIGIN in query
                        rows = [
                            {
                                "id": ORIGIN,
                                "target_service_key": OLD_KEY,
                                "status": self.status,
                                "completion_result": "failure"
                                if self.status == "completed"
                                else None,
                            }
                        ]
                else:
                    pytest.fail("unexpected recovery query")
                return httpx.Response(200, json={"rows": rows})
            case _:
                pytest.fail(f"unexpected recovery request: {request.method} {path}")


def test_operator_previews_and_resumes_only_paused_invocation(capsys):
    fixture = RecoveryFixture(status="paused")
    transport = httpx.MockTransport(fixture.transport)
    assert run_cli(ARGS, transport=transport) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "resume" and output["applied"] is False
    assert not fixture.patches

    assert run_cli([*ARGS, "--apply"], transport=transport) == 0
    assert len(fixture.patches) == 1
    assert run_cli([*ARGS, "--apply"], transport=transport) == 1
    assert len(fixture.patches) == 1


def test_operator_starts_one_linked_attempt_with_the_frozen_inputs(capsys):
    fixture = RecoveryFixture()
    transport = httpx.MockTransport(fixture.transport)
    retry = [*ARGS, "--attempt", "3"]
    assert run_cli(retry, transport=transport) == 0
    output = capsys.readouterr().out
    assert "Private fixture body" not in output
    assert json.loads(output)["applied"] is False
    assert not fixture.starts

    assert run_cli([*retry, "--apply"], transport=transport) == 0
    assert run_cli([*retry, "--apply"], transport=transport) == 0
    assert len(fixture.starts) == 1
    sent = fixture.starts[0]
    assert sent["attempt"] == 3
    assert sent["base_commit"] == SHA and sent["read_basis"] == BASIS
    assert sent["fact_condition"] == "C2"
    assert sent["fact_snapshot"] == fixture.input["fact_snapshot"]
    assert sent["recovery_from"] == {"invocation_id": ORIGIN, "attempt": 2}


def test_operator_blocks_retry_when_pr_dispatch_may_have_happened(capsys):
    fixture = RecoveryFixture(pr_recorded=True)
    transport = httpx.MockTransport(fixture.transport)
    assert run_cli([*ARGS, "--attempt", "3", "--apply"], transport=transport) == 1
    assert "PR dispatch" in capsys.readouterr().err
    assert not fixture.patches and not fixture.starts


def test_ordinary_workflow_request_keeps_the_existing_wire_payload():
    from agentteam_contracts import IssueRunRequestV2

    request = IssueRunRequestV2(
        issue_id="gh-1-8", repo="Anchal-T/sqlparse", title="Example issue"
    )
    assert "recovery_from" not in json.loads(request.model_dump_json())
