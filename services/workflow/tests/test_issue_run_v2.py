import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentteam_common import stable_id
from agentteam_common.cli import run_cli
from agentteam_contracts import (
    ApprovalDecision,
    ApprovalResult,
    IssueRunRequestV2,
    IssueRunResultV2,
    ModelOutcome,
)

from services.memory.app.sync import validate_proposal
from services.workflow.app import issue_run_v2 as v2_mod
from services.workflow.app import main as workflow_main
from services.workflow.app.sandbox import LocalGitSandbox

NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)
HEAD_SHA_1 = "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111"
HEAD_SHA_2 = "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222"


def _init_sqlparse_like_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "sqlparse").mkdir()
    (path / "tests").mkdir()
    (path / "sqlparse" / "__init__.py").write_text("", encoding="utf-8")
    (path / "sqlparse" / "lexer.py").write_text(
        "def tokenize_word(word: str) -> str:\n"
        "    if not word:\n"
        "        raise ValueError('empty')\n"
        "    return word.upper()\n",
        encoding="utf-8",
    )
    (path / "tests" / "test_lexer.py").write_text(
        "from sqlparse.lexer import tokenize_word\n\n\n"
        "def test_tokenize_word_upper():\n"
        "    assert tokenize_word('select') == 'SELECT'\n",
        encoding="utf-8",
    )
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, env=env, check=True)
    subprocess.run(["git", "add", "."], cwd=path, env=env, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, env=env, check=True)
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


@dataclass
class FakeWorkflowCtx:
    workflow_key: str
    state: dict[str, Any] = field(default_factory=dict)
    journal: dict[str, Any] = field(default_factory=dict)
    promises: dict[str, asyncio.Future[Any]] = field(default_factory=dict)
    sleeps: list[tuple[str, float]] = field(default_factory=list)

    def key(self) -> str:
        return self.workflow_key

    def set(self, name: str, value: Any) -> None:
        self.state[name] = value

    async def get(self, name: str, type_hint: Any = None) -> Any:
        return self.state.get(name)

    async def sleep(self, delta: Any, name: str = "") -> None:
        self.sleeps.append((name, delta.total_seconds()))

    def promise(self, name: str, type_hint: Any = None) -> Any:
        loop = asyncio.get_running_loop()
        fut = self.promises.setdefault(name, loop.create_future())

        class _Promise:
            async def value(self_inner) -> Any:
                raw = await fut
                if type_hint is not None and hasattr(type_hint, "model_validate"):
                    return type_hint.model_validate(raw)
                return raw

            async def resolve(self_inner, val: Any) -> None:
                if fut.done():
                    import restate

                    raise restate.TerminalError(
                        "promise already completed", status_code=409
                    )
                payload = (
                    val.model_dump(mode="json") if hasattr(val, "model_dump") else val
                )
                fut.set_result(payload)

        return _Promise()

    async def run_typed(
        self,
        name: str,
        fn: Any,
        options: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Restate run_typed accepts either RunOptions as 3rd positional arg or omitted.
        import restate

        if not isinstance(options, restate.RunOptions) and options is not None:
            args = (options, *args)
            options = None
        if name in self.journal:
            stored = self.journal[name]
            hint = getattr(options, "type_hint", None)
            if hint is not None and hasattr(hint, "model_validate"):
                return hint.model_validate(stored)
            return stored
        res = fn(*args, **kwargs)
        if hasattr(res, "__await__"):
            res = await res
        self.journal[name] = (
            res.model_dump(mode="json") if hasattr(res, "model_dump") else res
        )
        return res


class FakeMemoryAndGithubBackend:
    def __init__(self, *, base_sha: str, pr_head_sha: str = HEAD_SHA_1) -> None:
        self.base_sha = base_sha
        self.pr_head_sha = pr_head_sha
        self.rounds: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, list[dict[str, Any]]] = {}
        self.prs_opened: list[dict[str, Any]] = []
        self.evidence_upserts: list[dict[str, Any]] = []
        self.statuses_set: list[tuple[str, dict[str, Any]]] = []

    async def post_memory(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/v1/rounds":
            rid = stable_id("round", payload["idempotency_key"])
            row = {
                "id": rid,
                "seq": payload["seq"],
                "status": "open",
                "base_commit": payload.get("base_commit"),
                "read_basis": payload["read_basis"],
                "committed_proposals": [],
            }
            self.rounds[rid] = row
            return row
        if path == "/v1/events":
            eid = stable_id("ev", payload["idempotency_key"])
            row = {
                "id": eid,
                "source": payload["source"],
                "kind": payload["kind"],
                "taint": payload["taint"],
                "taint_label": "ok",
                "payload": payload["payload"],
            }
            self.events[eid] = row
            return row
        if path == "/v1/runs":
            run_id = stable_id("run", payload["idempotency_key"])
            self.runs[run_id] = {**payload, "_id": run_id}
            return {"id": run_id}
        if path.endswith("/proposals"):
            round_id = path.split("/")[3]
            pid = stable_id("prop", payload["idempotency_key"])
            prop = {
                "_id": pid,
                "id": pid,
                "round_id": round_id,
                "agent_id": payload["agent_id"],
                "run_id": payload["run_id"],
                "op": payload["op"],
                "status": "pending",
                "reason": None,
                "evidence_event_ids": payload["evidence_event_ids"],
                "payload": payload["payload"],
            }
            self.proposals.setdefault(round_id, []).append(prop)
            return prop
        if path.endswith("/commit"):
            round_id = path.split("/")[3]
            props = self.proposals.get(round_id, [])
            accepted: list[str] = []
            for prop in props:
                run = self.runs.get(prop["run_id"])
                ev_map = {
                    eid: self.events.get(eid) for eid in prop["evidence_event_ids"]
                }
                verdict, reason, _ = validate_proposal(round_id, prop, run, ev_map)
                prop["status"] = verdict
                prop["reason"] = reason
                if verdict == "accepted":
                    accepted.append(prop["id"])
            row = {
                **self.rounds[round_id],
                "status": "committed",
                "committed_proposals": accepted,
            }
            self.rounds[round_id] = row
            return row
        raise AssertionError(f"unexpected memory path: {path}")

    async def call_github(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if method == "GET" and path == "/v1/repo/head":
            return {
                "repo": "Anchal-T/sqlparse",
                "default_branch": "main",
                "head_sha": self.base_sha,
            }
        if method == "GET" and path.startswith("/v1/prs/"):
            return {
                "pr_number": 15,
                "issue_key": "gh-1-7",
                "attempt": 1,
                "head_sha": self.pr_head_sha,
                "state": "open",
                "decision": "pending",
            }
        if method == "POST" and path == "/v1/prs":
            assert payload is not None
            self.prs_opened.append(payload)
            return {
                "outcome": "ok",
                "decided_at": NOW.isoformat(),
                "pr_number": 15,
                "html_url": "https://github.com/Anchal-T/sqlparse/pull/15",
                "head_sha": self.pr_head_sha,
                "branch": f"agent/{payload['issue_id']}",
            }
        if method == "PUT" and path.endswith("/evidence"):
            assert payload is not None
            self.evidence_upserts.append(payload)
            return {
                "outcome": "ok",
                "decided_at": NOW.isoformat(),
                "pr_number": 15,
                "head_sha": payload["head_sha"],
                "comment_id": 88,
            }
        if method == "POST" and path.startswith("/v1/commits/"):
            sha = path.split("/")[3]
            assert payload is not None
            self.statuses_set.append((sha, payload))
            return {
                "outcome": "ok",
                "decided_at": NOW.isoformat(),
                "head_sha": sha,
            }
        raise AssertionError(f"unexpected github call: {method} {path}")


def _scripted_model(replies_by_role: dict[str, list[str]]):
    counters: dict[str, int] = {}

    def fake_call(request):
        idx = counters.get(request.role, 0)
        counters[request.role] = idx + 1
        text = replies_by_role[request.role][idx]
        return ModelOutcome(
            outcome="completed",
            request_id=f"mc-{request.role}-{idx}",
            decided_at=NOW,
            model_pinned="openai/gpt-oss-20b",
            model_reported="openai/gpt-oss-20b",
            run_id=request.run_id,
            step=request.step,
            text=text,
            tokens_in=30,
            tokens_out=15,
            started_at=NOW,
            finished_at=NOW,
        )

    return fake_call, counters


@pytest.mark.anyio
async def test_issue_run_v2_happy_path_stale_head_guard_and_duplicate_approve(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    base_sha = _init_sqlparse_like_repo(repo)
    backend = FakeMemoryAndGithubBackend(base_sha=base_sha, pr_head_sha=HEAD_SHA_1)

    fake_call, _ = _scripted_model(
        {
            "triager": [
                json.dumps(
                    {
                        "kind": "bug",
                        "priority": "high",
                        "summary": "tokenize_word raises on empty string instead of returning empty.",
                        "area": "lexer-keywords",
                    }
                )
            ],
            "reproducer": [
                json.dumps(
                    {
                        "action": "bash",
                        "command": (
                            "cat <<'EOF' > tests/test_empty_word.py\n"
                            "from sqlparse.lexer import tokenize_word\n\n"
                            "def test_empty_returns_empty():\n"
                            "    assert tokenize_word('') == ''\n"
                            "EOF"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "action": "submit",
                        "test_node_id": "tests/test_empty_word.py::test_empty_returns_empty",
                        "summary": "Reproduced ValueError on empty word",
                    }
                ),
            ],
            "fixer": [
                json.dumps(
                    {
                        "action": "bash",
                        "command": (
                            "cat <<'EOF' > sqlparse/lexer.py\n"
                            "def tokenize_word(word: str) -> str:\n"
                            "    if not word:\n"
                            "        return ''\n"
                            "    return word.upper()\n"
                            "EOF"
                        ),
                    }
                ),
                json.dumps({"action": "submit", "summary": "Return '' on empty word"}),
            ],
            "tester": [
                json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Clean fix and regression test.",
                        "notes": "",
                    }
                )
            ],
        }
    )

    monkeypatch.setattr(v2_mod, "call_model", fake_call)
    monkeypatch.setattr(v2_mod, "post_memory", backend.post_memory)
    monkeypatch.setattr(v2_mod, "call_github_api", backend.call_github)
    monkeypatch.setattr(
        v2_mod,
        "make_sandbox",
        lambda sha: LocalGitSandbox(
            repo_path=repo, base_commit=sha, work_root=tmp_path / "sb"
        ),
    )

    ctx = FakeWorkflowCtx("gh-1-7")
    task = asyncio.create_task(
        v2_mod.run_issue_v2(
            ctx,  # type: ignore[arg-type]
            IssueRunRequestV2(
                issue_id="gh-1-7",
                repo="Anchal-T/sqlparse",
                title="Empty word crashes tokenize_word",
                body="Calling tokenize_word('') raises ValueError.",
            ),
        )
    )

    # Wait until workflow reaches awaiting_approval.
    for _ in range(100):
        if task.done():
            task.result()
        if ctx.state.get("stage") == "awaiting_approval":
            break
        await asyncio.sleep(0.05)
    assert ctx.state.get("stage") == "awaiting_approval"
    assert len(backend.prs_opened) == 1
    assert len(backend.evidence_upserts) == 1
    assert backend.statuses_set[-1][1]["state"] == "pending"

    # Stale head SHA is rejected without resolving pr_approval!
    stale = await v2_mod.approve_issue_v2(
        ctx,  # type: ignore[arg-type]
        ApprovalDecision(
            approved=True,
            by="Anchal-T",
            head_sha=HEAD_SHA_2,
        ),
    )
    assert stale.status == "stale_head"
    assert not task.done()

    # Valid approval at HEAD_SHA_1 resolves the workflow to merged + sets success status!
    recorded = await v2_mod.approve_issue_v2(
        ctx,  # type: ignore[arg-type]
        ApprovalDecision(
            approved=True,
            by="Anchal-T",
            head_sha=HEAD_SHA_1,
        ),
    )
    assert recorded.status == "recorded"
    result = await task
    assert isinstance(result, IssueRunResultV2)
    assert result.status == "approved"
    assert result.stage == "merged"
    assert result.pr_number == 15
    assert result.pr_head_sha == HEAD_SHA_1
    assert backend.statuses_set[-1] == (
        HEAD_SHA_1,
        {
            "idempotency_key": "gh-1-7:v2:status:final",
            "state": "success",
            "description": "Approved by Anchal-T; ready to merge.",
            "context": "agent-team",
        },
    )

    # Duplicate approval after completion returns already_decided idempotently.
    dup = await v2_mod.approve_issue_v2(
        ctx,  # type: ignore[arg-type]
        ApprovalDecision(approved=True, by="Anchal-T", head_sha=HEAD_SHA_1),
    )
    assert dup.status == "already_decided"


@pytest.mark.anyio
async def test_issue_run_v2_retries_fixer_on_changes_requested_and_handles_rejection(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    base_sha = _init_sqlparse_like_repo(repo)
    backend = FakeMemoryAndGithubBackend(base_sha=base_sha, pr_head_sha=HEAD_SHA_1)

    fake_call, counters = _scripted_model(
        {
            "triager": [
                json.dumps(
                    {
                        "kind": "bug",
                        "priority": "medium",
                        "summary": "Empty word raises.",
                        "area": "lexer-keywords",
                    }
                )
            ],
            "reproducer": [
                json.dumps(
                    {
                        "action": "bash",
                        "command": (
                            "cat <<'EOF' > tests/test_empty_word.py\n"
                            "from sqlparse.lexer import tokenize_word\n\n"
                            "def test_empty_returns_empty():\n"
                            "    assert tokenize_word('') == ''\n"
                            "EOF"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "action": "submit",
                        "test_node_id": "tests/test_empty_word.py::test_empty_returns_empty",
                        "summary": "Repro added",
                    }
                ),
            ],
            "fixer": [
                # Fix attempt 1: breaks existing test_tokenize_word_upper!
                json.dumps(
                    {
                        "action": "bash",
                        "command": (
                            "cat <<'EOF' > sqlparse/lexer.py\n"
                            "def tokenize_word(word: str) -> str:\n"
                            "    return ''\n"
                            "EOF"
                        ),
                    }
                ),
                json.dumps({"action": "submit", "summary": "Always return empty"}),
                # Fix attempt 2: fixes properly!
                json.dumps(
                    {
                        "action": "bash",
                        "command": (
                            "cat <<'EOF' > sqlparse/lexer.py\n"
                            "def tokenize_word(word: str) -> str:\n"
                            "    if not word:\n"
                            "        return ''\n"
                            "    return word.upper()\n"
                            "EOF"
                        ),
                    }
                ),
                json.dumps({"action": "submit", "summary": "Handle empty and upper"}),
            ],
            "tester": [
                # Tester attempt 1: even if it said approve, T0 suite failure forces changes_requested!
                json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Attempt 1",
                        "notes": "",
                    }
                ),
                # Tester attempt 2: approves clean fix!
                json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Attempt 2 clean.",
                        "notes": "",
                    }
                ),
            ],
        }
    )

    monkeypatch.setattr(v2_mod, "call_model", fake_call)
    monkeypatch.setattr(v2_mod, "post_memory", backend.post_memory)
    monkeypatch.setattr(v2_mod, "call_github_api", backend.call_github)
    monkeypatch.setattr(
        v2_mod,
        "make_sandbox",
        lambda sha: LocalGitSandbox(
            repo_path=repo, base_commit=sha, work_root=tmp_path / "sb2"
        ),
    )

    ctx = FakeWorkflowCtx("gh-1-8")
    task = asyncio.create_task(
        v2_mod.run_issue_v2(
            ctx,  # type: ignore[arg-type]
            IssueRunRequestV2(
                issue_id="gh-1-8",
                repo="Anchal-T/sqlparse",
                title="Empty word crashes tokenize_word",
            ),
        )
    )
    for _ in range(120):
        if task.done():
            task.result()
        if ctx.state.get("stage") == "awaiting_approval":
            break
        await asyncio.sleep(0.05)
    assert ctx.state.get("stage") == "awaiting_approval"
    assert counters["tester"] == 2

    # Maintainer rejects PR -> stage closed, reason stored, failure commit status set!
    await v2_mod.approve_issue_v2(
        ctx,  # type: ignore[arg-type]
        ApprovalDecision(
            approved=False,
            by="Anchal-T",
            reason="Please handle whitespace-only strings too.",
            head_sha=HEAD_SHA_1,
        ),
    )
    result = await task
    assert result.status == "rejected"
    assert result.stage == "closed"
    assert result.fix_attempts == 2
    assert result.approval is not None
    assert result.approval.reason == "Please handle whitespace-only strings too."
    assert backend.statuses_set[-1][1]["state"] == "failure"
    decision_events = [ev for ev in backend.events.values() if ev["kind"] == "decision"]
    assert len(decision_events) == 1
    assert decision_events[0]["taint"] == 0
    assert (
        decision_events[0]["payload"]["reason"]
        == "Please handle whitespace-only strings too."
    )


def test_cli_approve_and_reject_fallback_calls_v2_approve_endpoint(capsys):
    received: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/issues/gh-1450220110-7/approve"
        body = json.loads(request.content)
        received.append(body)
        res = ApprovalResult(
            issue_id=body["issue_id"],
            attempt=body["attempt"],
            status="recorded",
            stage="merged" if body["approved"] else "closed",
        )
        return httpx.Response(200, json=res.model_dump(mode="json"))

    transport = httpx.MockTransport(handler)
    code1 = run_cli(
        [
            "approve",
            "gh-1450220110-7",
            "--by",
            "Anchal-T",
            "--workflow-url",
            "http://workflow:8002",
        ],
        transport=transport,
    )
    assert code1 == 0
    assert received[-1]["approved"] is True
    assert received[-1]["source"] == "cli"

    code2 = run_cli(
        [
            "approve",
            "gh-1450220110-7",
            "--attempt",
            "2",
            "--reject",
            "--by",
            "Anchal-T",
            "--reason",
            "Missing whitespace test",
            "--workflow-url",
            "http://workflow:8002",
        ],
        transport=transport,
    )
    assert code2 == 0
    assert received[-1]["approved"] is False
    assert received[-1]["attempt"] == 2
    assert received[-1]["reason"] == "Missing whitespace test"
    assert "closed" in capsys.readouterr().out
    assert hasattr(workflow_main, "issue_run_v2")
