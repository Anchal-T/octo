import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentteam_common import stable_id
from agentteam_contracts import (
    ApprovalDecision,
    IssueRunRequestV2,
    IssueRunResultV2,
    ModelOutcome,
    RunStartAccepted,
    RunStatusV2,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services.github.app.main import build_runtime, close_runtime, create_app
from services.github.app.rules import evaluate_pr_reviews
from services.github.app.schemas import GithubPullRequest, GithubReview
from services.github.app.settings import GithubSettings
from services.github.tests.github_fakes import FakeGithub
from services.memory.app.sync import validate_proposal
from services.workflow.app import issue_run_v2 as v2_mod
from services.workflow.app.sandbox import LocalGitSandbox

NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)
SHA_1 = "1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa"
SHA_2 = "2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb"


def _pr(
    *,
    head_sha: str = SHA_1,
    state: str = "open",
    merged: bool = False,
    author: str = "octo-agent-team[bot]",
) -> GithubPullRequest:
    return GithubPullRequest(
        number=10,
        html_url="https://github.com/Anchal-T/sqlparse/pull/10",
        state="closed" if state == "closed" else "open",
        merged=merged,
        draft=True,
        head_ref="agent/gh-1450220110-7",
        head_sha=head_sha,
        base_ref="main",
        author_login=author,
    )


def _rev(
    rid: int,
    author: str,
    state: str,
    commit_id: str,
    *,
    body: str = "",
    minutes: int = 0,
) -> GithubReview:
    return GithubReview(
        id=rid,
        author_login=author,
        state=state,
        commit_id=commit_id,
        body=body,
        submitted_at=NOW + timedelta(minutes=minutes),
    )


def test_evaluate_pr_reviews_handles_dismissed_stale_self_multi_approver_and_closed():
    pr = _pr(head_sha=SHA_1)

    # 1. Valid approval on current head_sha
    ok = evaluate_pr_reviews(
        pr,
        [_rev(1, "Anchal-T", "APPROVED", SHA_1)],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert ok.decision == "approved"
    assert ok.by == "Anchal-T"

    # 2. Dismissed review does NOT count as approval!
    dismissed = evaluate_pr_reviews(
        pr,
        [
            _rev(1, "Anchal-T", "APPROVED", SHA_1, minutes=1),
            _rev(2, "Anchal-T", "DISMISSED", SHA_1, minutes=2),
        ],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert dismissed.decision == "pending"

    # 3. Stale review after new push (SHA_1 != SHA_2) does NOT count as approval!
    pushed_pr = _pr(head_sha=SHA_2)
    stale = evaluate_pr_reviews(
        pushed_pr,
        [_rev(1, "Anchal-T", "APPROVED", SHA_1, minutes=1)],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert stale.decision == "pending"

    # 4. COMMENTED after APPROVED does not wipe approval.
    commented = evaluate_pr_reviews(
        pr,
        [
            _rev(1, "Anchal-T", "APPROVED", SHA_1, minutes=1),
            _rev(2, "Anchal-T", "COMMENTED", SHA_1, body="nit", minutes=2),
        ],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert commented.decision == "approved"

    # 5. Multiple approvers: any CHANGES_REQUESTED on current SHA wins over APPROVED.
    conflict = evaluate_pr_reviews(
        pr,
        [
            _rev(1, "collab", "APPROVED", SHA_1, minutes=1),
            _rev(
                2,
                "Anchal-T",
                "CHANGES_REQUESTED",
                SHA_1,
                body="Fix edge case",
                minutes=2,
            ),
        ],
        approvers=("Anchal-T", "collab"),
        self_login="octo-agent-team[bot]",
    )
    assert conflict.decision == "rejected"
    assert conflict.reason == "Fix edge case"

    # 6. Bot self-approval or PR-author approval or outsider approval never counts.
    self_or_outsider = evaluate_pr_reviews(
        pr,
        [
            _rev(1, "octo-agent-team[bot]", "APPROVED", SHA_1, minutes=1),
            _rev(2, "random-user", "APPROVED", SHA_1, minutes=2),
        ],
        approvers=("Anchal-T", "octo-agent-team[bot]"),
        self_login="octo-agent-team[bot]",
    )
    assert self_or_outsider.decision == "pending"

    # 7. Closed without merge -> rejected; merged -> approved.
    closed_eval = evaluate_pr_reviews(
        _pr(state="closed", merged=False),
        [],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert closed_eval.decision == "rejected"
    assert closed_eval.reason == "pr_closed"

    merged_eval = evaluate_pr_reviews(
        _pr(state="closed", merged=True),
        [],
        approvers=("Anchal-T",),
        self_login="octo-agent-team[bot]",
    )
    assert merged_eval.decision == "approved"


def _rsa_keypair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("utf-8")
    pub = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return priv, pub


def _init_demo_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "sqlparse").mkdir()
    (path / "tests").mkdir()
    (path / "sqlparse" / "__init__.py").write_text("", encoding="utf-8")
    (path / "sqlparse" / "lexer.py").write_text(
        "def strip_quotes(val: str) -> str:\n"
        '    if val[0] == "\'" and val[-1] == "\'":\n'
        "        return val[1:-1]\n"
        "    return val\n",
        encoding="utf-8",
    )
    (path / "tests" / "test_lexer.py").write_text(
        "from sqlparse.lexer import strip_quotes\n\n\n"
        "def test_strip_quotes_quoted():\n"
        "    assert strip_quotes(\"'abc'\") == 'abc'\n",
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
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


@dataclass
class _InProcessWorkflowCtx:
    workflow_key: str
    state: dict[str, Any] = field(default_factory=dict)
    journal: dict[str, Any] = field(default_factory=dict)
    promises: dict[str, asyncio.Future[Any]] = field(default_factory=dict)

    def key(self) -> str:
        return self.workflow_key

    def set(self, name: str, value: Any) -> None:
        self.state[name] = value

    async def get(self, name: str, type_hint: Any = None) -> Any:
        return self.state.get(name)

    async def sleep(self, delta: Any, name: str = "") -> None:
        return None

    def promise(self, name: str, type_hint: Any = None) -> Any:
        loop = asyncio.get_running_loop()
        fut = self.promises.setdefault(name, loop.create_future())

        class _P:
            async def value(self) -> Any:
                raw = await fut
                if type_hint is not None and hasattr(type_hint, "model_validate"):
                    return type_hint.model_validate(raw)
                return raw

            async def resolve(self, val: Any) -> None:
                import restate

                if fut.done():
                    raise restate.TerminalError(
                        "promise already completed", status_code=409
                    )
                fut.set_result(
                    val.model_dump(mode="json") if hasattr(val, "model_dump") else val
                )

        return _P()

    async def run_typed(
        self, name: str, fn: Any, options: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
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


class _InMemoryXtdbBridge:
    def __init__(self) -> None:
        self.rounds: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, list[dict[str, Any]]] = {}

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            accepted: list[str] = []
            for prop in self.proposals.get(round_id, []):
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


@pytest.mark.anyio
async def test_thin_loop_end_to_end_approve_after_new_push_and_reject_path(
    monkeypatch, tmp_path
):
    repo_dir = tmp_path / "sqlparse_fork"
    base_sha = _init_demo_repo(repo_dir)
    priv_pem, pub_pem = _rsa_keypair()
    clock = [NOW]

    fake_gh = FakeGithub(
        repo="Anchal-T/sqlparse",
        self_login="octo-agent-team[bot]",
        repo_id=1450220110,
        app_public_key_pem=pub_pem,
        app_id="4242",
        app_slug="octo-agent-team",
        installation_id=9001,
        clock=lambda: clock[0],
        default_branch="main",
        default_head_sha=base_sha,
    )

    # In-process Restate workflow engine hosting IssueRunV2 invocations.
    invocations: dict[
        str, tuple[_InProcessWorkflowCtx, asyncio.Task[IssueRunResultV2]]
    ] = {}

    async def handle_workflow_http(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        # /v2/issues/{issue_id}/start | run | approve
        assert parts[0] == "v2" and parts[1] == "issues"
        issue_id = parts[2]
        action = parts[3]
        if request.method == "POST" and action == "start":
            body = json.loads(request.content)
            req = IssueRunRequestV2.model_validate(body)
            wf_key = issue_id if req.attempt == 1 else f"{issue_id}-r{req.attempt}"
            if wf_key not in invocations:
                ctx = _InProcessWorkflowCtx(wf_key)
                t = asyncio.create_task(v2_mod.run_issue_v2(ctx, req))  # type: ignore[arg-type]
                invocations[wf_key] = (ctx, t)
            accepted = RunStartAccepted(
                issue_id=issue_id, invocation_id=f"inv-{wf_key}", attempt=req.attempt
            )
            return httpx.Response(202, json=accepted.model_dump(mode="json"))
        if request.method == "GET" and action == "run":
            attempt = int(request.url.params.get("attempt", "1"))
            wf_key = issue_id if attempt == 1 else f"{issue_id}-r{attempt}"
            if wf_key not in invocations:
                return httpx.Response(404, json={"detail": "not found"})
            ctx, t = invocations[wf_key]
            if t.done():
                res = t.result()
                return httpx.Response(200, json=res.model_dump(mode="json"))
            st = await v2_mod.info_issue_v2(ctx)  # type: ignore[arg-type]
            assert isinstance(st, RunStatusV2)
            return httpx.Response(202, json=st.model_dump(mode="json"))
        if request.method == "POST" and action == "approve":
            body = json.loads(request.content)
            attempt = int(body.get("attempt", 1))
            wf_key = issue_id if attempt == 1 else f"{issue_id}-r{attempt}"
            ctx, _ = invocations[wf_key]
            dec = ApprovalDecision(
                approved=body["approved"],
                by=body["by"],
                reason=body.get("reason"),
                head_sha=body.get("head_sha"),
                review_id=body.get("review_id"),
                source=body.get("source", "github_review"),
            )
            out = await v2_mod.approve_issue_v2(ctx, dec)  # type: ignore[arg-type]
            return httpx.Response(200, json=out.model_dump(mode="json"))
        return httpx.Response(404, json={"detail": "unknown route"})

    class _AsyncRouterTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.github.com":
                return fake_gh.handle(request)
            return await handle_workflow_http(request)

    settings = GithubSettings(
        app_id="4242",
        app_private_key=priv_pem,
        repo="Anchal-T/sqlparse",
        workflow_version="v2",
        sqlite_path=str(tmp_path / "github.sqlite"),
    )
    runtime = build_runtime(
        settings,
        transport=_AsyncRouterTransport(),
        clock=lambda: clock[0],
        sleep=lambda _s: asyncio.sleep(0),
    )
    gh_app = create_app(runtime)

    memory = _InMemoryXtdbBridge()

    # Wire IssueRunV2's call_github_api right into the real FastAPI github service app!
    async def call_github_service(
        method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gh_app), base_url="http://github:8003"
        ) as client:
            resp = await client.request(method, path, json=payload)
            resp.raise_for_status()
            return resp.json()

    # Scripted free Groq model replies for two complete runs (Issue #7 and Issue #8).
    repro_bash = json.dumps(
        {
            "action": "bash",
            "command": (
                "cat <<'EOF' > tests/test_empty_quotes.py\n"
                "from sqlparse.lexer import strip_quotes\n\n"
                "def test_strip_quotes_empty():\n"
                "    assert strip_quotes('') == ''\n"
                "EOF"
            ),
        }
    )
    repro_submit = json.dumps(
        {
            "action": "submit",
            "test_node_id": "tests/test_empty_quotes.py::test_strip_quotes_empty",
            "summary": "strip_quotes('') raises IndexError",
        }
    )
    fix_bash = json.dumps(
        {
            "action": "bash",
            "command": (
                "cat <<'EOF' > sqlparse/lexer.py\n"
                "def strip_quotes(val: str) -> str:\n"
                '    if len(val) >= 2 and val[0] == "\'" and val[-1] == "\'":\n'
                "        return val[1:-1]\n"
                "    return val\n"
                "EOF"
            ),
        }
    )
    fix_submit = json.dumps(
        {"action": "submit", "summary": "Guard len(val) >= 2 in strip_quotes"}
    )
    triage_reply = json.dumps(
        {
            "kind": "bug",
            "priority": "high",
            "summary": "strip_quotes raises IndexError on empty string.",
            "area": "lexer-keywords",
        }
    )
    tester_reply = json.dumps(
        {
            "verdict": "approve",
            "summary": "Clean fix verified on clean checkout.",
            "notes": "",
        }
    )
    role_queues = {
        "triager": [triage_reply, triage_reply],
        "reproducer": [repro_bash, repro_submit, repro_bash, repro_submit],
        "fixer": [fix_bash, fix_submit, fix_bash, fix_submit],
        "tester": [tester_reply, tester_reply],
    }

    def fake_model(request):
        text = role_queues[request.role].pop(0)
        return ModelOutcome(
            outcome="completed",
            request_id=f"mc-{request.run_id}-{request.step}",
            decided_at=clock[0],
            model_pinned="openai/gpt-oss-20b",
            model_reported="openai/gpt-oss-20b",
            run_id=request.run_id,
            step=request.step,
            text=text,
            tokens_in=40,
            tokens_out=20,
            started_at=clock[0],
            finished_at=clock[0],
        )

    monkeypatch.setattr(v2_mod, "call_model", fake_model)
    monkeypatch.setattr(v2_mod, "post_memory", memory.post)
    monkeypatch.setattr(v2_mod, "call_github_api", call_github_service)
    monkeypatch.setattr(
        v2_mod,
        "make_sandbox",
        lambda sha: LocalGitSandbox(
            repo_path=repo_dir, base_commit=sha, work_root=tmp_path / "sandboxes"
        ),
    )

    try:
        await runtime.poller.prepare()

        # --- Scenario A: Issue #7 -> PR -> dismissed review -> new push -> re-request -> approve ---
        clock[0] = NOW + timedelta(minutes=1)
        fake_gh.add_issue(
            7,
            title="strip_quotes('') crashes with IndexError",
            body="Passing an empty string to strip_quotes raises IndexError.",
            labels=("agent-team",),
            author="Anchal-T",
            created_at=clock[0],
            updated_at=clock[0],
        )
        c1 = await runtime.poller.run_cycle()
        assert c1.started == 1

        wf_ctx_7, wf_task_7 = invocations["gh-1450220110-7"]
        for _ in range(150):
            if wf_task_7.done():
                wf_task_7.result()
            if wf_ctx_7.state.get("stage") == "awaiting_approval":
                break
            await asyncio.sleep(0.05)
        assert wf_ctx_7.state.get("stage") == "awaiting_approval"

        pr_7_num = wf_ctx_7.state["pr_number"]
        sha_initial = wf_ctx_7.state["pr_head_sha"]
        # Decision 1: GitHub App bot authored the PR, NOT Anchal-T!
        assert fake_gh.pulls[pr_7_num]["user"]["login"] == "octo-agent-team[bot]"
        assert fake_gh.review_requests[pr_7_num] == [["Anchal-T"]]
        assert fake_gh.statuses[sha_initial][-1]["state"] == "pending"

        # Dismissed review does NOT approve!
        clock[0] = NOW + timedelta(minutes=2)
        fake_gh.add_review(
            pr_7_num,
            author="Anchal-T",
            state="APPROVED",
            commit_id=sha_initial,
            submitted_at=clock[0],
        )
        clock[0] = NOW + timedelta(minutes=3)
        fake_gh.add_review(
            pr_7_num,
            author="Anchal-T",
            state="DISMISSED",
            commit_id=sha_initial,
            submitted_at=clock[0],
        )
        await runtime.poller.run_cycle()
        assert not wf_task_7.done()

        # Approve at sha_initial, THEN push a new commit sha_pushed before poll -> stale review!
        clock[0] = NOW + timedelta(minutes=4)
        fake_gh.add_review(
            pr_7_num,
            author="Anchal-T",
            state="APPROVED",
            commit_id=sha_initial,
            submitted_at=clock[0],
        )
        sha_pushed = "9999cccc9999cccc9999cccc9999cccc9999cccc"
        fake_gh.push_pr_commit(pr_7_num, sha_pushed)

        clock[0] = NOW + timedelta(minutes=5)
        await runtime.poller.run_cycle()
        # Stale review did NOT approve, AND review was re-requested on sha_pushed!
        assert not wf_task_7.done()
        assert fake_gh.review_requests[pr_7_num] == [["Anchal-T"], ["Anchal-T"]]
        assert fake_gh.statuses[sha_pushed][-1]["state"] == "pending"

        # Now Anchal-T approves at sha_pushed -> workflow resolves to merged + success status!
        clock[0] = NOW + timedelta(minutes=6)
        fake_gh.add_review(
            pr_7_num,
            author="Anchal-T",
            state="APPROVED",
            commit_id=sha_pushed,
            submitted_at=clock[0],
        )
        await runtime.poller.run_cycle()
        final_7 = await wf_task_7
        assert final_7.status == "approved"
        assert final_7.stage == "merged"
        assert final_7.pr_head_sha == sha_pushed
        assert fake_gh.statuses[sha_pushed][-1]["state"] == "success"

        clock[0] = NOW + timedelta(minutes=7)
        await runtime.poller.run_cycle()
        row_7 = runtime.store.get_issue("gh-1450220110-7")
        assert row_7 is not None
        assert row_7.state == "finished"
        assert row_7.result_status == "approved"

        # --- Scenario B: Issue #8 -> PR -> CHANGES_REQUESTED -> rejected / closed ---
        clock[0] = NOW + timedelta(minutes=8)
        fake_gh.add_issue(
            8,
            title="Another strip_quotes empty input case",
            body="Reproduce and fix empty input.",
            labels=("agent-team",),
            author="Anchal-T",
            created_at=clock[0],
            updated_at=clock[0],
        )
        await runtime.poller.run_cycle()
        wf_ctx_8, wf_task_8 = invocations["gh-1450220110-8"]
        for _ in range(150):
            if wf_task_8.done():
                wf_task_8.result()
            if wf_ctx_8.state.get("stage") == "awaiting_approval":
                break
            await asyncio.sleep(0.05)

        pr_8_num = wf_ctx_8.state["pr_number"]
        sha_8 = wf_ctx_8.state["pr_head_sha"]
        clock[0] = NOW + timedelta(minutes=9)
        fake_gh.add_review(
            pr_8_num,
            author="Anchal-T",
            state="CHANGES_REQUESTED",
            commit_id=sha_8,
            body="Also strip double quotes cleanly.",
            submitted_at=clock[0],
        )
        await runtime.poller.run_cycle()
        final_8 = await wf_task_8
        assert final_8.status == "rejected"
        assert final_8.stage == "closed"
        assert final_8.approval is not None
        assert final_8.approval.reason == "Also strip double quotes cleanly."
        assert fake_gh.statuses[sha_8][-1]["state"] == "failure"

        clock[0] = NOW + timedelta(minutes=10)
        await runtime.poller.run_cycle()
        row_8 = runtime.store.get_issue("gh-1450220110-8")
        assert row_8 is not None
        assert row_8.state == "finished"
        assert row_8.result_status == "rejected"
    finally:
        await close_runtime(runtime)
