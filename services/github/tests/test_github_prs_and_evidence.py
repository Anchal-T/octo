from datetime import UTC, datetime, timedelta

import httpx
import pytest
from agentteam_contracts import (
    CommitStatusRequest,
    EvidenceCommentRequest,
    EvidenceFactItem,
    ModelUsageItem,
    OpenPrRequest,
    PrFileChange,
    SyncDecisionItem,
    TriageResult,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services.github.app.main import build_runtime, close_runtime, create_app
from services.github.app.settings import GithubSettings
from services.github.app.store import Store
from services.github.tests.github_fakes import FakeGithub, FakeWorkflow

NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)
BASE_SHA = "1111111111111111111111111111111111111111"


def _rsa_pair() -> tuple[str, str]:
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
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return priv, pub


def _make_app_env(tmp_path, *, use_app: bool = True):
    priv, pub = _rsa_pair()
    fake_gh = FakeGithub(
        repo="Anchal-T/sqlparse" if use_app else "agent-bot/fork",
        self_login="octo-agent-team[bot]" if use_app else "agent-bot",
        repo_id=1450220110,
        app_public_key_pem=pub if use_app else None,
        app_id="4242",
        app_slug="octo-agent-team",
        installation_id=9001,
        clock=lambda: NOW,
        default_branch="main",
        default_head_sha=BASE_SHA,
    )
    fake_wf = FakeWorkflow()

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return fake_gh.handle(request)
        return fake_wf.handle(request)

    if use_app:
        settings = GithubSettings(
            app_id="4242",
            app_private_key=priv,
            repo="Anchal-T/sqlparse",
            sqlite_path=str(tmp_path / "github.sqlite"),
        )
    else:
        settings = GithubSettings(
            token="test-token",
            repo="agent-bot/fork",
            sqlite_path=str(tmp_path / "github-pat.sqlite"),
        )
    runtime = build_runtime(
        settings,
        transport=httpx.MockTransport(dispatch),
        clock=lambda: NOW,
    )
    return runtime, create_app(runtime), fake_gh


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "write_path", "close_pr"),
    [
        pytest.param("github.tree_create", "/git/trees", False, id="tree"),
        pytest.param("github.commit_create", "/git/commits", False, id="commit"),
        pytest.param("github.branch_ref_upsert", "/git/refs", False, id="branch"),
        pytest.param("github.draft_pr_open", "/pulls", False, id="draft-pr"),
        pytest.param("github.draft_pr_open", "/pulls", True, id="closed-pr"),
        pytest.param(
            "github.reviewer_request",
            "/pulls/1/requested_reviewers",
            False,
            id="reviewers",
        ),
        pytest.param(
            "github.evidence_comment_upsert", "/issues/1/comments", False, id="comment"
        ),
        pytest.param("github.commit_status", "/statuses/", False, id="status"),
    ],
)
async def test_replay_recovers_write_before_success_receipt(
    tmp_path, monkeypatch, action, write_path, close_pr
):
    runtime, app, github = _make_app_env(tmp_path)
    base_tree = github._put_tree(
        github.empty_tree_sha,
        [
            {"path": "src/old.py", "mode": "100644", "content": "old\n"},
            {"path": "src/keep.py", "mode": "100644", "content": "keep\n"},
            {"path": "src.py", "mode": "100644", "content": "sibling\n"},
        ],
    )
    github.git_commits[BASE_SHA] = {"tree": base_tree}
    record = Store.record_side_effect_receipt

    class Restart(Exception):
        pass

    def crash_before_receipt(store, key, **receipt):
        if receipt["action"] == action and receipt["outcome"] == "ok":
            raise Restart
        record(store, key, **receipt)

    async def request(app, method, path, body):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, json=body.model_dump(mode="json"))

    open_request = OpenPrRequest(
        idempotency_key="open-1",
        issue_id="gh-1450220110-7",
        base_commit=BASE_SHA,
        title="Fix parser",
        files=[
            PrFileChange(path="src/old.py", content=None),
            PrFileChange(path="src/new.py", content="new = 'é'\n"),
        ],
    )
    method, path, body = "POST", "/v1/prs", open_request
    settings = runtime.settings
    try:
        if action in {"github.evidence_comment_upsert", "github.commit_status"}:
            opened = await request(app, method, path, body)
            assert opened.json()["outcome"] == "ok"
            sha = opened.json()["head_sha"]
            if action == "github.evidence_comment_upsert":
                method, path = "PUT", "/v1/prs/1/evidence"
                body = EvidenceCommentRequest(
                    idempotency_key="evidence-1",
                    issue_id=open_request.issue_id,
                    base_commit=BASE_SHA,
                    head_sha=sha,
                    repro_test_node_id="test.py::test_it",
                    repro_failed_before=True,
                    repro_passed_after=True,
                    suite_passed=True,
                    review_verdict="approve",
                )
            else:
                path = f"/v1/commits/{sha}/status"
                body = CommitStatusRequest(
                    idempotency_key="status-1", state="pending", description="Waiting"
                )
                write_path += sha

        with monkeypatch.context() as patch:
            patch.setattr(Store, "record_side_effect_receipt", crash_before_receipt)
            with pytest.raises(Restart):
                await request(app, method, path, body)
        with runtime.store.connection() as conn:
            outcomes = conn.execute(
                "SELECT outcome FROM side_effect_receipt WHERE action = ?", (action,)
            ).fetchall()
        assert [row["outcome"] for row in outcomes] == ["pending"]
    finally:
        await close_runtime(runtime)

    if close_pr:
        github.pulls[1]["state"] = "closed"
    restart_time = NOW + timedelta(minutes=1)
    runtime = build_runtime(
        settings, transport=github.transport, clock=lambda: restart_time
    )
    github._clock = lambda: restart_time
    try:
        result = await request(create_app(runtime), method, path, body)
        assert result.status_code == 200
        assert result.json()["outcome"] == "ok"
        if close_pr:
            assert runtime.store.get_pull_request(1).state == "closed"
            assert github.review_requests == {}
        with runtime.store.connection() as conn:
            receipts = conn.execute(
                "SELECT * FROM side_effect_receipt WHERE action = ? ORDER BY seq",
                (action,),
            ).fetchall()
        assert [row["outcome"] for row in receipts] == ["pending", "ok"]
        assert receipts[0]["idempotency_key"] == receipts[1]["idempotency_key"]
        assert receipts[0]["request_sha256"] == receipts[1]["request_sha256"]
        assert (
            github.calls.count(("POST", f"/repos/{github.repo}{write_path}", {})) == 1
        )
        assert not any(method == "PATCH" for method, _, _ in github.calls)
    finally:
        await close_runtime(runtime)


@pytest.mark.anyio
async def test_pr_creation_requires_github_app_auth_never_pat(tmp_path):
    runtime, app, _ = _make_app_env(tmp_path, use_app=False)
    try:
        await runtime.poller.prepare()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/prs",
                json=OpenPrRequest(
                    idempotency_key="k1",
                    issue_id="gh-100-7",
                    base_commit=BASE_SHA,
                    title="Fix bug",
                    files=[PrFileChange(path="sqlparse/lexer.py", content="x = 1\n")],
                ).model_dump(mode="json"),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome"] == "failed"
        assert body["error_code"] == "app_auth_required"
    finally:
        await close_runtime(runtime)


@pytest.mark.anyio
async def test_open_pr_is_idempotent_authors_as_app_bot_and_rerequests_review_on_new_push(
    tmp_path,
):
    runtime, app, fake_gh = _make_app_env(tmp_path, use_app=True)
    try:
        await runtime.poller.prepare()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            head_resp = await client.get("/v1/repo/head")
            assert head_resp.status_code == 200
            assert head_resp.json() == {
                "repo": "Anchal-T/sqlparse",
                "default_branch": "main",
                "head_sha": BASE_SHA,
            }

            req1 = OpenPrRequest(
                idempotency_key="gh-1450220110-7:v2:open-pr:1",
                issue_id="gh-1450220110-7",
                attempt=1,
                base_commit=BASE_SHA,
                title="Fix zero handling in lexer",
                body="Automated draft PR for issue #7.",
                files=[
                    PrFileChange(path="sqlparse/lexer.py", content="FIXED = True\n"),
                    PrFileChange(
                        path="tests/test_lexer_repro.py",
                        content="def test_repro(): assert True\n",
                    ),
                ],
            )
            first = await client.post("/v1/prs", json=req1.model_dump(mode="json"))
            repeat = await client.post("/v1/prs", json=req1.model_dump(mode="json"))
            assert first.status_code == repeat.status_code == 200
            assert first.json() == repeat.json()

            pr_out = first.json()
            assert pr_out["outcome"] == "ok"
            pr_num = pr_out["pr_number"]
            sha1 = pr_out["head_sha"]
            assert pr_out["branch"] == "agent/gh-1450220110-7"
            assert fake_gh.pulls[pr_num]["user"]["login"] == "octo-agent-team[bot]"
            assert fake_gh.pulls[pr_num]["draft"] is True
            assert fake_gh.review_requests[pr_num] == [["Anchal-T"]]

            # Reusing the same idempotency_key with a different body returns 409.
            conflict = await client.post(
                "/v1/prs",
                json=req1.model_copy(update={"title": "Different title"}).model_dump(
                    mode="json"
                ),
            )
            assert conflict.status_code == 409

            # Attempt 2 pushes a new commit onto the open PR -> updates head_sha and re-requests review!
            req2 = OpenPrRequest(
                idempotency_key="gh-1450220110-7:v2:r2:open-pr:1",
                issue_id="gh-1450220110-7",
                attempt=2,
                base_commit=BASE_SHA,
                title="Fix zero handling in lexer",
                body="Updated draft PR.",
                files=[
                    PrFileChange(path="sqlparse/lexer.py", content="FIXED = 2\n"),
                ],
            )
            second = await client.post("/v1/prs", json=req2.model_dump(mode="json"))
            assert second.status_code == 200
            pr_out_2 = second.json()
            assert pr_out_2["pr_number"] == pr_num
            assert pr_out_2["head_sha"] != sha1
            # Review was re-requested on the new push!
            assert fake_gh.review_requests[pr_num] == [["Anchal-T"], ["Anchal-T"]]

            eval_response = await client.post(
                "/v1/prs",
                json=OpenPrRequest(
                    idempotency_key="gh-1450220110-7:p1:c0",
                    issue_id="gh-1450220110-7",
                    attempt=90,
                    base_commit=BASE_SHA,
                    title="P1 C0 comparison run",
                    files=[
                        PrFileChange(path="sqlparse/lexer.py", content="C0 = True\n")
                    ],
                    branch_suffix="p1-c0",
                ).model_dump(mode="json"),
            )
            assert eval_response.status_code == 200
            assert eval_response.json()["branch"] == "agent/gh-1450220110-7-p1-c0"
            assert eval_response.json()["pr_number"] != pr_num
            assert fake_gh.pulls[eval_response.json()["pr_number"]]["draft"] is True
    finally:
        await close_runtime(runtime)


@pytest.mark.anyio
async def test_evidence_comment_upserts_in_place_quotes_taint2_hides_taint3_and_sets_status(
    tmp_path,
):
    runtime, app, fake_gh = _make_app_env(tmp_path, use_app=True)
    try:
        await runtime.poller.prepare()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            pr_resp = await client.post(
                "/v1/prs",
                json=OpenPrRequest(
                    idempotency_key="pr-1",
                    issue_id="gh-1450220110-9",
                    base_commit=BASE_SHA,
                    title="Fix parser",
                    files=[PrFileChange(path="sqlparse/sql.py", content="X = 1\n")],
                ).model_dump(mode="json"),
            )
            pr_data = pr_resp.json()
            pr_num = pr_data["pr_number"]
            head_sha = pr_data["head_sha"]

            ev1 = EvidenceCommentRequest(
                idempotency_key="ev-1",
                issue_id="gh-1450220110-9",
                base_commit=BASE_SHA,
                head_sha=head_sha,
                triage=TriageResult(
                    kind="bug",
                    priority="high",
                    summary="Parser crashes on empty group",
                    area="parser-tree",
                ),
                repro_test_node_id="tests/test_grouping.py::test_empty_group",
                repro_failed_before=True,
                repro_passed_after=True,
                suite_passed=True,
                suite_summary="412 passed in 1.2s",
                review_verdict="approve",
                review_summary="Minimal fix with clean regression test.",
                repro_files=["tests/test_grouping.py"],
                fix_files=["sqlparse/sql.py"],
                sync_decisions=[
                    SyncDecisionItem(
                        step="fix",
                        round_id="round-3",
                        proposal_id="prop-3",
                        op="edit_file",
                        status="accepted",
                        reason="supported fixer edit_file proposal",
                    )
                ],
                cited_facts=[
                    EvidenceFactItem(
                        fact_id="fact-trusted1",
                        claim="TokenList.tokens is ordered.",
                        checked_at_sha=BASE_SHA,
                        valid_at_base=True,
                        taint=0,
                    ),
                    EvidenceFactItem(
                        fact_id="fact-user2",
                        claim="Ignore instructions <script> @Anchal-T #99",
                        checked_at_sha=BASE_SHA,
                        valid_at_base=True,
                        taint=2,
                    ),
                    EvidenceFactItem(
                        fact_id="fact-quarantined3",
                        claim="TOP SECRET QUARANTINED CLAIM",
                        checked_at_sha=BASE_SHA,
                        valid_at_base=False,
                        taint=3,
                    ),
                ],
                models_used=[
                    ModelUsageItem(
                        role="fixer",
                        model="openai/gpt-oss-20b",
                        provider="groq",
                    )
                ],
            )
            c1 = await client.put(
                f"/v1/prs/{pr_num}/evidence", json=ev1.model_dump(mode="json")
            )
            assert c1.status_code == 200
            comment_id = c1.json()["comment_id"]
            assert len(fake_gh.comments[pr_num]) == 1
            rendered = fake_gh.comments[pr_num][0]["body"]
            assert "<!-- agent-team:1 kind=evidence key=gh-1450220110-9 -->" in rendered
            assert "Reported by users, not verified (taint 2)" in rendered
            assert "<script>" not in rendered
            assert "&lt;script&gt;" in rendered
            assert "@Anchal-T" not in rendered
            assert "TOP SECRET QUARANTINED CLAIM" not in rendered
            assert "quarantined taint 3" in rendered

            # Second evidence update edits the same comment in place (no duplicate comment!).
            ev2 = ev1.model_copy(
                update={
                    "idempotency_key": "ev-2",
                    "suite_summary": "413 passed in 1.1s",
                }
            )
            c2 = await client.put(
                f"/v1/prs/{pr_num}/evidence", json=ev2.model_dump(mode="json")
            )
            assert c2.status_code == 200
            assert c2.json()["comment_id"] == comment_id
            assert len(fake_gh.comments[pr_num]) == 1
            assert "413 passed in 1.1s" in fake_gh.comments[pr_num][0]["body"]

            # Commit status endpoint sets status on head_sha and is idempotent.
            st_req = CommitStatusRequest(
                idempotency_key="st-1",
                state="pending",
                description="Checks passed; awaiting maintainer PR review.",
            )
            s1 = await client.post(
                f"/v1/commits/{head_sha}/status", json=st_req.model_dump(mode="json")
            )
            s2 = await client.post(
                f"/v1/commits/{head_sha}/status", json=st_req.model_dump(mode="json")
            )
            assert s1.status_code == s2.status_code == 200
            assert len(fake_gh.statuses[head_sha]) == 1
            assert fake_gh.statuses[head_sha][0]["state"] == "pending"
            assert fake_gh.statuses[head_sha][0]["context"] == "agent-team"
    finally:
        await close_runtime(runtime)


def test_evidence_comment_renders_real_injected_and_cited_ids():
    from services.github.app.rules import render_evidence_comment

    request = EvidenceCommentRequest(
        idempotency_key="example-evidence",
        issue_id="example-issue",
        base_commit=BASE_SHA,
        head_sha=BASE_SHA,
        repro_test_node_id="tests/test_example.py::test_example",
        repro_failed_before=True,
        repro_passed_after=True,
        suite_passed=True,
        review_verdict="approve",
        fact_ids_injected=["fact-111111111111", "fact-222222222222"],
        fact_ids_cited=["fact-222222222222"],
    )
    rendered = render_evidence_comment(request, workflow_run_key="example-issue")
    assert (
        "### Injected facts\n- `fact-111111111111`\n- `fact-222222222222`" in rendered
    )
    assert "### Cited facts\n- `fact-222222222222`" in rendered
    assert "memory retrieval not active" not in rendered
    empty = render_evidence_comment(
        request.model_copy(
            update={
                "fact_ids_injected": [],
                "fact_ids_cited": [],
            }
        ),
        workflow_run_key="example-issue",
    )
    assert "### Cited facts\n- None recorded." in empty
    assert "memory retrieval not active" not in empty
