from datetime import UTC, datetime

import httpx
import pytest
from agentteam_common import (
    Settings,
    is_safe_relative_path,
    is_test_path,
)
from agentteam_contracts import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
    CommitStatusRequest,
    EvidenceCommentRequest,
    GithubWriteOutcome,
    IssueRunRequest,
    IssueRunRequestV2,
    IssueRunResultV2,
    ModelCallRequest,
    OpenPrRequest,
    PrFileChange,
    Sampling,
)
from pydantic import ValidationError

from services.workflow.app.models.client import ModelClient
from services.workflow.app.models.ledger import Ledger
from services.workflow.app.models.pools import pin_for
from services.workflow.app.models.store import Store

FIXED_NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)


def test_path_helpers_classify_safe_test_and_source_paths():
    assert is_safe_relative_path("sqlparse/lexer.py")
    assert is_safe_relative_path("tests/test_regressions.py")
    assert not is_safe_relative_path("")
    assert not is_safe_relative_path("/etc/passwd")
    assert not is_safe_relative_path("../secret.py")
    assert not is_safe_relative_path("tests/../../secret.py")
    assert not is_safe_relative_path(".git/config")
    assert not is_safe_relative_path("a\\b.py")

    assert is_test_path("tests/test_regressions.py")
    assert is_test_path("tests/helpers/utils.py")
    assert is_test_path("conftest.py")
    assert is_test_path("test_standalone.py")
    assert is_test_path("src/pkg/parser_test.py")
    assert not is_test_path("sqlparse/lexer.py")
    assert not is_test_path("src/marshmallow/fields.py")
    assert not is_test_path("../tests/test_escape.py")


@pytest.mark.parametrize("role", ["triager", "reproducer", "fixer", "tester"])
def test_pins_and_model_call_requests_support_all_four_phase4_roles(role, tmp_path):
    settings = Settings.from_env()
    pin = pin_for(role, settings)
    assert pin.role == role
    assert pin.pool == "groq"
    assert pin.prompt_token_cap + pin.max_tokens <= settings.groq_tpm

    req = ModelCallRequest(
        run_id=f"run-{role}",
        step=role,
        role=role,
        messages=[{"role": "user", "content": "Check behavior"}],
        sampling=Sampling(),
        taint=1,
        source="github_issue",
    )
    assert req.role == role

    def provider(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(
            200,
            json={
                "model": pin.model,
                "choices": [{"message": {"content": '{"action":"submit"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        return body

    role_settings = Settings(
        models_sqlite_path=str(tmp_path / f"{role}.sqlite"),
        groq_api_key="test-key",
    )
    store = Store(role_settings)
    client = ModelClient(
        settings=role_settings,
        ledger=Ledger(role_settings, store),
        store=store,
        http=httpx.Client(transport=httpx.MockTransport(provider)),
        now=lambda: FIXED_NOW,
    )
    outcome = client.call(req)
    assert outcome.outcome == "completed"

    with store.connection() as conn:
        attempt_role = conn.execute("SELECT role FROM call_attempt").fetchone()["role"]
        log_role = conn.execute("SELECT role FROM call_log").fetchone()["role"]
    assert attempt_role == role
    assert log_role == role


def test_settings_reject_role_token_budget_exceeding_groq_tpm(monkeypatch):
    monkeypatch.setenv("FIXER_PROMPT_TOKEN_CAP", "7000")
    monkeypatch.setenv("FIXER_MAX_TOKENS", "1500")
    with pytest.raises(ValueError, match="GROQ_TPM"):
        Settings.from_env()


def test_phase4_contracts_validate_paths_outcomes_and_v2_compatibility():
    base_req = IssueRunRequest(
        issue_id="gh-100-7",
        repo="Anchal-T/sqlparse",
        title="Broken lexer token",
        body="Details",
    )
    # IssueRunRequest dump shape stays unchanged for V1 callers.
    assert set(base_req.model_dump().keys()) == {
        "issue_id",
        "repo",
        "title",
        "body",
        "read_basis",
        "attempt",
    }
    v2_req = IssueRunRequestV2.model_validate(base_req.model_dump())
    assert v2_req.base_commit is None

    with pytest.raises(ValidationError):
        PrFileChange(path="../outside.py", content="x")

    pr_req = OpenPrRequest(
        idempotency_key="gh-100-7:v2:open-pr",
        issue_id="gh-100-7",
        base_commit="abcdef1234567",
        title="Fix lexer token",
        files=[PrFileChange(path="sqlparse/lexer.py", content="x = 1\n")],
    )
    assert pr_req.attempt == 1

    ev_req = EvidenceCommentRequest(
        idempotency_key="gh-100-7:v2:evidence",
        issue_id="gh-100-7",
        base_commit="abcdef1234567",
        head_sha="1234567abcdef",
        repro_test_node_id="tests/test_regressions.py::test_issue_7",
        repro_failed_before=True,
        repro_passed_after=True,
        suite_passed=True,
        review_verdict="approve",
    )
    assert ev_req.attempt == 1

    status_req = CommitStatusRequest(
        idempotency_key="gh-100-7:v2:status:pending",
        state="pending",
        description="Awaiting PR review",
    )
    assert status_req.context == "agent-team"

    ok = GithubWriteOutcome(outcome="ok", decided_at=FIXED_NOW, pr_number=12)
    assert ok.pr_number == 12
    with pytest.raises(ValidationError):
        GithubWriteOutcome(outcome="throttled", decided_at=FIXED_NOW)
    with pytest.raises(ValidationError):
        GithubWriteOutcome(outcome="failed", decided_at=FIXED_NOW, error_code="only")

    decision = ApprovalDecision(approved=True, by="Anchal-T", head_sha="1234567abcdef")
    approval_req = ApprovalRequest(
        issue_id="gh-100-7", approved=False, by="Anchal-T", reason="Fix edge case"
    )
    approval_res = ApprovalResult(
        issue_id="gh-100-7", status="recorded", stage="closed"
    )
    result_v2 = IssueRunResultV2(
        issue_id="gh-100-7",
        status="approved",
        stage="merged",
        base_commit="abcdef1234567",
        pr_number=12,
        pr_url="https://github.com/Anchal-T/sqlparse/pull/12",
        pr_head_sha="1234567abcdef",
        approval=decision,
    )
    assert approval_req.reason == "Fix edge case"
    assert approval_res.stage == "closed"
    assert result_v2.stage == "merged"
