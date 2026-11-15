from datetime import UTC, datetime

import pytest
from agentteam_contracts import FactView, IssueRunRequestV2

from services.workflow.app import issue_run_v2 as v2_mod


def test_live_condition_contract_requires_frozen_c2_snapshot_and_keeps_c0_empty():
    request = IssueRunRequestV2(
        issue_id="gh-1450220110-5",
        repo="Anchal-T/sqlparse",
        title="Issue title",
        base_commit="a" * 40,
        attempt=91,
        fact_condition="C2",
        fact_snapshot=[],
        pr_branch_suffix="p1-c2",
    )
    assert request.fact_condition == "C2"

    with pytest.raises(ValueError, match="frozen fact snapshot"):
        IssueRunRequestV2(
            issue_id="gh-1450220110-5",
            repo="Anchal-T/sqlparse",
            title="Issue title",
            fact_condition="C2",
        )
    with pytest.raises(ValueError, match="cannot carry a fact snapshot"):
        IssueRunRequestV2(
            issue_id="gh-1450220110-5",
            repo="Anchal-T/sqlparse",
            title="Issue title",
            fact_condition="C0",
            fact_snapshot=[],
        )


class _Journal:
    def __init__(self):
        self.steps = []
        self.events = []

    async def run_step(self, name, fn, *args, **kwargs):
        self.steps.append(name)
        return await fn(*args, **kwargs)

    async def record_event(self, name, event):
        self.events.append((name, event))
        return f"event-{len(self.events)}"


@pytest.mark.anyio
async def test_c0_skips_memory_retrieval(monkeypatch):
    async def forbidden_query(*args, **kwargs):
        raise AssertionError("C0 must not query memory")

    monkeypatch.setattr(v2_mod, "query_fact_block", forbidden_query)
    journal = _Journal()
    result = await v2_mod.read_facts(
        journal,  # type: ignore[arg-type]
        prefix="run",
        step="repro",
        round_id="round",
        base_commit="a" * 40,
        text="issue",
        now=datetime(2026, 11, 16, tzinfo=UTC),
        fact_condition="C0",
    )

    assert result["text"] == ""
    assert result["injected_ids"] == []
    assert result["event_ids"] == []
    assert journal.steps == []
    assert journal.events == []


@pytest.mark.anyio
async def test_c2_uses_frozen_fact_snapshot_without_memory_io(monkeypatch):
    async def forbidden_memory(*args, **kwargs):
        raise AssertionError("frozen C2 retrieval must not read live memory")

    monkeypatch.setattr(v2_mod, "post_memory", forbidden_memory)
    now = datetime(2026, 11, 16, tzinfo=UTC)
    fact = FactView(
        id="fact-123456789abc",
        claim="the frozen useful fact",
        kind="bug_cause",
        valid_from_commit="a" * 40,
        confidence=1.0,
        last_confirmed_at=now,
    )
    journal = _Journal()
    result = await v2_mod.read_facts(
        journal,  # type: ignore[arg-type]
        prefix="run",
        step="repro",
        round_id="round",
        base_commit="a" * 40,
        text="issue",
        now=now,
        fact_condition="C2",
        fact_snapshot=[fact.model_dump(mode="json")],
    )

    assert result["injected_ids"] == [fact.id]
    assert fact.claim in result["text"]
    assert journal.steps == ["query-facts:repro"]
    assert journal.events[0][1].payload["condition"] == "C2"
