import hashlib
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from agentteam_contracts import EventWrite, ProposalWrite, RoundWrite, RunWrite

from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.scip_indexer import ScipIndexer
from services.memory.app.store import MemoryStore, RoundConflict

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTTEAM_TEST_XTDB") != "1" or os.getenv("XTDB_PORT") != "5433",
    reason="set AGENTTEAM_TEST_XTDB=1 and XTDB_PORT=5433 to opt into XTDB tests",
)

COMMIT_TIME = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
SYMBOL = "scip-python python demo 1.0 `demo.module`/example()."


class FixedIndexer(ScipIndexer):
    def __init__(self, sha: str):
        self.sha = sha

    def load(self, sha: str) -> ScipIndex:
        return ScipIndex(
            {
                SYMBOL: (
                    ScipDefinition(
                        symbol=SYMBOL,
                        relative_path="module.py",
                        range=(0, 0, 1),
                        enclosing_range=None,
                    ),
                )
            },
            sha=self.sha,
        )

    def committer_date(self, sha: str) -> datetime:
        return COMMIT_TIME


def _new_round(store, base_commit):
    key = str(uuid4())
    row = store.create_round(
        RoundWrite(
            idempotency_key=key,
            seq=1,
            base_commit=base_commit,
            read_basis=COMMIT_TIME,
        )
    )
    return row["_id"]


def _new_run_and_evidence(store, round_id, role="triager"):
    evidence_ids = [
        store.create_event(
            EventWrite(
                idempotency_key=str(uuid4()),
                source="user_feedback",
                kind="note",
                taint=taint,
                payload={},
            )
        )["_id"]
        for taint in (1, 3)
    ]
    run_id = store.create_run(
        RunWrite(
            idempotency_key=str(uuid4()),
            agent_id=f"{role}-test",
            role=role,
            round_id=round_id,
            workflow_id="workflow-test",
            step="triage",
            outcome="routed",
            event_ids=evidence_ids,
            started_at=COMMIT_TIME,
        )
    )["_id"]
    return run_id, evidence_ids


def _proposal(
    store, round_id, run_id, evidence_ids, *, payload, key=None, agent_id="triager-test"
):
    return store.create_proposal(
        ProposalWrite(
            idempotency_key=key or str(uuid4()),
            round_id=round_id,
            agent_id=agent_id,
            run_id=run_id,
            op="assert_fact",
            evidence_event_ids=evidence_ids,
            payload=payload,
        )
    )


def _payload(**overrides):
    return {
        "claim": "A supported user report",
        "kind": "user_report",
        "area": "demo",
        "anchor_names": [SYMBOL],
        "confidence": 0.75,
        "decay_half_life_days": 30,
        "evidence_test": None,
        "supersedes": None,
        **overrides,
    }


def _read(store, query, params) -> dict | None:
    with store._connect() as conn:
        return conn.execute(query, params).fetchone()


def test_commit_round_persists_normalized_facts_with_proposal_verdicts():
    store = MemoryStore()
    base_commit = "a" * 40
    store.scip_indexer = FixedIndexer(base_commit)
    round_id = _new_round(store, base_commit)
    run_id, evidence_ids = _new_run_and_evidence(store, round_id)

    accepted = _proposal(store, round_id, run_id, evidence_ids, payload=_payload())
    empty_anchor = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        payload=_payload(anchor_names=[]),
    )
    bad_confidence = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        payload=_payload(confidence=1.1),
    )
    unresolved_anchor = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        payload=_payload(anchor_names=["missing-symbol"]),
    )

    result = store.commit_round(round_id)

    fact_id = "fact-" + hashlib.sha1(accepted["_id"].encode()).hexdigest()[:12]
    fact = _read(
        store,
        "SELECT _id, _valid_from, anchors, provenance_event_ids, confidence, taint, "
        "valid_from_commit, created_in_round, proposal_id FROM facts WHERE _id = %s",
        (fact_id,),
    )
    assert fact is not None
    assert result["status"] == "committed"
    assert set(result["committed_proposals"]) == {
        accepted["_id"],
        empty_anchor["_id"],
    }
    assert fact["_valid_from"] == COMMIT_TIME
    assert fact["anchors"] == [{"symbol": SYMBOL, "sha": base_commit}]
    assert fact["provenance_event_ids"] == evidence_ids
    assert fact["confidence"] == 0.75
    assert fact["taint"] == 3
    assert fact["valid_from_commit"] == base_commit
    assert fact["created_in_round"] == round_id
    assert fact["proposal_id"] == accepted["_id"]
    empty_fact_id = (
        "fact-" + hashlib.sha1(empty_anchor["_id"].encode()).hexdigest()[:12]
    )
    empty_fact = _read(
        store, "SELECT anchors FROM facts WHERE _id = %s", (empty_fact_id,)
    )
    assert empty_fact is not None
    assert empty_fact["anchors"] == []
    accepted_row = store.get_proposal(accepted["_id"])
    empty_anchor_row = store.get_proposal(empty_anchor["_id"])
    bad_confidence_row = store.get_proposal(bad_confidence["_id"])
    unresolved_anchor_row = store.get_proposal(unresolved_anchor["_id"])
    assert accepted_row is not None and accepted_row["status"] == "accepted"
    assert empty_anchor_row is not None and empty_anchor_row["status"] == "accepted"
    assert bad_confidence_row is not None and bad_confidence_row["status"] == "rejected"
    assert (
        unresolved_anchor_row is not None
        and unresolved_anchor_row["status"] == "rejected"
    )
    for proposal in (bad_confidence, unresolved_anchor):
        rejected_fact_id = (
            "fact-" + hashlib.sha1(proposal["_id"].encode()).hexdigest()[:12]
        )
        assert (
            _read(store, "SELECT _id FROM facts WHERE _id = %s", (rejected_fact_id,))
            is None
        )


def test_reproducer_can_assert_only_test_backed_bug_cause_facts():
    store = MemoryStore()
    base_commit = "e" * 40
    store.scip_indexer = FixedIndexer(base_commit)
    round_id = _new_round(store, base_commit)
    run_id, evidence_ids = _new_run_and_evidence(store, round_id, role="reproducer")

    valid = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        agent_id="reproducer-test",
        payload=_payload(kind="bug_cause", evidence_test="tests/test_bug.py::test_bug"),
    )
    missing_test = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        agent_id="reproducer-test",
        payload=_payload(kind="bug_cause", evidence_test=None),
    )
    wrong_kind = _proposal(
        store,
        round_id,
        run_id,
        evidence_ids,
        agent_id="reproducer-test",
        payload=_payload(
            kind="user_report", evidence_test="tests/test_bug.py::test_bug"
        ),
    )

    result = store.commit_round(round_id)

    assert result["committed_proposals"] == [valid["_id"]]
    fact_id = "fact-" + hashlib.sha1(valid["_id"].encode()).hexdigest()[:12]
    fact = _read(
        store,
        "SELECT kind, evidence_test FROM facts WHERE _id = %s",
        (fact_id,),
    )
    assert fact == {"kind": "bug_cause", "evidence_test": "tests/test_bug.py::test_bug"}
    for proposal in (missing_test, wrong_kind):
        row = store.get_proposal(proposal["_id"])
        assert row is not None and row["status"] == "rejected"


def test_fact_commit_requires_a_matching_index_and_base_commit():
    store = MemoryStore()

    for base_commit, index_sha in (
        (None, None),
        ("a" * 40, None),
        ("b" * 40, "c" * 40),
    ):
        store.scip_indexer = FixedIndexer(index_sha) if index_sha else None
        round_id = _new_round(store, base_commit)
        run_id, evidence_ids = _new_run_and_evidence(store, round_id)
        proposal = _proposal(store, round_id, run_id, evidence_ids, payload=_payload())

        result = store.commit_round(round_id)

        assert result["committed_proposals"] == []
        proposal_row = store.get_proposal(proposal["_id"])
        assert proposal_row is not None and proposal_row["status"] == "rejected"


def test_fact_insert_conflict_rolls_back_proposal_and_round_updates():
    store = MemoryStore()
    base_commit = "d" * 40
    store.scip_indexer = FixedIndexer(base_commit)
    round_id = _new_round(store, base_commit)
    run_id, evidence_ids = _new_run_and_evidence(store, round_id)
    proposal = _proposal(store, round_id, run_id, evidence_ids, payload=_payload())
    fact_id = "fact-" + hashlib.sha1(proposal["_id"].encode()).hexdigest()[:12]
    with store._connect() as conn:
        conn.execute("INSERT INTO facts (_id) VALUES (%s)", (fact_id,))

    with pytest.raises(RoundConflict):
        store.commit_round(round_id)

    round_row = store.get_round(round_id)
    proposal_row = store.get_proposal(proposal["_id"])
    assert round_row is not None and round_row["status"] == "syncing"
    assert proposal_row is not None and proposal_row["status"] == "pending"
