import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from psycopg.types.json import Jsonb

from services.memory.app.dag import Commit, Dag
from services.memory.app.fact_closer import FactCloser
from services.memory.app.store import MemoryStore

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTTEAM_TEST_XTDB") != "1" or os.getenv("XTDB_PORT") != "5433",
    reason="set AGENTTEAM_TEST_XTDB=1 and XTDB_PORT=5433 to opt into XTDB tests",
)


def test_fact_lookup_and_closure_write_are_idempotent():
    store = MemoryStore()
    fact_id = f"fact-{uuid4().hex}"
    committed_at = datetime(2026, 1, 1, tzinfo=UTC)
    symbol = f"symbol-{uuid4().hex}"
    anchors = [
        {"symbol": symbol, "sha": "commit-a"},
        {"symbol": f"other-{uuid4().hex}", "sha": "commit-a"},
    ]
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO facts (_id, _valid_from, valid_from_commit, anchors, evidence_test) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                fact_id,
                committed_at,
                "commit-a",
                Jsonb(anchors),
                "tests/test_x.py::test_x",
            ),
        )

    assert store.facts_for_symbols([symbol, "missing"]) == [
        {
            "_id": fact_id,
            "valid_from_commit": "commit-a",
            "anchors": anchors,
            "evidence_test": "tests/test_x.py::test_x",
        }
    ]

    closure = {
        "fact_id": fact_id,
        "closing_commit": "commit-b",
        "branch": "main",
        "closed_by_agent": "on_push",
        "reason": "anchored symbol changed",
        "evidence_test": "tests/test_x.py::test_x",
    }
    closure_id = store.record_fact_closure(
        **closure, committed_at=datetime(2026, 1, 2, tzinfo=UTC)
    )
    assert (
        store.record_fact_closure(
            **closure, committed_at=datetime(2026, 1, 2, tzinfo=UTC)
        )
        == closure_id
    )
    assert store.closures_for_facts([fact_id]) == [
        {
            "fact_id": fact_id,
            "closing_commit": "commit-b",
        }
    ]


def test_main_projection_ends_valid_time_at_the_first_merge():
    store = MemoryStore()
    root_sha, main_sha, side_sha, merge_sha = [uuid4().hex + "0" * 8 for _ in range(4)]
    times = [datetime(2026, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)]
    root = Commit(root_sha, (), times[0], "main", None, ())
    main = Commit(main_sha, (root_sha,), times[1], "main", None, ())
    side = Commit(side_sha, (root_sha,), times[2], "feature", None, ())
    merge = Commit(merge_sha, (main_sha, side_sha), times[3], "main", None, ())
    for commit in (root, main, side, merge):
        store.record_commit(commit)

    fact_id = f"fact-{uuid4().hex}"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO facts (_id, _valid_from, valid_from_commit, anchors) "
            "VALUES (%s, %s, %s, %s)",
            (
                fact_id,
                times[0],
                root_sha,
                Jsonb([{"symbol": "sym", "sha": root_sha}]),
            ),
        )
    store.record_fact_closure(
        fact_id=fact_id,
        closing_commit=side_sha,
        branch="feature",
        closed_by_agent="on_push",
        reason="anchored symbol changed",
        committed_at=times[2],
    )
    closer = FactCloser(Dag((root, main, side, merge)), store)

    assert closer.advance_main(main_sha) == []
    assert closer.advance_main(merge_sha) == [fact_id]
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT _valid_from, _valid_to FROM facts FOR ALL VALID_TIME WHERE _id = %s",
            (fact_id,),
        ).fetchall()
    assert rows == [{"_valid_from": times[0], "_valid_to": times[3]}]


def test_backfill_closes_late_facts_and_projects_the_time_linear_view():
    store = MemoryStore()
    root_sha, changed_sha, tip_sha = [uuid4().hex + "0" * 8 for _ in range(3)]
    times = [datetime(2026, 1, day, tzinfo=UTC) for day in (1, 2, 3)]
    root = Commit(root_sha, (), times[0], "main", None, ())
    changed = Commit(
        changed_sha, (root_sha,), times[1], "main", None, (f"sym-{uuid4().hex}",)
    )
    tip = Commit(tip_sha, (changed_sha,), times[2], "main", None, ())
    for commit in (root, changed, tip):
        store.record_commit(commit)

    fact_id = f"fact-{uuid4().hex}"
    symbol = changed.changed_symbols[0]
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO facts (_id, _valid_from, valid_from_commit, anchors) "
            "VALUES (%s, %s, %s, %s)",
            (fact_id, times[0], root_sha, Jsonb([{"symbol": symbol, "sha": root_sha}])),
        )
    closer = FactCloser(Dag((root, changed, tip)), store)

    assert closer.backfill([fact_id], main_head=tip_sha) == [changed_sha]
    assert closer.backfill([fact_id], main_head=tip_sha) == [changed_sha]
    assert store.closures_for_facts([fact_id]) == [
        {"fact_id": fact_id, "closing_commit": changed_sha}
    ]
    with store._connect() as conn:
        row = conn.execute(
            "SELECT _valid_from, _valid_to FROM facts FOR ALL VALID_TIME WHERE _id = %s",
            (fact_id,),
        ).fetchone()
    assert row == {"_valid_from": times[0], "_valid_to": times[1]}


def test_fact_closer_persists_only_facts_valid_at_the_parent():
    store = MemoryStore()
    root_sha, parent_sha, child_sha = [uuid4().hex + "0" * 8 for _ in range(3)]
    times = [datetime(2026, 1, day, tzinfo=UTC) for day in (1, 2, 3)]
    changed_symbol = f"changed-{uuid4().hex}"
    commits = (
        Commit(root_sha, (), times[0], "main", None, ()),
        Commit(parent_sha, (root_sha,), times[1], "main", None, ()),
        Commit(child_sha, (parent_sha,), times[2], "main", None, (changed_symbol,)),
    )
    for commit in commits:
        store.record_commit(commit)

    fact_ids = [f"fact-{uuid4().hex}" for _ in range(3)]
    rows = (
        (fact_ids[0], root_sha, changed_symbol),
        (fact_ids[1], child_sha, changed_symbol),
        (fact_ids[2], root_sha, changed_symbol),
    )
    with store._connect() as conn:
        for fact_id, valid_from, symbol in rows:
            conn.execute(
                "INSERT INTO facts (_id, _valid_from, valid_from_commit, anchors) "
                "VALUES (%s, %s, %s, %s)",
                (
                    fact_id,
                    times[0],
                    valid_from,
                    Jsonb([{"symbol": symbol, "sha": valid_from}]),
                ),
            )
    store.record_fact_closure(
        fact_id=fact_ids[2],
        closing_commit=parent_sha,
        branch="main",
        closed_by_agent="on_push",
        reason="already stale",
        committed_at=times[1],
    )

    closer = FactCloser(Dag(commits), store)
    assert closer.close_facts_for_commit(child_sha, "main") == [fact_ids[0]]
    assert closer.close_facts_for_commit(child_sha, "main") == [fact_ids[0]]
    assert {
        (row["fact_id"], row["closing_commit"])
        for row in store.closures_for_facts(fact_ids)
    } == {(fact_ids[2], parent_sha), (fact_ids[0], child_sha)}
