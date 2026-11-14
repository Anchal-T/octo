import os
from datetime import UTC, datetime

import pytest

from services.memory.app.dag import Commit, Dag, UnknownCommit
from services.memory.app.store import IdempotencyConflict, MemoryStore

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTTEAM_TEST_XTDB") != "1" or os.getenv("XTDB_PORT") != "5433",
    reason="set AGENTTEAM_TEST_XTDB=1 and XTDB_PORT=5433 to opt into XTDB tests",
)

COMMITTED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _commit(sha, parents=(), *, branch="main", changed=(), merge_side=()):
    return Commit(
        sha=sha,
        parents=parents,
        committed_at=COMMITTED_AT,
        branch=branch,
        author="test",
        changed_symbols=changed,
        merge_side_symbols=merge_side,
    )


def test_commit_records_preserve_parent_diffs_and_are_idempotent():
    store = MemoryStore()
    root = _commit("a" * 40)
    main = _commit("b" * 40, (root.sha,), changed=("sym-main",))
    side = _commit("c" * 40, (root.sha,), branch="feature", changed=("sym-side",))
    merge = _commit(
        "d" * 40,
        (main.sha, side.sha),
        changed=("sym-main",),
        merge_side=((side.sha, ("sym-side", "sym-merge")),),
    )

    for commit in (root, main, side, merge):
        store.record_commit(commit)

    row = store.get_commit(merge.sha)
    assert row is not None
    with store._connect() as conn:
        validity = conn.execute(
            "SELECT _valid_from FROM commits WHERE _id = %s", (merge.sha,)
        ).fetchone()
    assert validity is not None
    assert validity["_valid_from"] == COMMITTED_AT
    assert row["committed_at"] == COMMITTED_AT
    assert row["parents"] == [main.sha, side.sha]
    assert row["changed_symbols"] == ["sym-main"]
    assert row["merge_side_symbols"] == [
        {"parent": side.sha, "symbols": ["sym-side", "sym-merge"]}
    ]
    loaded = Dag.from_commits(store.get_commits())
    assert loaded.commits[merge.sha] == merge
    assert store.record_commit(merge) == row

    with pytest.raises(IdempotencyConflict):
        store.record_commit(
            _commit(merge.sha, (main.sha, side.sha), branch="different-branch")
        )
    with pytest.raises(UnknownCommit):
        store.record_commit(_commit("e" * 40, ("f" * 40,)))


def test_metadata_bootstrap_can_upgrade_without_losing_valid_time():
    from dataclasses import replace
    from uuid import uuid4

    store = MemoryStore()
    root = replace(_commit(uuid4().hex + "0" * 8), symbols_indexed=False)
    base = replace(_commit(uuid4().hex + "0" * 8, (root.sha,)), symbols_indexed=False)
    store.record_commit_metadata([root, base])
    store.record_commit_metadata([root, base])
    assert not store.get_dag().commits[base.sha].symbols_indexed
    with pytest.raises(ValueError, match="not indexed"):
        store.get_dag().commits[base.sha].symbols_vs(root.sha)
    indexed = replace(base, symbols_indexed=True, changed_symbols=("example.symbol",))
    store.record_commit(indexed)
    assert store.record_commit(indexed)["changed_symbols"] == ["example.symbol"]
    store.record_commit_metadata([root, base])
    assert store.get_dag().commits[base.sha] == indexed
    with store._connect() as conn:
        row = conn.execute(
            "SELECT _valid_from FROM commits WHERE _id = %s", (base.sha,)
        ).fetchone()
    assert row["_valid_from"] == COMMITTED_AT
    with pytest.raises(IdempotencyConflict):
        store.record_commit_metadata([replace(base, author="different")])
