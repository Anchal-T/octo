from datetime import UTC, datetime

import pytest

from services.memory.app.dag import Commit, Dag

COMMITTED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _commit(
    sha,
    parents=(),
    *,
    branch="main",
    changed_symbols=(),
    merge_side_symbols=(),
):
    return Commit(
        sha=sha,
        parents=parents,
        committed_at=COMMITTED_AT,
        branch=branch,
        author=None,
        changed_symbols=changed_symbols,
        merge_side_symbols=merge_side_symbols,
    )


def test_dag_tracks_branch_ancestry_descendants_and_parent_symbol_diffs():
    root = _commit("a")
    main = _commit("b", ("a",), changed_symbols=("sym-main",))
    branch = _commit("c", ("a",), branch="feature", changed_symbols=("sym-branch",))
    merge = _commit(
        "d",
        ("b", "c"),
        changed_symbols=("sym-main",),
        merge_side_symbols=(("c", ("sym-branch", "sym-merge")),),
    )
    dag = Dag((root, main, branch, merge))

    assert dag.ancestors("d") == frozenset({"a", "b", "c", "d"})
    assert dag.is_ancestor("c", "d")
    assert not dag.is_ancestor("c", "b")
    assert dag.descendants("a") == frozenset({"b", "c", "d"})
    assert merge.symbols_vs("b") == ("sym-main",)
    assert merge.symbols_vs("c") == ("sym-branch", "sym-merge")

    dag.add(_commit("e", ("d",)))
    assert dag.ancestors("e") == frozenset({"a", "b", "c", "d", "e"})
    assert "e" in dag.descendants("a")


def test_dag_rejects_unknown_bases_and_parents():
    dag = Dag((_commit("a"),))

    with pytest.raises(KeyError):
        dag.ancestors("missing")
    with pytest.raises(KeyError):
        dag.add(_commit("b", ("missing",)))
