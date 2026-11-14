from datetime import UTC, datetime

import pytest

from services.memory.app.dag import Commit, Dag, UnknownCommit
from services.memory.app.fact_query import filter_valid, valid_at


def _make_commit(sha: str, parents: tuple[str, ...], day: int) -> Commit:
    return Commit(
        sha=sha,
        parents=parents,
        committed_at=datetime(2026, 1, day, tzinfo=UTC),
        branch="main",
        author="test",
        changed_symbols=(),
    )


@pytest.fixture
def test_dag() -> Dag:
    # c0 -> c1 -> c2 -> c3
    #  \-> s1 -> s2
    commits = [
        _make_commit("c0", (), 1),
        _make_commit("c1", ("c0",), 2),
        _make_commit("c2", ("c1",), 3),
        _make_commit("c3", ("c2",), 4),
        _make_commit("s1", ("c0",), 5),
        _make_commit("s2", ("s1",), 6),
    ]
    return Dag.from_commits(commits)


def test_1_1_fact_from_ancestor_with_no_closure_is_returned(test_dag):
    fact = {
        "_id": "fact-1",
        "valid_from_commit": "c0",
        "claim": "Lexer tokenizes correctly",
    }
    closures = {}

    assert valid_at(fact, set(), test_dag, "c1") is True

    valid, dropped = filter_valid([fact], closures, test_dag, "c1")
    assert valid == [fact]
    assert dropped == {"not_ancestor": 0, "closed": 0, "expired": 0}


def test_1_2_fact_created_on_non_ancestor_branch_is_dropped(test_dag):
    fact = {
        "_id": "fact-side",
        "valid_from_commit": "s1",
        "claim": "Feature branch fact",
    }
    closures = {}

    assert valid_at(fact, set(), test_dag, "c1") is False

    valid, dropped = filter_valid([fact], closures, test_dag, "c1")
    assert valid == []
    assert dropped == {"not_ancestor": 1, "closed": 0, "expired": 0}


def test_1_3_fact_closed_by_ancestor_commit_is_dropped(test_dag):
    fact = {
        "_id": "fact-1",
        "valid_from_commit": "c0",
        "claim": "Old behavior",
    }
    closures = {"fact-1": {"c1"}}

    assert valid_at(fact, {"c1"}, test_dag, "c2") is False

    valid, dropped = filter_valid([fact], closures, test_dag, "c2")
    assert valid == []
    assert dropped == {"not_ancestor": 0, "closed": 1, "expired": 0}


def test_1_4_fact_closed_only_on_side_branch_is_kept(test_dag):
    fact = {
        "_id": "fact-1",
        "valid_from_commit": "c0",
        "claim": "Still true on main",
    }
    closures = {"fact-1": {"s1"}}

    assert valid_at(fact, {"s1"}, test_dag, "c1") is True

    valid, dropped = filter_valid([fact], closures, test_dag, "c1")
    assert valid == [fact]
    assert dropped == {"not_ancestor": 0, "closed": 0, "expired": 0}

    # But on side branch s2, it is closed:
    assert valid_at(fact, {"s1"}, test_dag, "s2") is False
    valid_side, dropped_side = filter_valid([fact], closures, test_dag, "s2")
    assert valid_side == []
    assert dropped_side == {"not_ancestor": 0, "closed": 1, "expired": 0}


def test_1_5_revert_reasserted_fact_lifecycle(test_dag):
    # C0 -> C1 -> C2
    # F1 asserted at C0, closed at C1
    # F2 asserted at C2 with supersedes="fact-1"
    f1 = {"_id": "fact-1", "valid_from_commit": "c0", "supersedes": None}
    f2 = {"_id": "fact-2", "valid_from_commit": "c2", "supersedes": "fact-1"}
    closures = {"fact-1": {"c1"}, "fact-2": set()}
    facts = [f1, f2]

    # At C0: F1 is valid, F2 is not yet an ancestor
    v_c0, d_c0 = filter_valid(facts, closures, test_dag, "c0")
    assert v_c0 == [f1]
    assert d_c0 == {"not_ancestor": 1, "closed": 0, "expired": 0}

    # At C1 (C1 <= base < C2): F1 is closed, F2 is not yet an ancestor -> both dropped
    v_c1, d_c1 = filter_valid(facts, closures, test_dag, "c1")
    assert v_c1 == []
    assert d_c1 == {"not_ancestor": 1, "closed": 1, "expired": 0}

    # At C2 (base >= C2): F1 is closed, F2 is valid!
    v_c2, d_c2 = filter_valid(facts, closures, test_dag, "c2")
    assert v_c2 == [f2]
    assert d_c2 == {"not_ancestor": 0, "closed": 1, "expired": 0}

    # At C3 (base > C2): F1 is closed, F2 is valid!
    v_c3, d_c3 = filter_valid(facts, closures, test_dag, "c3")
    assert v_c3 == [f2]
    assert d_c3 == {"not_ancestor": 0, "closed": 1, "expired": 0}


def test_1_6_ancestry_filter_disabled_returns_all(test_dag):
    f_side = {"_id": "fact-side", "valid_from_commit": "s1"}
    f_closed = {"_id": "fact-closed", "valid_from_commit": "c0"}
    closures = {"fact-closed": {"c1"}}
    facts = [f_side, f_closed]

    # With ancestry_filter=True, both are dropped at c2 (f_side is not_ancestor, f_closed is closed)
    v_filt, d_filt = filter_valid(facts, closures, test_dag, "c2", ancestry_filter=True)
    assert v_filt == []
    assert d_filt == {"not_ancestor": 1, "closed": 1, "expired": 0}

    # With ancestry_filter=False, both are returned!
    v_no_filt, d_no_filt = filter_valid(
        facts, closures, test_dag, "c2", ancestry_filter=False
    )
    assert v_no_filt == facts
    assert d_no_filt == {"not_ancestor": 0, "closed": 0, "expired": 0}


def test_unknown_base_raises_key_error(test_dag):
    with pytest.raises((KeyError, UnknownCommit)):
        valid_at({"valid_from_commit": "c0"}, set(), test_dag, "unknown-sha")

    with pytest.raises((KeyError, UnknownCommit)):
        filter_valid([], {}, test_dag, "unknown-sha")
