from datetime import UTC, datetime, timedelta

from services.memory.app.dag import Commit, Dag
from services.memory.app.fact_closer import FactCloser


class FakeStore:
    def __init__(self, facts=(), closures=()):
        self.facts = list(facts)
        self.closures = list(closures)
        self.ended = []

    def facts_for_symbols(self, symbols):
        return [
            fact
            for fact in self.facts
            if any(anchor["symbol"] in symbols for anchor in fact["anchors"])
        ]

    def closures_for_facts(self, fact_ids):
        return [closure for closure in self.closures if closure["fact_id"] in fact_ids]

    def facts_by_ids(self, fact_ids):
        return [fact for fact in self.facts if fact["_id"] in fact_ids]

    def all_fact_closures(self):
        return list(self.closures)

    def end_fact_valid_time(self, fact_ids, committed_at):
        self.ended.append((tuple(fact_ids), committed_at))

    def record_fact_closure(self, **closure):
        key = (closure["fact_id"], closure["closing_commit"])
        if not any(
            (row["fact_id"], row["closing_commit"]) == key for row in self.closures
        ):
            self.closures.append(closure)
        return f"{key[0]}@{key[1]}"


def _commit(sha, parents=(), changed=(), merge_side=(), branch="main"):
    return Commit(
        sha=sha,
        parents=parents,
        committed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=len(sha)),
        branch=branch,
        author=None,
        changed_symbols=changed,
        merge_side_symbols=merge_side,
    )


def _fact(fact_id, valid_from, *symbols):
    return {
        "_id": fact_id,
        "valid_from_commit": valid_from,
        "anchors": [{"symbol": symbol, "sha": valid_from} for symbol in symbols],
        "evidence_test": f"tests/{fact_id}.py::test_fact",
    }


def test_closes_only_facts_valid_at_parent_with_changed_anchors_and_is_idempotent():
    root = _commit("a")
    parent = _commit("b", (root.sha,), changed=("other",))
    child = _commit("c", (parent.sha,), changed=("changed",))
    dag = Dag((root, parent, child))
    store = FakeStore(
        facts=(
            _fact("fact-valid", root.sha, "changed"),
            _fact("fact-unrelated", root.sha, "other-symbol"),
            _fact("fact-not-yet-valid", child.sha, "changed"),
            _fact("fact-already-closed", root.sha, "changed"),
        ),
        closures=({"fact_id": "fact-already-closed", "closing_commit": parent.sha},),
    )
    closer = FactCloser(dag, store)

    assert closer.close_facts_for_commit(child.sha, "main") == ["fact-valid"]
    assert closer.close_facts_for_commit(child.sha, "main") == ["fact-valid"]
    assert [
        row["fact_id"] for row in store.closures if row["closing_commit"] == child.sha
    ] == ["fact-valid"]


def test_backfill_walks_descendants_and_closes_a_late_fact():
    root = _commit("a")
    changed = _commit("b", (root.sha,), changed=("late-symbol",))
    tip = _commit("c", (changed.sha,))
    store = FakeStore(facts=(_fact("fact-late", root.sha, "late-symbol"),))

    assert FactCloser(Dag((root, changed, tip)), store).backfill(
        ["fact-late"], main_head=tip.sha
    ) == [changed.sha]
    assert store.closures[0]["closing_commit"] == changed.sha
    assert store.closures[0]["closed_by_agent"] == "backfill"
    assert store.ended == [(("fact-late",), changed.committed_at)]


def test_advance_main_projects_a_branch_closure_at_the_first_merge():
    root = _commit("a")
    main = _commit("b", (root.sha,))
    side = _commit("c", (root.sha,), branch="feature")
    merge = _commit("d", (main.sha, side.sha))
    store = FakeStore(
        closures=(
            {
                "fact_id": "fact-side",
                "closing_commit": side.sha,
            },
        )
    )
    closer = FactCloser(Dag((root, main, side, merge)), store)

    assert closer.advance_main(main.sha) == []
    assert closer.advance_main(merge.sha) == ["fact-side"]
    assert store.ended == [(("fact-side",), merge.committed_at)]


def test_merge_checks_fact_validity_and_changed_symbols_against_each_parent():
    root = _commit("a")
    main = _commit("b", (root.sha,), changed=("main-symbol",))
    side = _commit("c", (root.sha,), changed=("side-symbol",), branch="feature")
    merge = _commit(
        "d",
        (main.sha, side.sha),
        changed=("main-symbol",),
        merge_side=((side.sha, ("side-symbol",)),),
    )
    dag = Dag((root, main, side, merge))
    store = FakeStore(
        facts=(
            _fact("fact-main", main.sha, "main-symbol"),
            _fact("fact-side", side.sha, "side-symbol"),
            _fact("fact-root", root.sha, "both"),
        )
    )

    closed = FactCloser(dag, store).close_facts_for_commit(merge.sha, "main")

    assert closed == ["fact-main", "fact-side"]
    assert {(row["fact_id"], row["closing_commit"]) for row in store.closures} == {
        ("fact-main", merge.sha),
        ("fact-side", merge.sha),
    }
