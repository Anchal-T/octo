from datetime import UTC, datetime, timedelta

from services.memory.app.time_linear_projector import TimeLinearProjector

from services.memory.app.dag import Commit, Dag


class FakeStore:
    def __init__(self):
        self.ended = []

    def end_fact_valid_time(self, fact_ids, committed_at):
        self.ended.append((tuple(fact_ids), committed_at))


def _commit(sha, parents=(), branch="main"):
    return Commit(
        sha=sha,
        parents=parents,
        committed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=len(sha)),
        branch=branch,
        author=None,
        changed_symbols=(),
    )


def test_projects_a_side_branch_closure_at_its_first_main_merge():
    root = _commit("a")
    main = _commit("b", (root.sha,))
    side = _commit("c", (root.sha,), branch="feature")
    merge = _commit("d", (main.sha, side.sha))
    store = FakeStore()

    end = TimeLinearProjector(Dag((root, main, side, merge)), store).end_for(
        "fact-1", side.sha, main_head=merge.sha
    )

    assert end == merge.committed_at
    assert store.ended == [(("fact-1",), merge.committed_at)]


def test_does_not_project_a_closure_not_yet_reachable_from_main():
    root = _commit("a")
    main = _commit("b", (root.sha,))
    side = _commit("c", (root.sha,), branch="feature")
    store = FakeStore()

    end = TimeLinearProjector(Dag((root, main, side)), store).end_for(
        "fact-1", side.sha, main_head=main.sha
    )

    assert end is None
    assert store.ended == []
