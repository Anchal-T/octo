from collections import defaultdict

from .dag import Dag
from .store import MemoryStore


class FactCloser:
    def __init__(self, dag: Dag, store: MemoryStore) -> None:
        self.dag = dag
        self.store = store
        self.main_head: str | None = None

    def close_facts_for_commit(
        self, sha: str, branch: str, by: str = "on_push"
    ) -> list[str]:
        return self._close_facts_for_commit(sha, branch, by)

    def advance_main(self, head: str) -> list[str]:
        path = self.dag.first_parent_path(head)
        closures_by_commit: dict[str, set[str]] = defaultdict(set)
        for closure in self.store.all_fact_closures():
            closures_by_commit[closure["closing_commit"]].add(closure["fact_id"])

        seen: frozenset[str] = frozenset()
        projected: set[str] = set()
        for sha in path:
            ancestors = self.dag.ancestors(sha)
            reached = ancestors - seen
            fact_ids = {
                fact_id
                for closing_commit in reached
                for fact_id in closures_by_commit.get(closing_commit, ())
                if fact_id not in projected
            }
            if fact_ids:
                self.store.end_fact_valid_time(
                    sorted(fact_ids), self.dag.commits[sha].committed_at
                )
                projected.update(fact_ids)
            seen = ancestors

        self.main_head = head
        return sorted(projected)

    def backfill(self, fact_ids: list[str], main_head: str | None = None) -> list[str]:
        facts = {fact["_id"]: fact for fact in self.store.facts_by_ids(fact_ids)}
        if not facts:
            return []
        main_head = main_head or self.main_head
        if main_head is None:
            raise ValueError("main_head is required to project backfilled closures")
        self.dag.first_parent_path(main_head)

        relevant_commits = set()
        facts_by_symbol: dict[str, set[str]] = defaultdict(set)
        for fact_id, fact in facts.items():
            valid_from = fact["valid_from_commit"]
            relevant_commits.update(self.dag.descendants(valid_from))
            for anchor in fact.get("anchors") or ():
                facts_by_symbol[anchor["symbol"]].add(fact_id)

        ordered_commits = self.dag.topological_order(relevant_commits)
        closures_by_fact: dict[str, set[str]] = defaultdict(set)
        for closure in self.store.closures_for_facts(list(facts)):
            closures_by_fact[closure["fact_id"]].add(closure["closing_commit"])

        closed: set[tuple[str, str]] = set()
        for sha in ordered_commits:
            commit = self.dag.commits[sha]
            parents_by_fact: dict[str, set[str]] = defaultdict(set)
            for parent in commit.parents:
                changed = set(commit.symbols_vs(parent))
                candidates = set().union(
                    *(
                        facts_by_symbol[symbol]
                        for symbol in changed
                        if symbol in facts_by_symbol
                    )
                )
                if not candidates:
                    continue
                ancestors = self.dag.ancestors(parent)
                for fact_id in candidates:
                    fact = facts[fact_id]
                    if fact["valid_from_commit"] not in ancestors:
                        continue
                    if any(
                        closing in ancestors for closing in closures_by_fact[fact_id]
                    ):
                        continue
                    parents_by_fact[fact_id].add(parent)

            for fact_id, parents in sorted(parents_by_fact.items()):
                if sha not in closures_by_fact[fact_id]:
                    self._record_closure(
                        facts[fact_id], commit, commit.branch, "backfill", parents
                    )
                    closures_by_fact[fact_id].add(sha)
                closed.add((fact_id, sha))

        self.main_head = main_head
        self.advance_main(main_head)
        closed_shas = {sha for _, sha in closed}
        return [sha for sha in ordered_commits if sha in closed_shas]

    def _close_facts_for_commit(
        self,
        sha: str,
        branch: str,
        by: str,
        fact_ids: set[str] | None = None,
    ) -> list[str]:
        commit = self.dag.commits[sha]
        changed_by_parent = {
            parent: set(commit.symbols_vs(parent)) for parent in commit.parents
        }
        changed_symbols = set().union(*changed_by_parent.values())
        if not changed_symbols:
            return []

        facts = self.store.facts_for_symbols(sorted(changed_symbols))
        facts_by_id = {fact["_id"]: fact for fact in facts}
        if fact_ids is not None:
            facts_by_id = {
                fact_id: fact
                for fact_id, fact in facts_by_id.items()
                if fact_id in fact_ids
            }
        if not facts_by_id:
            return []

        closure_rows = self.store.closures_for_facts(list(facts_by_id))
        closures_by_fact: dict[str, set[str]] = defaultdict(set)
        for closure in closure_rows:
            closures_by_fact[closure["fact_id"]].add(closure["closing_commit"])

        parents_by_fact: dict[str, set[str]] = defaultdict(set)
        for parent, symbols in changed_by_parent.items():
            if not symbols:
                continue
            ancestors = self.dag.ancestors(parent)
            for fact_id, fact in facts_by_id.items():
                if fact["valid_from_commit"] not in ancestors:
                    continue
                if any(
                    closing_commit in ancestors
                    for closing_commit in closures_by_fact[fact_id]
                ):
                    continue
                if any(
                    anchor.get("symbol") in symbols
                    for anchor in fact.get("anchors") or ()
                ):
                    parents_by_fact[fact_id].add(parent)

        closed: list[str] = []
        for fact_id, parents in sorted(parents_by_fact.items()):
            if sha in closures_by_fact[fact_id]:
                closed.append(fact_id)
                continue
            self._record_closure(facts_by_id[fact_id], commit, branch, by, parents)
            closed.append(fact_id)
        return closed

    def _record_closure(
        self, fact: dict, commit, branch: str, by: str, parents: set[str]
    ) -> None:
        reason = "anchored symbol changed (vs parent " + ",".join(sorted(parents)) + ")"
        self.store.record_fact_closure(
            fact_id=fact["_id"],
            closing_commit=commit.sha,
            branch=branch,
            closed_by_agent=by,
            reason=reason,
            committed_at=commit.committed_at,
            evidence_test=fact.get("evidence_test"),
        )
