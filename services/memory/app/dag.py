from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from heapq import heapify, heappop, heappush


class UnknownCommit(KeyError):
    pass


@dataclass(frozen=True)
class Commit:
    sha: str
    parents: tuple[str, ...]
    committed_at: datetime
    branch: str
    author: str | None
    changed_symbols: tuple[str, ...]
    merge_side_symbols: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def symbols_vs(self, parent: str) -> tuple[str, ...]:
        if not self.parents or parent not in self.parents:
            raise KeyError(parent)
        if parent == self.parents[0]:
            return self.changed_symbols
        try:
            return dict(self.merge_side_symbols)[parent]
        except KeyError:
            raise ValueError(
                f"missing merge-side symbols for parent {parent}"
            ) from None


class Dag:
    def __init__(self, commits: Iterable[Commit] = ()) -> None:
        self.commits: dict[str, Commit] = {}
        self.children: dict[str, set[str]] = {}
        self._ancestors_cache: dict[str, frozenset[str]] = {}
        self._descendants_cache: dict[str, frozenset[str]] = {}
        for commit in commits:
            self.add(commit)

    @classmethod
    def from_commits(cls, commits: Iterable[Commit]) -> "Dag":
        by_sha: dict[str, Commit] = {}
        for commit in commits:
            previous = by_sha.get(commit.sha)
            if previous is not None and previous != commit:
                raise ValueError(
                    f"commit {commit.sha} was recorded with different data"
                )
            by_sha[commit.sha] = commit

        missing = {
            parent
            for commit in by_sha.values()
            for parent in commit.parents
            if parent not in by_sha
        }
        if missing:
            raise KeyError(min(missing))

        children: dict[str, set[str]] = {sha: set() for sha in by_sha}
        indegree = {sha: len(commit.parents) for sha, commit in by_sha.items()}
        for commit in by_sha.values():
            for parent in commit.parents:
                children[parent].add(commit.sha)
        ready = [sha for sha, degree in indegree.items() if degree == 0]
        heapify(ready)
        dag = cls()
        while ready:
            sha = heappop(ready)
            dag.add(by_sha[sha])
            for child in children[sha]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heappush(ready, child)
        if len(dag.commits) != len(by_sha):
            raise ValueError("commit graph contains a cycle")
        return dag

    def add(self, commit: Commit) -> None:
        existing = self.commits.get(commit.sha)
        if existing is not None:
            if existing != commit:
                raise ValueError(
                    f"commit {commit.sha} was recorded with different data"
                )
            return
        for parent in commit.parents:
            if parent not in self.commits:
                raise KeyError(parent)

        self.commits[commit.sha] = commit
        self.children.setdefault(commit.sha, set())
        for parent in commit.parents:
            self.children[parent].add(commit.sha)
        self._ancestors_cache.clear()
        self._descendants_cache.clear()

    def ancestors(self, base: str) -> frozenset[str]:
        if base not in self.commits:
            raise KeyError(base)
        if base not in self._ancestors_cache:
            found = set()
            pending = [base]
            while pending:
                sha = pending.pop()
                if sha in found:
                    continue
                found.add(sha)
                pending.extend(self.commits[sha].parents)
            self._ancestors_cache[base] = frozenset(found)
        return self._ancestors_cache[base]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor in self.ancestors(descendant)

    def descendants(self, sha: str) -> frozenset[str]:
        if sha not in self.commits:
            raise KeyError(sha)
        if sha not in self._descendants_cache:
            found = set()
            pending = list(self.children[sha])
            while pending:
                child = pending.pop()
                if child in found:
                    continue
                found.add(child)
                pending.extend(self.children[child])
            self._descendants_cache[sha] = frozenset(found)
        return self._descendants_cache[sha]

    def first_parent_path(self, head: str) -> list[str]:
        if head not in self.commits:
            raise KeyError(head)
        path = []
        sha = head
        while True:
            path.append(sha)
            parents = self.commits[sha].parents
            if not parents:
                break
            sha = parents[0]
        path.reverse()
        return path

    def topological_order(self, shas: Iterable[str] | None = None) -> list[str]:
        pending = set(self.commits if shas is None else shas)
        missing = pending - self.commits.keys()
        if missing:
            raise KeyError(min(missing))
        indegree = {
            sha: sum(parent in pending for parent in self.commits[sha].parents)
            for sha in pending
        }
        ready = [sha for sha, degree in indegree.items() if degree == 0]
        heapify(ready)
        ordered = []
        while ready:
            parent = heappop(ready)
            ordered.append(parent)
            for child in self.children[parent] & pending:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heappush(ready, child)
        if len(ordered) != len(pending):
            raise ValueError("commit graph contains a cycle")
        return ordered

    def topological_descendants(self, sha: str) -> list[str]:
        return self.topological_order(self.descendants(sha))
