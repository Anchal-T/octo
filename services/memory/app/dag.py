from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime


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
