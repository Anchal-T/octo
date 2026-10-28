from collections.abc import Iterable

from .dag import Commit, Dag
from .fact_closer import FactCloser
from .git_repo import GitRepo
from .scip import ScipIndex
from .scip_indexer import ScipIndexer
from .store import MemoryStore
from .symbol_differ import SymbolDiffer


class Repo:
    def __init__(
        self, store: MemoryStore, git_repo: GitRepo, scip_indexer: ScipIndexer
    ) -> None:
        self.store = store
        self.git_repo = git_repo
        self.scip_indexer = scip_indexer

    def on_push(self, shas: Iterable[str], branch: str) -> list[str]:
        shas = list(dict.fromkeys(shas))
        if not shas:
            return []
        git_commits = self.git_repo.rev_list_parents(shas[-1])
        branches = {sha: branch for sha in shas}
        main_head = shas[-1] if branch == "main" else None
        return self._process(shas, branches, git_commits, main_head)

    def replay(self, shas_topo: Iterable[str], main_head: str) -> list[str]:
        shas = list(dict.fromkeys(shas_topo))
        if not shas:
            return []
        git_commits = self.git_repo.rev_list_parents("--all")
        if main_head not in git_commits:
            raise KeyError(main_head)

        main_path = set()
        sha = main_head
        while True:
            main_path.add(sha)
            parents = git_commits[sha]
            if not parents:
                break
            sha = parents[0]
        branches = {sha: "main" if sha in main_path else "history" for sha in shas}
        return self._process(shas, branches, git_commits, main_head)

    def advance_main(self, head: str) -> list[str]:
        dag = Dag.from_commits(self.store.get_commits())
        return FactCloser(dag, self.store).advance_main(head)

    def _process(
        self,
        shas: list[str],
        branches: dict[str, str],
        git_commits: dict[str, tuple[str, ...]],
        main_head: str | None,
    ) -> list[str]:
        positions = {sha: index for index, sha in enumerate(shas)}
        for sha, index in positions.items():
            if sha not in git_commits:
                raise KeyError(sha)
            for parent in git_commits[sha]:
                if parent in positions and positions[parent] >= index:
                    raise ValueError("commits must be in topological order")

        dag = Dag.from_commits(self.store.get_commits())
        for sha in shas:
            for parent in git_commits[sha]:
                if parent not in dag.commits and parent not in positions:
                    raise KeyError(parent)

        indexes: dict[str, ScipIndex] = {}
        closer = FactCloser(dag, self.store)
        for sha in shas:
            parents = git_commits[sha]
            commit = dag.commits.get(sha)
            if commit is None:
                child_index, changed_symbols, merge_side_symbols = self._diff(
                    sha, parents, indexes
                )
                commit = Commit(
                    sha=sha,
                    parents=parents,
                    committed_at=self.git_repo.committer_date(sha),
                    branch=branches[sha],
                    author=self.git_repo.author(sha),
                    changed_symbols=tuple(sorted(changed_symbols)),
                    merge_side_symbols=tuple(
                        (parent, tuple(sorted(symbols)))
                        for parent, symbols in merge_side_symbols
                    ),
                )
                self.store.record_commit(commit)
                dag.add(commit)
                indexes[sha] = child_index

            closer.close_facts_for_commit(sha, commit.branch)

        if main_head is not None:
            closer.advance_main(main_head)
        return shas

    def _diff(
        self, sha: str, parents: tuple[str, ...], indexes: dict[str, ScipIndex]
    ) -> tuple[ScipIndex, set[str], list[tuple[str, set[str]]]]:
        if not parents:
            return self._index_at(sha, indexes), set(), []

        first_parent = parents[0]
        parent_index = self._index_at(first_parent, indexes)
        if self.git_repo.changed_py(first_parent, sha):
            child_index = self.scip_indexer.load(sha)
            changed_symbols = SymbolDiffer.diff(
                parent_index,
                child_index,
                self.git_repo.diff_u0(first_parent, sha),
            )
        else:
            child_index = self.scip_indexer.reuse(first_parent, sha)
            changed_symbols = set()

        merge_side_symbols = []
        for parent in parents[1:]:
            if self.git_repo.changed_py(parent, sha):
                side_index = self._index_at(parent, indexes)
                symbols = SymbolDiffer.diff(
                    side_index, child_index, self.git_repo.diff_u0(parent, sha)
                )
            else:
                symbols = set()
            merge_side_symbols.append((parent, symbols))

        return child_index, changed_symbols, merge_side_symbols

    def _index_at(self, sha: str, indexes: dict[str, ScipIndex]) -> ScipIndex:
        if sha not in indexes:
            indexes[sha] = self.scip_indexer.load(sha)
        return indexes[sha]
