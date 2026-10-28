import subprocess
from dataclasses import replace

from services.memory.app.git_repo import GitRepo
from services.memory.app.repo import Repo
from services.memory.app.scip import ScipDefinition, ScipIndex


class FakeStore:
    def __init__(self, facts):
        self.commits = {}
        self.facts = facts
        self.closures = []
        self.ended = []

    def get_commits(self):
        return list(self.commits.values())

    def record_commit(self, commit):
        self.commits[commit.sha] = commit

    def facts_for_symbols(self, symbols):
        return [
            fact
            for fact in self.facts
            if any(anchor["symbol"] in symbols for anchor in fact["anchors"])
        ]

    def closures_for_facts(self, fact_ids):
        return [closure for closure in self.closures if closure["fact_id"] in fact_ids]

    def record_fact_closure(self, **closure):
        key = (closure["fact_id"], closure["closing_commit"])
        if not any(
            (row["fact_id"], row["closing_commit"]) == key for row in self.closures
        ):
            self.closures.append(closure)
        return f"{key[0]}@{key[1]}"

    def all_fact_closures(self):
        return self.closures

    def end_fact_valid_time(self, fact_ids, committed_at):
        self.ended.append((tuple(fact_ids), committed_at))


class FakeIndexer:
    def __init__(self, indexes):
        self.indexes = indexes
        self.calls = []
        self.reuses = []

    def load(self, sha):
        self.calls.append(sha)
        return self.indexes[sha]

    def reuse(self, source_sha, target_sha):
        self.reuses.append((source_sha, target_sha))
        return replace(self.indexes[source_sha], sha=target_sha)


def _git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _definition(symbol, relative_path, start, end):
    return ScipDefinition(
        symbol=symbol,
        relative_path=relative_path,
        range=(start, 0, start, 5),
        enclosing_range=(start, 0, end, 0),
    )


def _index(sha, f_symbol, g_symbol):
    return ScipIndex(
        {
            f_symbol: (_definition(f_symbol, "module.py", 0, 2),),
            g_symbol: (_definition(g_symbol, "module.py", 2, 4),),
        },
        sha=sha,
    )


def test_on_push_records_parent_specific_symbols_and_closes_facts(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "branch", "-M", "main")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")
    (path / "module.py").write_text(
        "def f():\n    return 1\ndef g():\n    return 1\n", encoding="utf-8"
    )

    _git(path, "add", "module.py")
    _git(path, "commit", "--quiet", "-m", "root")
    root = _git(path, "rev-parse", "HEAD")

    (path / "module.py").write_text(
        "def f():\n    return 2\ndef g():\n    return 1\n", encoding="utf-8"
    )
    _git(path, "commit", "--quiet", "-am", "main change")
    main = _git(path, "rev-parse", "HEAD")
    (path / "README.md").write_text("docs\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "--quiet", "-m", "docs")
    docs = _git(path, "rev-parse", "HEAD")

    _git(path, "switch", "--quiet", "-c", "feature", root)
    (path / "module.py").write_text(
        "def f():\n    return 1\ndef g():\n    return 2\n", encoding="utf-8"
    )
    _git(path, "commit", "--quiet", "-am", "feature change")
    side = _git(path, "rev-parse", "HEAD")

    _git(path, "switch", "--quiet", "main")
    _git(path, "merge", "--quiet", "--no-ff", "feature", "-m", "merge")
    merge = _git(path, "rev-parse", "HEAD")

    f_symbol, g_symbol = "module.f().", "module.g()."
    indexer = FakeIndexer(
        {sha: _index(sha, f_symbol, g_symbol) for sha in (root, main, side, merge)}
    )

    store = FakeStore(
        [
            {
                "_id": "fact-f",
                "valid_from_commit": root,
                "anchors": [{"symbol": f_symbol, "sha": root}],
                "evidence_test": None,
            }
        ]
    )
    shas = list(GitRepo(path).rev_list_parents(merge))

    processor = Repo(store, GitRepo(path), indexer)
    processor.replay(shas, main_head=merge)
    processor.on_push(shas, branch="main")

    assert store.commits[side].branch == "history"
    assert store.commits[main].changed_symbols == (f_symbol,)
    assert store.commits[docs].changed_symbols == ()
    assert store.commits[side].changed_symbols == (g_symbol,)
    assert store.commits[merge].changed_symbols == (g_symbol,)
    assert dict(store.commits[merge].merge_side_symbols)[side] == (f_symbol,)
    assert {(row["fact_id"], row["closing_commit"]) for row in store.closures} == {
        ("fact-f", main),
        ("fact-f", merge),
    }
    assert set(store.ended) == {(("fact-f",), store.commits[main].committed_at)}
    assert indexer.calls == [root, main, side, merge]
    assert docs not in indexer.calls
    assert (main, docs) in indexer.reuses
