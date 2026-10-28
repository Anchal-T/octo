import subprocess
import sys
from datetime import datetime

import pytest

from services.memory.app.scip_indexer import ScipIndexer


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _indexer(tmp_path, scip_script):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "module.py").write_text("def example():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "--quiet", "-m", "initial")
    sha = _git(repo, "rev-parse", "HEAD")

    calls = tmp_path / "calls.txt"
    indexer_script = tmp_path / "fake_scip_python.py"
    indexer_script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(calls)!r}).open('a').write('index\\n')\n"
        "args = sys.argv\n"
        "assert args[1] == 'index'\n"
        "assert args[args.index('--project-version') + 1] == '0'\n"
        "Path(args[args.index('--output') + 1]).write_bytes(b'fake-index')\n",
        encoding="utf-8",
    )
    printer_script = tmp_path / "fake_scip.py"
    printer_script.write_text(scip_script, encoding="utf-8")
    indexer = ScipIndexer(
        repo_path=repo,
        cache_dir=tmp_path / "cache",
        project_name="demo",
        scip_python_command=(sys.executable, str(indexer_script)),
        scip_command=(sys.executable, str(printer_script)),
    )
    return indexer, repo, sha, calls


def test_indexes_a_commit_once_and_loads_the_cached_definition(tmp_path):
    symbol = "scip-python python demo 0 `module`/example()."
    printer = (
        "import json\n"
        "print(json.dumps({'documents': [{'relative_path': 'module.py', 'occurrences': ["
        f"{{'range': [0, 4, 11], 'symbol': {symbol!r}, 'symbol_roles': 1}}"
        "]}]}))\n"
    )
    indexer, repo, sha, calls = _indexer(tmp_path, printer)

    first = indexer.load(sha)
    second = indexer.load(sha)

    assert first.sha == sha
    assert second.definitions_for(symbol) == first.definitions_for(symbol)
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    worktrees = [
        line.removeprefix("worktree ")
        for line in _git(repo, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    assert worktrees == [str(repo.resolve())]


def test_reuses_a_cached_index_for_a_commit_without_python_changes(tmp_path):
    symbol = "scip-python python demo 0 `module`/example()."
    printer = (
        "import json\n"
        "print(json.dumps({'documents': [{'relative_path': 'module.py', 'occurrences': ["
        f"{{'range': [0, 4, 11], 'symbol': {symbol!r}, 'symbol_roles': 1}}"
        "]}]}))\n"
    )
    indexer, repo, parent, calls = _indexer(tmp_path, printer)
    parent_index = indexer.load(parent)
    (repo / "README.md").write_text("docs\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "docs")
    child = _git(repo, "rev-parse", "HEAD")

    reused = indexer.reuse(parent, child)

    assert reused.sha == child
    assert reused.definitions_for(symbol) == parent_index.definitions_for(symbol)
    assert indexer.load(child) == reused
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_returns_git_committer_date_for_a_commit(tmp_path):
    indexer, repo, sha, _ = _indexer(tmp_path, "print('{}')\n")

    assert indexer.committer_date(sha) == datetime.fromisoformat(
        _git(repo, "show", "-s", "--format=%cI", sha)
    )


def test_removes_temporary_worktree_when_scip_print_fails(tmp_path):
    indexer, repo, sha, _ = _indexer(tmp_path, "import sys\nsys.exit(1)\n")

    with pytest.raises(subprocess.CalledProcessError):
        indexer.index(sha)

    worktrees = [
        line.removeprefix("worktree ")
        for line in _git(repo, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    assert worktrees == [str(repo.resolve())]
