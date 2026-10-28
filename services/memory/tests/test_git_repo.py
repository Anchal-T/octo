import subprocess
from datetime import datetime

from services.memory.app.git_repo import GitRepo, Hunk


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repo(path):
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")
    return GitRepo(path)


def test_diff_u0_parses_replacements_additions_and_file_deletions(tmp_path):
    path = tmp_path / "repo"
    repo = _repo(path)
    (path / "module.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (path / "removed.py").write_text("gone\n", encoding="utf-8")
    parent = _commit(path, "parent")

    (path / "module.py").write_text("one\n++ body\n", encoding="utf-8")
    (path / "removed.py").unlink()
    (path / "new.py").write_text("added\n", encoding="utf-8")
    child = _commit(path, "child")

    assert repo.diff_u0(parent, child) == [
        Hunk("module.py", old_start=2, old_len=2, new_start=2, new_len=1),
        Hunk("new.py", old_start=0, old_len=0, new_start=1, new_len=1),
        Hunk("removed.py", old_start=1, old_len=1, new_start=0, new_len=0),
    ]
    assert repo.changed_py(parent, child)


def test_diff_hunk_parser_ignores_header_like_changed_source(tmp_path):
    path = tmp_path / "repo"
    repo = _repo(path)
    (path / "module.py").write_text(
        "zero\none\ntwo\nthree\nfour\nfive\n", encoding="utf-8"
    )
    parent = _commit(path, "parent")

    (path / "module.py").write_text(
        "zero\n++ body\ntwo\nthree\nchanged\nfive\n", encoding="utf-8"
    )
    child = _commit(path, "child")

    assert repo.diff_u0(parent, child) == [
        Hunk("module.py", old_start=2, old_len=1, new_start=2, new_len=1),
        Hunk("module.py", old_start=5, old_len=1, new_start=5, new_len=1),
    ]


def test_rev_list_parents_and_commit_metadata(tmp_path):
    path = tmp_path / "repo"
    repo = _repo(path)
    (path / "module.py").write_text("before\n", encoding="utf-8")
    parent = _commit(path, "parent")
    (path / "module.py").write_text("after\n", encoding="utf-8")
    child = _commit(path, "child")

    assert repo.rev_list_parents(child) == {
        parent: (),
        child: (parent,),
    }
    assert repo.committer_date(child) == datetime.fromisoformat(
        _git(path, "show", "-s", "--format=%cI", child)
    )
    assert repo.author(child) == "Test"


def test_changed_py_ignores_non_python_files(tmp_path):
    path = tmp_path / "repo"
    repo = _repo(path)
    (path / "README.md").write_text("before\n", encoding="utf-8")
    parent = _commit(path, "parent")

    (path / "README.md").write_text("after\n", encoding="utf-8")
    child = _commit(path, "child")

    assert not repo.changed_py(parent, child)
