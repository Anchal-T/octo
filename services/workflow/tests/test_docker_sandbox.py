import subprocess
from pathlib import Path

import pytest
from agentteam_common import is_test_path

from services.workflow.app.sandbox import (
    DockerSandbox,
    LocalGitSandbox,
    SandboxLost,
    trim_output,
)


def _init_sample_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pkg").mkdir()
    (path / "tests").mkdir()
    (path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (path / "pkg" / "calc.py").write_text(
        "def divide(a: int, b: int) -> float:\n    return a / b\n",
        encoding="utf-8",
    )
    (path / "tests" / "test_calc.py").write_text(
        "from pkg.calc import divide\n\n\ndef test_divide_basic():\n    assert divide(6, 2) == 3.0\n",
        encoding="utf-8",
    )
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, env=env, check=True)
    subprocess.run(["git", "add", "."], cwd=path, env=env, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=path,
        env=env,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


def test_trim_output_preserves_head_and_tail_when_over_limit():
    raw = "HEAD:" + ("x" * 500) + ":TAIL"
    trimmed = trim_output(raw, max_chars=120)
    assert trimmed.startswith("HEAD:")
    assert trimmed.endswith(":TAIL")
    assert "trimmed" in trimmed
    assert len(trimmed) <= 160


def test_docker_sandbox_runs_isolated_container_handles_timeout_and_rebuilds(tmp_path):
    repo = tmp_path / "repo"
    sha = _init_sample_repo(repo)
    recorded: list[tuple[list[str], str | None]] = []
    state = {"container_num": 0, "lost": False, "diff": ""}

    def fake_runner(argv, *, input_text=None, timeout_s=None):
        recorded.append((list(argv), input_text))
        if argv[:2] == ["docker", "create"]:
            state["container_num"] += 1
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"cid-{state['container_num']}\n", stderr=""
            )
        if argv[:2] in (["docker", "start"], ["docker", "cp"], ["docker", "rm"]):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[:2] == ["docker", "exec"]:
            if state["lost"]:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="",
                    stderr="Error response from daemon: No such container: cid-1",
                )
            if "sleep 999" in " ".join(argv):
                raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout_s or 1.0)
            if "git apply" in " ".join(argv):
                state["diff"] = input_text or ""
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if "diff" in argv and "--name-only" not in argv:
                return subprocess.CompletedProcess(
                    argv, 0, stdout=state["diff"], stderr=""
                )
            return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")
        raise AssertionError(f"unexpected command: {argv}")

    sb = DockerSandbox(
        image="python:3.12-slim",
        repo_path=repo,
        base_commit=sha,
        network="none",
        runner=fake_runner,
    )
    sb.start()
    create_cmd = recorded[0][0]
    assert "--network" in create_cmd
    assert create_cmd[create_cmd.index("--network") + 1] == "none"

    obs = sb.exec("echo hi", timeout_s=10)
    assert obs.exit_code == 0
    assert obs.timed_out is False
    assert "ok" in obs.stdout_trimmed

    # Timeouts return Observation instead of raising.
    timed_out = sb.exec("sleep 999", timeout_s=0.5)
    assert timed_out.timed_out is True
    assert timed_out.exit_code == 124

    # Vanished container raises SandboxLost; rebuild restores image + last_diff.
    state["lost"] = True
    with pytest.raises(SandboxLost):
        sb.exec("pytest")

    state["lost"] = False
    sample_diff = "diff --git a/pkg/calc.py b/pkg/calc.py\n"
    sb.rebuild(sample_diff)
    assert sb.container_id == "cid-2"
    assert state["diff"] == sample_diff
    sb.close()
    assert sb.container_id is None


def test_docker_sandbox_rejects_unsafe_base_commit_and_pytest_flag_injection(tmp_path):
    repo = tmp_path / "repo"
    sha = _init_sample_repo(repo)

    with pytest.raises(ValueError, match="commit SHA"):
        DockerSandbox(
            image="python:3.12-slim",
            repo_path=repo,
            base_commit="--evil-flag",
        )

    with (
        LocalGitSandbox(repo_path=repo, base_commit=sha, work_root=tmp_path) as sb,
        pytest.raises(ValueError, match="test node"),
    ):
        sb.pytest(["--rootdir=/etc"])


def test_real_git_and_pytest_flow_captures_untracked_tests_and_enforces_role_scopes(
    tmp_path,
):
    repo = tmp_path / "repo"
    sha = _init_sample_repo(repo)

    # 1. Reproducer creates a brand new untracked test file AND accidentally touches source.
    with LocalGitSandbox(
        repo_path=repo, base_commit=sha, work_root=tmp_path / "repro"
    ) as repro_sb:
        repro_sb.exec(
            "cat <<'EOF' > tests/test_zero.py\n"
            "import pytest\n"
            "from pkg.calc import divide\n\n"
            "def test_divide_by_zero_returns_none():\n"
            "    assert divide(1, 0) is None\n"
            "EOF\n"
            "echo '# accidental edit' >> pkg/calc.py"
        )
        reverted = repro_sb.enforce_allowed_changes(is_test_path)
        assert reverted == ("pkg/calc.py",)
        assert repro_sb.changed_files() == ("tests/test_zero.py",)

        repro_diff = repro_sb.git_diff()
        assert "tests/test_zero.py" in repro_diff
        assert "pkg/calc.py" not in repro_diff

        repro_run = repro_sb.pytest(
            ["tests/test_zero.py::test_divide_by_zero_returns_none"]
        )
        assert repro_run.passed is False
        assert repro_run.exit_code == 1

    # 2. Fixer starts from base + repro_diff, tries to tamper with tests/test_zero.py, and fixes pkg/calc.py.
    with LocalGitSandbox(
        repo_path=repo, base_commit=sha, work_root=tmp_path / "fix"
    ) as fix_sb:
        fix_sb.apply(repro_diff)
        fix_sb.exec(
            "cat <<'EOF' > pkg/calc.py\n"
            "def divide(a: int, b: int) -> float | None:\n"
            "    if b == 0:\n"
            "        return None\n"
            "    return a / b\n"
            "EOF\n"
            "echo 'def test_divide_by_zero_returns_none(): pass' > tests/test_zero.py"
        )
        tampered = fix_sb.enforce_allowed_changes(
            lambda path: not is_test_path(path),
            base_diff=repro_diff,
        )
        assert "tests/test_zero.py" in tampered
        fix_files = [f for f in fix_sb.changed_files() if not is_test_path(f)]
        assert fix_files == ["pkg/calc.py"]
        fix_diff = fix_sb.git_diff(paths=fix_files)

        after_fix = fix_sb.pytest(
            ["tests/test_zero.py::test_divide_by_zero_returns_none"]
        )
        assert after_fix.passed is True
        assert after_fix.exit_code == 0

    # 3. Clean checkout verification applies repro_diff (fails) then fix_diff (passes all).
    with LocalGitSandbox(
        repo_path=repo, base_commit=sha, work_root=tmp_path / "clean"
    ) as clean_sb:
        clean_sb.apply(repro_diff)
        before = clean_sb.pytest(
            ["tests/test_zero.py::test_divide_by_zero_returns_none"]
        )
        assert before.exit_code == 1

        clean_sb.apply(fix_diff)
        after = clean_sb.pytest(
            ["tests/test_zero.py::test_divide_by_zero_returns_none"]
        )
        suite = clean_sb.pytest()
        assert after.passed is True
        assert suite.passed is True

        contents = clean_sb.read_files(["pkg/calc.py", "tests/test_zero.py"])
        assert "if b == 0:" in (contents["pkg/calc.py"] or "")
        assert "assert divide(1, 0) is None" in (contents["tests/test_zero.py"] or "")
