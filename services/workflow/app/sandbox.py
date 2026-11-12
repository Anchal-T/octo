from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from agentteam_common import is_safe_relative_path

_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_LOST_CONTAINER_PATTERNS = (
    "no such container",
    "is not running",
    "cannot exec in a stopped state",
    "can not exec in a stopped state",
)

_GIT_EXCLUDE_PATTERNS = (
    "__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n"
    ".ruff_cache/\n"
    ".mypy_cache/\n"
    ".coverage\n"
    "*.egg-info/\n"
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class SandboxLost(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    stdout_trimmed: str
    exit_code: int
    git_diff: str
    timed_out: bool = False
    reverted_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestResult:
    passed: bool
    exit_code: int
    output_trimmed: str
    node_ids: tuple[str, ...] = ()


class Sandbox(Protocol):
    base_commit: str

    def start(self) -> None: ...
    def exec(self, cmd: str, timeout_s: float = 60.0) -> Observation: ...
    def git_diff(self, paths: Sequence[str] | None = None) -> str: ...
    def changed_files(self) -> tuple[str, ...]: ...
    def apply(self, diff: str) -> None: ...
    def restore_to(self, base_diff: str = "") -> None: ...
    def enforce_allowed_changes(
        self,
        keep_predicate: Callable[[str], bool],
        *,
        base_diff: str = "",
    ) -> tuple[str, ...]: ...
    def pytest(
        self,
        node_ids: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> TestResult: ...
    def read_files(self, paths: Sequence[str]) -> dict[str, str | None]: ...
    def rebuild(self, last_diff: str = "") -> None: ...
    def close(self) -> None: ...


def trim_output(text: str, max_chars: int = 2000) -> str:
    if max_chars < 40 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - (2 * half)
    return f"{text[:half]}\n[... trimmed {omitted} characters ...]\n{text[-half:]}"


def validate_commit_sha(sha: str) -> str:
    if not isinstance(sha, str) or _COMMIT_SHA_RE.fullmatch(sha) is None:
        raise ValueError("base_commit must be a hexadecimal commit SHA")
    return sha


def validate_test_node_id(node_id: str) -> str:
    if (
        not isinstance(node_id, str)
        or not node_id.strip()
        or node_id != node_id.strip()
        or node_id.startswith("-")
        or any(ch in node_id for ch in ("\x00", "\n", "\r"))
    ):
        raise ValueError("invalid pytest test node id")
    file_part = node_id.split("::", 1)[0]
    if not is_safe_relative_path(file_part):
        raise ValueError("pytest test node id must reference a safe relative file path")
    return node_id


def _default_runner(
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _split_diff_by_file(diff_text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current_path: str | None = None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            matched = _DIFF_FILE_RE.match(line.rstrip("\r\n"))
            current_path = matched.group(2) if matched else None
            if current_path is not None:
                blocks[current_path] = [line]
            continue
        if current_path is not None:
            blocks[current_path].append(line)
    return {path: "".join(lines) for path, lines in blocks.items()}


class _BaseGitSandbox:
    _python_bin: str = "python"

    def __init__(
        self,
        *,
        base_commit: str,
        max_output_chars: int = 2000,
    ) -> None:
        self.base_commit = validate_commit_sha(base_commit)
        self.max_output_chars = max_output_chars

    def _run_in_workspace(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_s: float | None = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def exec(self, cmd: str, timeout_s: float = 60.0) -> Observation:
        try:
            proc = self._run_in_workspace(["bash", "-lc", cmd], timeout_s=timeout_s)
        except subprocess.TimeoutExpired as timed_out:
            partial = (timed_out.stdout or "") + (timed_out.stderr or "")
            return Observation(
                stdout_trimmed=trim_output(
                    f"{partial}\n[command timed out after {timeout_s}s]".strip(),
                    self.max_output_chars,
                ),
                exit_code=124,
                git_diff=self.git_diff(),
                timed_out=True,
            )
        combined = proc.stdout
        if proc.stderr:
            combined = f"{combined}\n{proc.stderr}" if combined else proc.stderr
        return Observation(
            stdout_trimmed=trim_output(combined, self.max_output_chars),
            exit_code=proc.returncode,
            git_diff=self.git_diff(),
            timed_out=False,
        )

    def _configure_git_excludes(self) -> None:
        self._run_in_workspace(
            ["sh", "-c", "mkdir -p .git/info && cat >> .git/info/exclude"],
            input_text=_GIT_EXCLUDE_PATTERNS,
        )

    def _stage_intent_to_add(self) -> None:
        self._run_in_workspace(["git", "add", "-N", "."])

    def git_diff(self, paths: Sequence[str] | None = None) -> str:
        if paths is not None and not list(paths):
            return ""
        self._stage_intent_to_add()
        cmd = [
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            "--no-ext-diff",
            "--no-color",
            "HEAD",
            "--",
        ]
        if paths:
            for path in paths:
                if not is_safe_relative_path(path):
                    raise ValueError(f"unsafe relative path: {path}")
                cmd.append(path)
        proc = self._run_in_workspace(cmd)
        return proc.stdout

    def changed_files(self) -> tuple[str, ...]:
        self._stage_intent_to_add()
        proc = self._run_in_workspace(
            [
                "git",
                "-c",
                "core.quotePath=false",
                "diff",
                "--name-only",
                "-z",
                "HEAD",
                "--",
            ]
        )
        items = [item for item in proc.stdout.split("\x00") if item]
        return tuple(dict.fromkeys(items))

    def apply(self, diff: str) -> None:
        if not diff.strip():
            return
        proc = self._run_in_workspace(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input_text=diff if diff.endswith("\n") else f"{diff}\n",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git apply failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )

    def restore_to(self, base_diff: str = "") -> None:
        reset = self._run_in_workspace(["git", "reset", "--hard", self.base_commit])
        if reset.returncode != 0:
            raise RuntimeError(f"git reset failed: {reset.stderr.strip()}")
        clean = self._run_in_workspace(["git", "clean", "-fdx"])
        if clean.returncode != 0:
            raise RuntimeError(f"git clean failed: {clean.stderr.strip()}")
        if base_diff.strip():
            self.apply(base_diff)

    def enforce_allowed_changes(
        self,
        keep_predicate: Callable[[str], bool],
        *,
        base_diff: str = "",
    ) -> tuple[str, ...]:
        changed = self.changed_files()
        allowed_files = [path for path in changed if keep_predicate(path)]
        disallowed_files = [path for path in changed if not keep_predicate(path)]

        base_by_file = _split_diff_by_file(base_diff) if base_diff.strip() else {}
        current_disallowed_diff = (
            self.git_diff(paths=disallowed_files) if disallowed_files else ""
        )
        current_by_file = _split_diff_by_file(current_disallowed_diff)

        tampered = sorted(
            (set(current_by_file) ^ set(base_by_file))
            | {
                path
                for path, block in current_by_file.items()
                if block != base_by_file.get(path)
            }
        )
        if not tampered:
            return ()

        allowed_diff = self.git_diff(paths=allowed_files) if allowed_files else ""
        self.restore_to(base_diff)
        if allowed_diff.strip():
            self.apply(allowed_diff)
        return tuple(tampered)

    def pytest(
        self,
        node_ids: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> TestResult:
        validated = tuple(validate_test_node_id(node) for node in (node_ids or ()))
        cmd = [self._python_bin, "-m", "pytest", "-q"]
        if validated:
            cmd.extend(["--", *validated])
        try:
            proc = self._run_in_workspace(cmd, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as timed_out:
            partial = (timed_out.stdout or "") + (timed_out.stderr or "")
            return TestResult(
                passed=False,
                exit_code=124,
                output_trimmed=trim_output(
                    f"{partial}\n[pytest timed out after {timeout_s}s]".strip(),
                    self.max_output_chars,
                ),
                node_ids=validated,
            )
        combined = proc.stdout
        if proc.stderr:
            combined = f"{combined}\n{proc.stderr}" if combined else proc.stderr
        return TestResult(
            passed=proc.returncode == 0,
            exit_code=proc.returncode,
            output_trimmed=trim_output(combined, self.max_output_chars),
            node_ids=validated,
        )

    def read_files(self, paths: Sequence[str]) -> dict[str, str | None]:
        results: dict[str, str | None] = {}
        for path in paths:
            if not is_safe_relative_path(path):
                raise ValueError(f"unsafe relative path: {path}")
            check = self._run_in_workspace(["test", "-f", path])
            if check.returncode != 0:
                results[path] = None
                continue
            cat = self._run_in_workspace(["cat", "--", path])
            if cat.returncode != 0:
                raise RuntimeError(f"failed reading file {path}: {cat.stderr.strip()}")
            results[path] = cat.stdout
        return results

    def rebuild(self, last_diff: str = "") -> None:
        self.close()
        self.start()
        if last_diff.strip():
            self.apply(last_diff)


class DockerSandbox(_BaseGitSandbox):
    def __init__(
        self,
        *,
        image: str,
        repo_path: str | Path,
        base_commit: str,
        network: str = "none",
        max_output_chars: int = 2000,
        runner: CommandRunner = _default_runner,
    ) -> None:
        super().__init__(base_commit=base_commit, max_output_chars=max_output_chars)
        if not image.strip():
            raise ValueError("sandbox image must not be empty")
        self.image = image.strip()
        self.repo_path = Path(repo_path).resolve()
        self.network = network.strip() or "none"
        self.runner = runner
        self.container_id: str | None = None

    def start(self) -> None:
        created = self.runner(
            [
                "docker",
                "create",
                "--rm",
                "--network",
                self.network,
                "--memory",
                "1g",
                "--pids-limit",
                "256",
                "-w",
                "/workspace",
                self.image,
                "sleep",
                "infinity",
            ],
            timeout_s=60.0,
        )
        if created.returncode != 0 or not created.stdout.strip():
            raise RuntimeError(f"docker create failed: {created.stderr.strip()}")
        cid = created.stdout.strip().splitlines()[0]
        self.container_id = cid
        try:
            started = self.runner(["docker", "start", cid], timeout_s=30.0)
            if started.returncode != 0:
                raise RuntimeError(f"docker start failed: {started.stderr.strip()}")
            copied = self.runner(
                ["docker", "cp", f"{self.repo_path}/.", f"{cid}:/workspace"],
                timeout_s=120.0,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"docker cp failed: {copied.stderr.strip()}")
            checkout = self._run_in_workspace(
                ["git", "checkout", "--force", self.base_commit]
            )
            if checkout.returncode != 0:
                raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")
            self._run_in_workspace(["git", "clean", "-fdx"])
            self._configure_git_excludes()
        except Exception:
            self.close()
            raise

    def _run_in_workspace(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_s: float | None = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        if self.container_id is None:
            raise SandboxLost("sandbox container is not started")
        docker_cmd = ["docker", "exec"]
        if input_text is not None:
            docker_cmd.append("-i")
        docker_cmd.extend(["-w", "/workspace", self.container_id, *argv])
        proc = self.runner(
            docker_cmd,
            input_text=input_text,
            timeout_s=timeout_s,
        )
        stderr_lower = (proc.stderr or "").lower()
        if proc.returncode != 0 and any(
            marker in stderr_lower for marker in _LOST_CONTAINER_PATTERNS
        ):
            self.container_id = None
            raise SandboxLost(proc.stderr.strip() or "sandbox container was lost")
        return proc

    def close(self) -> None:
        if self.container_id is None:
            return
        cid = self.container_id
        self.container_id = None
        self.runner(["docker", "rm", "-f", cid], timeout_s=30.0)


class LocalGitSandbox(_BaseGitSandbox):
    _python_bin: str = sys.executable

    def __init__(
        self,
        *,
        repo_path: str | Path,
        base_commit: str,
        work_root: str | Path | None = None,
        max_output_chars: int = 2000,
    ) -> None:
        super().__init__(base_commit=base_commit, max_output_chars=max_output_chars)
        self.repo_path = Path(repo_path).resolve()
        self.work_root = Path(work_root) if work_root is not None else None
        self.workspace: Path | None = None

    def start(self) -> None:
        if self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(
                tempfile.mkdtemp(prefix="sandbox-", dir=str(self.work_root))
            )
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="agentteam-sandbox-"))
        self.workspace = temp_dir
        shutil.copytree(self.repo_path, temp_dir, dirs_exist_ok=True)
        checkout = self._run_in_workspace(
            ["git", "checkout", "--force", self.base_commit]
        )
        if checkout.returncode != 0:
            self.close()
            raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")
        self._run_in_workspace(["git", "clean", "-fdx"])
        self._configure_git_excludes()

    def _run_in_workspace(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_s: float | None = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        if self.workspace is None or not self.workspace.exists():
            raise SandboxLost("local sandbox workspace is missing")
        env = {
            **os.environ,
            "PYTHONPATH": str(self.workspace),
        }
        return subprocess.run(
            list(argv),
            cwd=self.workspace,
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

    def close(self) -> None:
        if self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
