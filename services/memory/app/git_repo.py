import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Hunk:
    file: str
    old_start: int
    old_len: int
    new_start: int
    new_len: int


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{7,64}")


def _require_commit_sha(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise ValueError("revision must be a hexadecimal commit SHA")
    return value


class GitRepo:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def rev_list_parents(self, rev: str | None = None) -> dict[str, tuple[str, ...]]:
        command = ["git", "rev-list", "--parents", "--topo-order", "--reverse"]
        if rev is None:
            command.append("--all")
        else:
            command.extend(("--end-of-options", _require_commit_sha(rev)))
        result = subprocess.run(
            command,
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            fields[0]: tuple(fields[1:])
            for line in result.stdout.splitlines()
            if (fields := line.split())
        }

    def committer_date(self, sha: str) -> datetime:
        sha = _require_commit_sha(sha)
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", sha],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return datetime.fromisoformat(result.stdout.strip())

    def author(self, sha: str) -> str:
        sha = _require_commit_sha(sha)
        result = subprocess.run(
            ["git", "show", "-s", "--format=%an", sha],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def diff_u0(self, a: str, b: str) -> list[Hunk]:
        a = _require_commit_sha(a)
        b = _require_commit_sha(b)
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.quotePath=false",
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--no-prefix",
                "--no-renames",
                "--unified=0",
                a,
                b,
                "--",
            ],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", a, b, "--"],
            cwd=self.path,
            check=True,
            capture_output=True,
        ).stdout
        files = [os.fsdecode(path) for path in paths.split(b"\0") if path]
        return self._parse_hunks(result.stdout, files)

    def changed_py(self, a: str, b: str) -> bool:
        a = _require_commit_sha(a)
        b = _require_commit_sha(b)
        result = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", a, b, "--"],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return any(path.endswith(".py") for path in result.stdout.split("\0"))

    @staticmethod
    def _parse_hunks(diff: str, files: list[str]) -> list[Hunk]:
        hunks: list[Hunk] = []
        file_index = -1
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                file_index += 1
                continue

            match = _HUNK_HEADER.match(line)
            if match is not None:
                if file_index >= len(files):
                    continue
                file = files[file_index]
                old_start, old_len, new_start, new_len = match.groups()
                hunks.append(
                    Hunk(
                        file=file,
                        old_start=int(old_start),
                        old_len=int(old_len or 1),
                        new_start=int(new_start),
                        new_len=int(new_len or 1),
                    )
                )

        return hunks
