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


class GitRepo:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def rev_list_parents(self, rev: str = "--all") -> dict[str, tuple[str, ...]]:
        result = subprocess.run(
            ["git", "rev-list", "--parents", "--topo-order", "--reverse", rev],
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
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", sha],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return datetime.fromisoformat(result.stdout.strip())

    def author(self, sha: str) -> str:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%an", sha],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def diff_u0(self, a: str, b: str) -> list[Hunk]:
        result = subprocess.run(
            [
                "git",
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
        return self._parse_hunks(result.stdout)

    def changed_py(self, a: str, b: str) -> bool:
        result = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", a, b, "--"],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return any(path.endswith(".py") for path in result.stdout.split("\0"))

    @staticmethod
    def _parse_hunks(diff: str) -> list[Hunk]:
        hunks: list[Hunk] = []
        old_file: str | None = None
        new_file: str | None = None
        in_hunk = False
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                old_file = new_file = None
                in_hunk = False
                continue

            match = _HUNK_HEADER.match(line)
            if match is not None:
                in_hunk = True
                file = new_file if new_file != "/dev/null" else old_file
                if file is None or file == "/dev/null":
                    continue
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
            elif not in_hunk and line.startswith("--- "):
                old_file = line[4:]
            elif not in_hunk and line.startswith("+++ "):
                new_file = line[4:]
        return hunks
