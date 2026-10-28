import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from shutil import copyfileobj

from .scip import ScipIndex


class ScipIndexer:
    def __init__(
        self,
        repo_path: str | Path,
        cache_dir: str | Path,
        project_name: str,
        scip_python_command: Sequence[str] = ("scip-python",),
        scip_command: Sequence[str] = ("scip",),
    ) -> None:
        if not project_name:
            raise ValueError("project_name is required")
        if not scip_python_command or not scip_command:
            raise ValueError("SCIP commands cannot be empty")
        self.repo_path = Path(repo_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.project_name = project_name
        self.scip_python_command = tuple(scip_python_command)
        self.scip_command = tuple(scip_command)

    def index(self, sha: str) -> Path:
        resolved_sha = self._resolve_sha(sha)
        cached = self.cache_dir / resolved_sha / "index.json"
        if cached.is_file():
            return cached

        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agentteam-scip-") as temporary:
            worktree = Path(temporary) / "repo"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), resolved_sha],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                index_file = worktree / "index.scip"
                subprocess.run(
                    [
                        *self.scip_python_command,
                        "index",
                        "--cwd",
                        str(worktree),
                        "--project-name",
                        self.project_name,
                        "--project-version",
                        "0",
                        "--output",
                        str(index_file),
                        "--quiet",
                    ],
                    cwd=worktree,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                fd, temporary_json = tempfile.mkstemp(dir=cached.parent, suffix=".json")
                try:
                    with os.fdopen(fd, "wb") as output:
                        subprocess.run(
                            [
                                *self.scip_command,
                                "print",
                                "--json",
                                str(index_file),
                            ],
                            cwd=worktree,
                            check=True,
                            stdout=output,
                            stderr=subprocess.PIPE,
                        )
                    os.replace(temporary_json, cached)
                finally:
                    if os.path.exists(temporary_json):
                        os.unlink(temporary_json)
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
        return cached

    def load(self, sha: str) -> ScipIndex:
        path = self.index(sha)
        return ScipIndex.load(path, sha=path.parent.name)

    def reuse(self, source_sha: str, target_sha: str) -> ScipIndex:
        source = self.index(source_sha)
        target = self.cache_dir / self._resolve_sha(target_sha) / "index.json"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".json")
            try:
                with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
                    copyfileobj(input_file, output)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ScipIndex.load(target, sha=target.parent.name)

    def committer_date(self, sha: str) -> datetime:
        resolved_sha = self._resolve_sha(sha)
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", resolved_sha],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return datetime.fromisoformat(result.stdout.strip())

    def _resolve_sha(self, sha: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", sha):
            raise ValueError("base commit must be a hexadecimal Git SHA")
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{sha}^{{commit}}"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().lower()
