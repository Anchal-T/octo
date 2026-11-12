from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import PurePosixPath


def is_safe_relative_path(path: str) -> bool:
    if not isinstance(path, str) or not path or path != path.strip():
        return False
    if path.startswith(("/", "\\", "~")) or any(
        ch in path for ch in ("\x00", "\n", "\r", "\\")
    ):
        return False
    raw_parts = path.split("/")
    if any(part in ("", ".", "..", ".git") for part in raw_parts):
        return False
    parsed = PurePosixPath(path)
    return not parsed.is_absolute() and ".." not in parsed.parts


_TRANSIENT_SEGMENTS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
)


def is_test_path(path: str) -> bool:
    if not is_safe_relative_path(path):
        return False
    parts = PurePosixPath(path).parts
    if any(part in _TRANSIENT_SEGMENTS or part.endswith(".egg-info") for part in parts):
        return False
    name = parts[-1]
    if name.endswith((".pyc", ".pyo", ".pyd")):
        return False
    if any(directory in ("tests", "test") for directory in parts[:-1]):
        return True
    if name == "conftest.py":
        return True
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def areas_for_diff(
    changed_paths: Iterable[str], repo_areas: Mapping[str, Iterable[str]]
) -> set[str]:
    paths = tuple(PurePosixPath(path).as_posix() for path in changed_paths)
    return {
        area
        for area, path_globs in repo_areas.items()
        if any(fnmatchcase(path, pattern) for path in paths for pattern in path_globs)
    }
