from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import PurePosixPath


def areas_for_diff(
    changed_paths: Iterable[str], repo_areas: Mapping[str, Iterable[str]]
) -> set[str]:
    paths = tuple(PurePosixPath(path).as_posix() for path in changed_paths)
    return {
        area
        for area, path_globs in repo_areas.items()
        if any(fnmatchcase(path, pattern) for path in paths for pattern in path_globs)
    }
