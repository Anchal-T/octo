from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: Any, deep: bool = False
) -> dict[Any, Any]:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("configuration mapping keys must be scalar") from error
        if duplicate:
            raise ValueError(f"duplicate configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class RepoConfig:
    name: str
    areas: dict[str, tuple[str, ...]]

    @property
    def allowed_area_names(self) -> tuple[str, ...]:
        return (*self.areas, "unknown")


@dataclass(frozen=True)
class AgentTeamConfig:
    repos: dict[str, RepoConfig]

    def repo(self, name: str) -> RepoConfig:
        try:
            return self.repos[name]
        except KeyError as error:
            raise ValueError(f"unknown repository: {name}") from error


def load_agentteam_config(path: str | Path = "agentteam.yaml") -> AgentTeamConfig:
    path = Path(path)
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}") from error
    if not isinstance(data, dict):
        raise TypeError("agentteam.yaml must contain a mapping")

    repo_entries = data.get("repos")
    if not isinstance(repo_entries, list) or not repo_entries:
        raise ValueError("agentteam.yaml repos must be a non-empty list")

    repos = {}
    for entry in repo_entries:
        if not isinstance(entry, dict):
            raise TypeError("each repository entry must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("repository name must be non-empty")
        name = name.strip()
        if name in repos:
            raise ValueError(f"duplicate repository name: {name}")

        area_entries = entry.get("areas")
        if not isinstance(area_entries, dict) or not area_entries:
            raise ValueError(f"repository {name} areas must be non-empty")
        areas = {}
        for area_name, path_globs in area_entries.items():
            if not isinstance(area_name, str) or not area_name.strip():
                raise ValueError(f"repository {name} has an empty area name")
            area_name = area_name.strip()
            if area_name == "unknown":
                raise ValueError("unknown is reserved and must not be configured")
            if area_name in areas:
                raise ValueError(f"duplicate area name: {area_name}")
            if not isinstance(path_globs, list) or not path_globs:
                raise ValueError(f"area {area_name} must have at least one path glob")
            if any(
                not isinstance(glob, str) or not glob.strip() for glob in path_globs
            ):
                raise ValueError(f"area {area_name} has an empty path glob")
            areas[area_name] = tuple(glob.strip() for glob in path_globs)
        repos[name] = RepoConfig(name=name, areas=areas)

    return AgentTeamConfig(repos=repos)
