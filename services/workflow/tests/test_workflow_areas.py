import pytest
from agentteam_common.areas import areas_for_diff
from agentteam_common.repo_config import load_agentteam_config


def _write_config(tmp_path, text):
    path = tmp_path / "agentteam.yaml"
    path.write_text(text)
    return path


def test_areas_for_diff_matches_all_areas_touched_by_a_multi_area_diff():
    config = load_agentteam_config()
    repo_areas = config.repo("andialbrecht/sqlparse").areas

    assert areas_for_diff(
        [
            "sqlparse/engine/grouping.py",
            "sqlparse/formatter.py",
            "docs/usage.rst",
        ],
        repo_areas,
    ) == {"parser-tree", "formatting", "docs-build"}


def test_areas_for_diff_returns_no_areas_for_unmatched_paths():
    repo_areas = load_agentteam_config().repo("andialbrecht/sqlparse").areas

    assert areas_for_diff(["README.md"], repo_areas) == set()


def test_agentteam_config_loads_the_decided_area_names():
    config = load_agentteam_config()

    sqlparse = config.repo("andialbrecht/sqlparse")
    marshmallow = config.repo("marshmallow-code/marshmallow")

    assert set(sqlparse.areas) == {
        "lexer-keywords",
        "parser-tree",
        "splitting",
        "formatting",
        "cli-api",
        "docs-build",
    }
    assert set(marshmallow.areas) == {
        "fields",
        "schema",
        "validate",
        "hooks",
        "errors",
        "utils-registry",
        "docs-build",
    }
    assert "unknown" not in sqlparse.areas
    assert sqlparse.allowed_area_names[-1] == "unknown"


@pytest.mark.parametrize(
    ("areas", "message"),
    [
        ('"": ["src/**"]', "area name"),
        ("", "areas must be non-empty"),
        ('unknown: ["src/**"]', "reserved"),
        ("parser: []", "path glob"),
    ],
)
def test_agentteam_config_rejects_invalid_areas(tmp_path, areas, message):
    path = _write_config(
        tmp_path,
        f"repos:\n  - name: example/repo\n    areas:\n      {areas}\n",
    )

    with pytest.raises(ValueError, match=message):
        load_agentteam_config(path)


def test_agentteam_config_rejects_empty_repository_name(tmp_path):
    path = _write_config(
        tmp_path,
        "repos:\n  - name: '   '\n    areas: {parser: [src/parser.py]}\n",
    )

    with pytest.raises(ValueError, match="repository name must be non-empty"):
        load_agentteam_config(path)


def test_agentteam_config_rejects_duplicate_area_names_in_yaml(tmp_path):
    path = _write_config(
        tmp_path,
        "repos:\n"
        "  - name: example/repo\n"
        "    areas:\n"
        "      parser: [src/parser.py]\n"
        "      parser: [src/other.py]\n",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_agentteam_config(path)


def test_agentteam_config_rejects_duplicate_repository_names(tmp_path):
    path = _write_config(
        tmp_path,
        "repos:\n"
        "  - name: example/repo\n"
        "    areas: {parser: [src/parser.py]}\n"
        "  - name: example/repo\n"
        "    areas: {docs: [docs/**]}\n",
    )

    with pytest.raises(ValueError, match="duplicate repository"):
        load_agentteam_config(path)
