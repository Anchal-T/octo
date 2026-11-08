import services.github.app.settings as settings_module
from services.github.app.settings import GithubSettings, SettingsError


def test_settings_defaults_and_token_repr():
    settings = GithubSettings(token="never-print", repo="bot/fork")

    assert settings.workflow_url == "http://workflow:8002"
    assert settings.poll_interval_s == 60
    assert settings.max_starts_per_cycle == 1
    assert settings.max_starts_per_day == 10
    assert settings.body_max_chars == 8000
    assert "never-print" not in repr(settings)


def test_from_env_uses_supplied_mapping_and_strips_workflow_url(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "process-token")
    settings = GithubSettings.from_env(
        {
            "GITHUB_TOKEN": "mapping-token",
            "GITHUB_REPO": "bot/fork",
            "WORKFLOW_URL": "http://workflow:8002/",
            "GITHUB_POLL_INTERVAL_S": "45",
            "GITHUB_MAX_STARTS_PER_CYCLE": "2",
            "GITHUB_MAX_STARTS_PER_DAY": "12",
            "GITHUB_BODY_MAX_CHARS": "9000",
        }
    )

    assert settings.token == "mapping-token"
    assert settings.repo == "bot/fork"
    assert settings.workflow_url == "http://workflow:8002"
    assert settings.poll_interval_s == 45
    assert settings.max_starts_per_cycle == 2
    assert settings.max_starts_per_day == 12
    assert settings.body_max_chars == 9000


def test_settings_module_does_not_construct_settings_at_import():
    assert not hasattr(settings_module, "settings")


def test_validate_for_start_names_missing_and_invalid_values():
    settings = GithubSettings()
    try:
        settings.validate_for_start()
    except SettingsError as error:
        assert "GITHUB_TOKEN" in str(error)
    else:
        raise AssertionError("missing token was accepted")

    valid = GithubSettings(token="token", repo="bot/fork")
    valid.validate_for_start()

    cases = [
        (GithubSettings(token="token", repo="bad/repo/name"), "GITHUB_REPO"),
        (
            GithubSettings(token="token", repo="bot/fork", poll_interval_s=29),
            "GITHUB_POLL_INTERVAL_S",
        ),
        (
            GithubSettings(
                token="token",
                repo="bot/fork",
                max_starts_per_cycle=3,
                max_starts_per_day=2,
            ),
            "GITHUB_MAX_STARTS_PER_DAY",
        ),
        (
            GithubSettings(token="token", repo="bot/fork", body_max_chars=99),
            "GITHUB_BODY_MAX_CHARS",
        ),
    ]
    for invalid, expected in cases:
        try:
            invalid.validate_for_start()
        except SettingsError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"invalid {expected} was accepted")


def test_process_own_issues_defaults_to_false_and_parses_env():
    assert GithubSettings().process_own_issues is False
    assert GithubSettings.from_env({}).process_own_issues is False

    for raw in ("true", "TRUE", " 1 ", "yes"):
        assert (
            GithubSettings.from_env({"GITHUB_PROCESS_OWN_ISSUES": raw})
        ).process_own_issues is True
    for raw in ("false", "0", "no", ""):
        assert (
            GithubSettings.from_env({"GITHUB_PROCESS_OWN_ISSUES": raw})
        ).process_own_issues is False


def test_process_own_issues_rejects_unrecognised_values():
    try:
        GithubSettings.from_env({"GITHUB_PROCESS_OWN_ISSUES": "maybe"})
    except SettingsError as error:
        assert "GITHUB_PROCESS_OWN_ISSUES" in str(error)
    else:
        raise AssertionError("unrecognised value was accepted")
