import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentteam_common import Settings, load_agentteam_config
from agentteam_contracts import ModelOutcome

from eval.triage_eval import (
    TaskItem,
    aggregate_eval,
    extract_issue_text,
    extract_patch_paths,
    gold_areas_for_patch,
    load_parquet_bundle,
    render_markdown_report,
    run_triage_eval,
    score_task,
    summary_to_json_dict,
)
from services.workflow.app.triage import triage_messages

REPO = "marshmallow-code/marshmallow"
MODEL = "openai/gpt-oss-20b"
FIXED_NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)


def _completed_outcome(
    text: str,
    *,
    run_id: str = "eval:triage:t1",
    tokens_in: int = 120,
    tokens_out: int = 40,
) -> ModelOutcome:
    return ModelOutcome(
        outcome="completed",
        request_id="mc-test",
        decided_at=FIXED_NOW,
        model_pinned=MODEL,
        model_reported=MODEL,
        run_id=run_id,
        step="triage",
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        started_at=FIXED_NOW,
        finished_at=FIXED_NOW + timedelta(seconds=1),
    )


def test_extract_patch_paths_handles_modified_added_deleted_and_renamed_files():
    patch = (
        "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/marshmallow/fields.py\n"
        "+++ b/src/marshmallow/fields.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " context\n"
        "diff --git a/src/marshmallow/new_mod.py b/src/marshmallow/new_mod.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/marshmallow/new_mod.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
        "diff --git a/src/marshmallow/gone.py b/src/marshmallow/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/src/marshmallow/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
        "diff --git a/src/marshmallow/legacy.py b/src/marshmallow/utils.py\n"
        "similarity index 100%\n"
        "rename from src/marshmallow/legacy.py\n"
        "rename to src/marshmallow/utils.py\n"
    )

    assert extract_patch_paths(patch) == (
        "src/marshmallow/fields.py",
        "src/marshmallow/new_mod.py",
        "src/marshmallow/gone.py",
        "src/marshmallow/legacy.py",
        "src/marshmallow/utils.py",
    )


def test_extract_patch_paths_ignores_triple_dash_and_plus_inside_hunks():
    patch = (
        "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
        "--- a/src/marshmallow/fields.py\n"
        "+++ b/src/marshmallow/fields.py\n"
        "@@ -10,4 +10,4 @@\n"
        " keep_before\n"
        "\n"
        "--- src/marshmallow/schema.py\n"
        "+++ src/marshmallow/validate.py\n"
        " keep_after\n"
    )

    assert extract_patch_paths(patch) == ("src/marshmallow/fields.py",)


def test_extract_patch_paths_supports_plain_unified_diff_with_timestamps():
    patch = (
        "--- a/src/marshmallow/validate.py\t2026-11-01 00:00:00 +0000\n"
        "+++ b/src/marshmallow/validate.py\t2026-11-01 00:01:00 +0000\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )

    assert extract_patch_paths(patch) == ("src/marshmallow/validate.py",)


def test_extract_issue_text_splits_title_and_body_and_wires_allowed_areas():
    config = load_agentteam_config().repo(REPO)
    task = TaskItem(
        instance_id="marshmallow-code__marshmallow-1229",
        repo=REPO,
        problem_statement=(
            "\r\n`only` argument inconsistent on Nested\r\n\r\n"
            "First paragraph.\n    >>> Schema()\n"
        ),
        patch="",
    )

    title, body = extract_issue_text(task)
    messages = triage_messages(title, body, allowed_areas=config.allowed_area_names)

    assert title == "`only` argument inconsistent on Nested"
    assert body == "First paragraph.\n    >>> Schema()"
    assert (
        '"area": "fields|schema|validate|hooks|errors|utils-registry|docs-build|unknown"'
        in messages[0]["content"]
    )


def test_gold_areas_for_patch_matches_single_multi_and_empty_marshmallow_patches():
    repo_areas = load_agentteam_config().repo(REPO).areas

    single_patch = (
        "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
        "--- a/src/marshmallow/fields.py\n"
        "+++ b/src/marshmallow/fields.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    multi_patch = (
        "diff --git a/CHANGELOG.rst b/CHANGELOG.rst\n"
        "--- a/CHANGELOG.rst\n+++ b/CHANGELOG.rst\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
        "--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/src/marshmallow/schema.py b/src/marshmallow/schema.py\n"
        "--- a/src/marshmallow/schema.py\n+++ b/src/marshmallow/schema.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    unmatched_patch = (
        "diff --git a/examples/flask_example.py b/examples/flask_example.py\n"
        "--- a/examples/flask_example.py\n+++ b/examples/flask_example.py\n@@ -1 +1 @@\n-a\n+b\n"
    )

    assert gold_areas_for_patch(single_patch, repo_areas) == ("fields",)
    assert gold_areas_for_patch(multi_patch, repo_areas) == (
        "docs-build",
        "fields",
        "schema",
    )
    assert gold_areas_for_patch(unmatched_patch, repo_areas) == ()


def test_score_task_covers_hit_multi_area_miss_parse_failure_empty_gold_and_failed_call():
    repo_cfg = load_agentteam_config().repo(REPO)
    areas = repo_cfg.areas
    allowed = repo_cfg.allowed_area_names

    multi_task = TaskItem(
        instance_id="m-multi",
        repo=REPO,
        problem_statement="Nested schema issue\nDetails",
        patch=(
            "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
            "--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/marshmallow/schema.py b/src/marshmallow/schema.py\n"
            "--- a/src/marshmallow/schema.py\n+++ b/src/marshmallow/schema.py\n@@ -1 +1 @@\n-a\n+b\n"
        ),
    )
    empty_gold_task = TaskItem(
        instance_id="m-empty",
        repo=REPO,
        problem_statement="Update example\nDetails",
        patch=(
            "diff --git a/examples/flask_example.py b/examples/flask_example.py\n"
            "--- a/examples/flask_example.py\n+++ b/examples/flask_example.py\n@@ -1 +1 @@\n-a\n+b\n"
        ),
    )

    hit = score_task(
        multi_task,
        _completed_outcome(
            '{"kind":"bug","priority":"medium","summary":"S.","area":"schema"}'
        ),
        repo_areas=areas,
        allowed_areas=allowed,
    )
    assert hit.verdict == "hit"
    assert hit.predicted_area == "schema"
    assert hit.gold_areas == ("fields", "schema")

    miss = score_task(
        multi_task,
        _completed_outcome(
            '{"kind":"bug","priority":"medium","summary":"S.","area":"validate"}'
        ),
        repo_areas=areas,
        allowed_areas=allowed,
    )
    assert miss.verdict == "miss"
    assert miss.predicted_area == "validate"

    unparsed = score_task(
        multi_task,
        _completed_outcome("not valid json at all"),
        repo_areas=areas,
        allowed_areas=allowed,
    )
    assert unparsed.verdict == "parse_failure"
    assert unparsed.parse_failure is True
    assert unparsed.predicted_area is None

    no_gold = score_task(
        empty_gold_task,
        _completed_outcome(
            '{"kind":"docs","priority":"low","summary":"S.","area":"unknown"}'
        ),
        repo_areas=areas,
        allowed_areas=allowed,
    )
    assert no_gold.verdict == "no_gold_area"
    assert no_gold.predicted_area == "unknown"
    assert no_gold.gold_areas == ()

    failed_call = score_task(
        multi_task,
        ModelOutcome(
            outcome="failed",
            request_id="mc-fail",
            decided_at=FIXED_NOW,
            model_pinned=MODEL,
            run_id="eval:triage:m-multi",
            step="triage",
            error_code="provider_http_500",
            reason="boom",
        ),
        repo_areas=areas,
        allowed_areas=allowed,
    )
    assert failed_call.verdict == "call_failed"
    assert failed_call.predicted_area is None


def test_aggregate_and_render_report_computes_accuracy_area_breakdown_and_confusions():
    repo_cfg = load_agentteam_config().repo(REPO)
    tasks = (
        TaskItem(
            instance_id="m-1",
            repo=REPO,
            problem_statement="Field bug\nBody",
            patch=(
                "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
                "--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n@@ -1 +1 @@\n-a\n+b\n"
            ),
        ),
        TaskItem(
            instance_id="m-2",
            repo=REPO,
            problem_statement="Multi area fix\nBody",
            patch=(
                "diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py\n"
                "--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n@@ -1 +1 @@\n-a\n+b\n"
                "diff --git a/src/marshmallow/schema.py b/src/marshmallow/schema.py\n"
                "--- a/src/marshmallow/schema.py\n+++ b/src/marshmallow/schema.py\n@@ -1 +1 @@\n-a\n+b\n"
            ),
        ),
        TaskItem(
            instance_id="m-3",
            repo=REPO,
            problem_statement="Validate bug\nBody",
            patch=(
                "diff --git a/src/marshmallow/validate.py b/src/marshmallow/validate.py\n"
                "--- a/src/marshmallow/validate.py\n+++ b/src/marshmallow/validate.py\n@@ -1 +1 @@\n-a\n+b\n"
            ),
        ),
        TaskItem(
            instance_id="m-4",
            repo=REPO,
            problem_statement="Broken json\nBody",
            patch=(
                "diff --git a/src/marshmallow/hooks_wrong.py b/src/marshmallow/decorators.py\n"
                "--- /dev/null\n+++ b/src/marshmallow/decorators.py\n@@ -0,0 +1 @@\n+a\n"
            ),
        ),
        TaskItem(
            instance_id="m-5",
            repo=REPO,
            problem_statement="Examples tweak\nBody",
            patch=(
                "diff --git a/examples/flask_example.py b/examples/flask_example.py\n"
                "--- a/examples/flask_example.py\n+++ b/examples/flask_example.py\n@@ -1 +1 @@\n-a\n+b\n"
            ),
        ),
    )
    outcomes = (
        _completed_outcome(
            '{"kind":"bug","priority":"low","summary":"S.","area":"fields"}'
        ),
        _completed_outcome(
            '{"kind":"bug","priority":"low","summary":"S.","area":"schema"}'
        ),
        _completed_outcome(
            '{"kind":"bug","priority":"low","summary":"S.","area":"fields"}'
        ),
        _completed_outcome("unparseable text"),
        _completed_outcome(
            '{"kind":"docs","priority":"low","summary":"S.","area":"docs-build"}'
        ),
    )
    records = tuple(
        score_task(
            task,
            outcome,
            repo_areas=repo_cfg.areas,
            allowed_areas=repo_cfg.allowed_area_names,
        )
        for task, outcome in zip(tasks, outcomes, strict=True)
    )

    summary = aggregate_eval(
        dataset_id="nebius/SWE-rebench",
        dataset_revision="deadbeef",
        repo=REPO,
        model_pinned=MODEL,
        provider="groq",
        evaluated_at="2026-11-12T12:00:00Z",
        split_counts={"test": 5, "filtered": 1},
        filtered_subset_of_test=True,
        configured_areas=tuple(repo_cfg.areas),
        tasks=tasks,
        records=records,
    )

    assert summary.total_tasks == 5
    assert summary.attempted == 5
    assert summary.scorable == 4
    assert summary.hits == 2
    assert summary.misses == 1
    assert summary.parse_failures == 1
    assert summary.no_gold_area == 1
    assert summary.failed_calls == 0
    assert summary.multi_area_gold_count == 1
    assert summary.accuracy_scorable == pytest.approx(2 / 4)
    assert summary.accuracy_all_attempted == pytest.approx(2 / 5)

    by_area = {row.area: row for row in summary.area_stats}
    assert by_area["fields"].gold_support == 2
    assert by_area["fields"].direct_hits == 1
    assert by_area["fields"].gold_task_hits == 2
    assert by_area["fields"].gold_misses == 0
    assert by_area["fields"].false_positives == 1
    assert by_area["validate"].gold_support == 1
    assert by_area["validate"].gold_misses == 1
    assert by_area["hooks"].gold_support == 1
    assert by_area["hooks"].gold_misses == 1

    markdown = render_markdown_report(summary)
    payload = summary_to_json_dict(summary)

    assert "nebius/SWE-rebench" in markdown
    assert "deadbeef" in markdown
    assert MODEL in markdown
    assert "50.0%" in markdown
    assert "40.0%" in markdown
    assert "m-3" in markdown
    assert "m-4" in markdown
    assert "m-5" in markdown
    assert payload["hits"] == 2
    assert payload["no_gold_area"] == 1


def test_load_parquet_bundle_deduplicates_filtered_subset_and_filters_repo(tmp_path):
    schema = pa.schema(
        [
            ("instance_id", pa.string()),
            ("repo", pa.string()),
            ("problem_statement", pa.string()),
            ("patch", pa.string()),
            ("base_commit", pa.string()),
            ("created_at", pa.string()),
        ]
    )
    test_rows = [
        {
            "instance_id": "marshmallow-code__marshmallow-10",
            "repo": REPO,
            "problem_statement": "Title 10\nBody 10",
            "patch": "--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n",
            "base_commit": "abc1",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "instance_id": "marshmallow-code__marshmallow-20",
            "repo": REPO,
            "problem_statement": "Title 20\nBody 20",
            "patch": "--- a/src/marshmallow/schema.py\n+++ b/src/marshmallow/schema.py\n",
            "base_commit": "abc2",
            "created_at": "2024-01-02T00:00:00Z",
        },
        {
            "instance_id": "other__repo-1",
            "repo": "other/repo",
            "problem_statement": "Other\nBody",
            "patch": "--- a/foo.py\n+++ b/foo.py\n",
            "base_commit": "def1",
            "created_at": "2024-01-03T00:00:00Z",
        },
    ]
    filtered_rows = [test_rows[0]]

    pq.write_table(
        pa.Table.from_pylist(test_rows, schema=schema),
        tmp_path / "test-00000-of-00001.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(filtered_rows, schema=schema),
        tmp_path / "filtered-00000-of-00001.parquet",
    )

    bundle = load_parquet_bundle(
        parquet_dir=tmp_path,
        repo=REPO,
        dataset_id="nebius/SWE-rebench",
        revision="rev123",
    )

    assert len(bundle.tasks) == 2
    assert [t.instance_id for t in bundle.tasks] == [
        "marshmallow-code__marshmallow-10",
        "marshmallow-code__marshmallow-20",
    ]
    assert bundle.split_counts == {"test": 2, "filtered": 1}
    assert bundle.filtered_subset_of_test is True


def test_run_triage_eval_resumes_from_cache_and_backs_off_on_429(tmp_path):
    tasks = (
        TaskItem(
            instance_id="m-1",
            repo=REPO,
            problem_statement="Fields bug\nDetails 1",
            patch="--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n",
        ),
        TaskItem(
            instance_id="m-2",
            repo=REPO,
            problem_statement="Schema bug\nDetails 2",
            patch="--- a/src/marshmallow/schema.py\n+++ b/src/marshmallow/schema.py\n",
        ),
    )
    settings = replace(
        Settings(),
        groq_api_key="fake-secret-do-not-print",
        models_sqlite_path=str(tmp_path / "cache" / "models.sqlite"),
    )
    cache_path = tmp_path / "cache" / "outcomes.json"
    calls: list[str] = []
    sleeps: list[float] = []
    now_ref = [FIXED_NOW]

    def clock() -> datetime:
        return now_ref[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now_ref[0] = now_ref[0] + timedelta(seconds=seconds)

    # First run: m-1 succeeds; m-2 gets 500 twice and stays incomplete.
    def first_provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]
        calls.append(user_msg)
        if "Fields bug" in user_msg:
            return httpx.Response(
                200,
                json={
                    "model": MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": '{"kind":"bug","priority":"low","summary":"S1.","area":"fields"}'
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 25},
                },
            )
        return httpx.Response(400, json={"error": {"message": "bad"}})

    first_summary = run_triage_eval(
        tasks=tasks,
        dataset_id="nebius/SWE-rebench",
        dataset_revision="rev1",
        repo=REPO,
        settings=settings,
        cache_path=cache_path,
        transport=httpx.MockTransport(first_provider),
        now=clock,
        sleep=fake_sleep,
    )
    assert first_summary.hits == 1
    assert first_summary.failed_calls == 1
    assert len(calls) == 2

    # Second run: m-1 must be served from cache (not re-sent); m-2 hits a 429 then 200.
    calls.clear()
    attempt_m2 = [0]

    def second_provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]
        calls.append(user_msg)
        attempt_m2[0] += 1
        if attempt_m2[0] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "50"},
                json={"error": {"message": "rate limit"}},
            )
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {
                        "message": {
                            "content": '{"kind":"bug","priority":"medium","summary":"S2.","area":"schema"}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 110, "completion_tokens": 30},
            },
        )

    second_summary = run_triage_eval(
        tasks=tasks,
        dataset_id="nebius/SWE-rebench",
        dataset_revision="rev1",
        repo=REPO,
        settings=settings,
        cache_path=cache_path,
        transport=httpx.MockTransport(second_provider),
        now=clock,
        sleep=fake_sleep,
    )

    assert second_summary.hits == 2
    assert second_summary.failed_calls == 0
    assert all("Fields bug" not in item for item in calls)
    assert len(calls) == 2
    assert any(wait >= 50 for wait in sleeps)
    assert "fake-secret-do-not-print" not in render_markdown_report(second_summary)
    assert "fake-secret-do-not-print" not in cache_path.read_text(encoding="utf-8")


def test_run_triage_eval_records_blocker_without_network_when_api_key_missing(tmp_path):
    tasks = (
        TaskItem(
            instance_id="m-1",
            repo=REPO,
            problem_statement="Fields bug\nDetails 1",
            patch="--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n",
        ),
    )
    settings = replace(
        Settings(),
        groq_api_key="",
        models_sqlite_path=str(tmp_path / "cache" / "models.sqlite"),
    )

    def forbid_network(_request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP must not be called when GROQ_API_KEY is missing")

    summary = run_triage_eval(
        tasks=tasks,
        dataset_id="nebius/SWE-rebench",
        dataset_revision="rev1",
        repo=REPO,
        settings=settings,
        cache_path=tmp_path / "cache" / "outcomes.json",
        transport=httpx.MockTransport(forbid_network),
        now=lambda: FIXED_NOW,
    )

    assert summary.blocker == "MissingGroqApiKey"
    assert summary.attempted == 0
    assert summary.accuracy_scorable is None
    report = render_markdown_report(summary)
    assert "MissingGroqApiKey" in report


def test_run_triage_eval_waits_for_minute_quota_and_stops_on_daily_quota(tmp_path):
    tasks = tuple(
        TaskItem(
            instance_id=f"m-{idx}",
            repo=REPO,
            problem_statement=f"Bug {idx}\nDetails",
            patch="--- a/src/marshmallow/fields.py\n+++ b/src/marshmallow/fields.py\n",
        )
        for idx in (1, 2, 3)
    )
    settings = replace(
        Settings(),
        groq_api_key="test-key",
        groq_rpm=1,
        groq_rpd=2,
        models_sqlite_path=str(tmp_path / "cache" / "models.sqlite"),
    )
    now_ref = [datetime(2026, 11, 12, 12, 0, 40, tzinfo=UTC)]
    sleeps: list[float] = []

    def clock() -> datetime:
        return now_ref[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now_ref[0] = now_ref[0] + timedelta(seconds=seconds)

    def ok_provider(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {
                        "message": {
                            "content": '{"kind":"bug","priority":"low","summary":"Ok.","area":"fields"}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 20},
            },
        )

    summary = run_triage_eval(
        tasks=tasks,
        dataset_id="nebius/SWE-rebench",
        dataset_revision="rev1",
        repo=REPO,
        settings=settings,
        cache_path=tmp_path / "cache" / "outcomes.json",
        max_wait_s=120.0,
        transport=httpx.MockTransport(ok_provider),
        now=clock,
        sleep=fake_sleep,
    )

    assert summary.attempted == 2
    assert summary.hits == 2
    assert summary.failed_calls == 1
    assert sleeps == [20.5, 60.0]
