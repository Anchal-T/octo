# Phase 3c — Offline Triage Accuracy Evaluation

## Run Metadata

- **Dataset:** `nebius/SWE-rebench`
- **Dataset revision:** `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881`
- **Target repository:** `marshmallow-code/marshmallow`
- **Unique tasks:** `61` (`test`: 61, `filtered`: 11; `filtered` subset of `test`: `True`)
- **Pinned model:** `openai/gpt-oss-20b` (provider: `groq`)
- **Evaluated at (UTC):** `2026-11-12T11:46:09.137186Z`

## Overall Accuracy & Scoring Rules

| Metric | Count / Rate | Formula / Rule |
|---|---:|---|
| **Accuracy (scorable tasks)** | **22 / 50 (44.0%)** | `hits / (attempted - no_gold_area)`; parse failures count as misses |
| **Accuracy (all attempted tasks)** | **22 / 61 (36.1%)** | `hits / attempted`; both parse failures and `no_gold_area` in denominator with 0 hits |
| Attempted / Total tasks | 61 / 61 | Completed triager model calls |
| Hits | 22 | `predicted_area in gold_areas` (one primary area suffices on multi-area diffs) |
| Misses (wrong area on non-empty gold) | 27 | `predicted_area not in gold_areas` |
| Parse failures | 1 | `parse_triage(...) is None`; recorded as `None` (not fabricated) and counted as miss |
| No gold area (`empty gold` bucket) | 11 | `areas_for_diff` returned `set()`; never counted as a hit |
| Multi-area gold tasks (strict `agentteam.yaml`) | 25 | `len(gold_areas) > 1` |
| Failed / skipped calls | 0 | Non-completed model calls |

### Treatment of Edge Cases

1. **Gold area set:** Computed strictly via `areas_for_diff(extract_patch_paths(patch), marshmallow.areas)` using `agentteam.yaml`.
2. **Multi-area patches:** A task is a **hit** when `TriageResult.area` matches any member of its non-empty gold set.
3. **Parse failures:** When `parse_triage` returns `None`, `predicted_area` is recorded as `None` (`parse_failure`) rather than fabricating an area, and counted as a miss on scorable tasks.
4. **Empty gold set (`no_gold_area`):** Tasks where no `agentteam.yaml` path glob matched the gold patch are placed in the separate `no_gold_area` bucket and never counted as hits (even if the model predicts `unknown`, since `unknown` is never in `areas_for_diff`). Both `hits / scorable` and `hits / attempted` are reported above.

### Path-Layout Observation (`marshmallow/` vs `src/marshmallow/`)

`agentteam.yaml` configures current `src/marshmallow/*.py` globs plus `docs-build` (`docs/**`, `CHANGELOG.rst`, `pyproject.toml`, `.github/workflows/**`). In `nebius/SWE-rebench`, **26** older marshmallow tasks predate the repository's move from `marshmallow/` to `src/marshmallow/`: those without `CHANGELOG.rst`/`docs/**` edits land in `no_gold_area`, while **16** legacy code fixes that also updated `CHANGELOG.rst` or `docs/**` receive `gold_areas = ['docs-build']` only under raw `agentteam.yaml` globs. As a diagnostic reference (without altering primary scoring or `agentteam.yaml`), mapping legacy `marshmallow/<path>` prefixes to `src/marshmallow/<path>` yields **39 / 59 (66.1%)** on scorable tasks (2 empty-gold tasks; 19 multi-code-area tasks).

## Per-Area Breakdown (Scorable Tasks)

| Area | Gold Support | Predicted (Scorable) | Direct Hits | Gold Task Hits (incl. multi-area) | Gold Misses | Wrong Predictions (FP) | Predicted on `no_gold_area` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fields` | 19 | 23 | 13 | 17 | 2 | 10 | 5 |
| `schema` | 15 | 15 | 7 | 9 | 6 | 8 | 2 |
| `validate` | 5 | 6 | 2 | 3 | 2 | 4 | 1 |
| `hooks` | 2 | 0 | 0 | 0 | 2 | 0 | 2 |
| `errors` | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| `utils-registry` | 9 | 2 | 0 | 7 | 2 | 2 | 0 |
| `docs-build` | 39 | 0 | 0 | 15 | 24 | 0 | 0 |
| `unknown` | 0 | 1 | 0 | 0 | 0 | 1 | 1 |

## Confusion & Non-Hit Breakdown (39 tasks)

| Instance ID | Gold Area Set | Predicted Area | Verdict | Changed Paths |
|---|---|---|---|---|
| `marshmallow-code__marshmallow-1002` | `docs-build` | `fields` | `miss` | `CHANGELOG.rst`, `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-1036` | `docs-build` | `schema` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `marshmallow/class_registry.py` |
| `marshmallow-code__marshmallow-1063` | *(empty)* | `schema` | `no_gold_area` | `marshmallow/utils.py` |
| `marshmallow-code__marshmallow-1067` | `docs-build` | `schema` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `docs/api_reference.rst`, `docs/extending.rst`, `marshmallow/fields.py`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1078` | `docs-build` | `fields` | `miss` | `CHANGELOG.rst`, `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-1079` | `docs-build` | `utils-registry` | `miss` | `CHANGELOG.rst`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1220` | `docs-build`, `hooks`, `schema` | `validate` | `miss` | `CHANGELOG.rst`, `docs/extending.rst`, `docs/quickstart.rst`, `docs/upgrading.rst`, `examples/flask_example.py`, `examples/peewee_example.py`, `performance/benchmark.py`, `src/marshmallow/decorators.py`, `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1249` | `utils-registry` | `fields` | `miss` | `src/marshmallow/utils.py` |
| `marshmallow-code__marshmallow-1252` | `utils-registry` | `validate` | `miss` | `src/marshmallow/utils.py` |
| `marshmallow-code__marshmallow-1304` | `docs-build`, `fields`, `schema` | `errors` | `miss` | `CHANGELOG.rst`, `docs/extending.rst`, `docs/quickstart.rst`, `docs/upgrading.rst`, `src/marshmallow/fields.py`, `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1343` | `docs-build`, `schema` | `validate` | `miss` | `CHANGELOG.rst`, `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1379` | `docs-build`, `schema` | `fields` | `miss` | `CHANGELOG.rst`, `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-1405` | `docs-build`, `fields` | `schema` | `miss` | `CHANGELOG.rst`, `src/marshmallow/fields.py` |
| `marshmallow-code__marshmallow-1524` | `validate` | *(parse_failure)* | `parse_failure` | `AUTHORS.rst`, `src/marshmallow/validate.py` |
| `marshmallow-code__marshmallow-168` | `docs-build` | `errors` | `miss` | `CHANGELOG.rst`, `docs/custom_fields.rst`, `docs/extending.rst`, `docs/nesting.rst`, `docs/quickstart.rst`, `docs/upgrading.rst`, `marshmallow/exceptions.py`, `marshmallow/fields.py`, `marshmallow/marshalling.py` |
| `marshmallow-code__marshmallow-2150` | `schema` | `fields` | `miss` | `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-2229` | `docs-build` | `unknown` | `miss` | `.github/dependabot.yml`, `.github/workflows/build-release.yml`, `.pre-commit-config.yaml`, `CHANGELOG.rst`, `RELEASING.md`, `src/marshmallow/__init__.py` |
| `marshmallow-code__marshmallow-2252` | *(empty)* | `unknown` | `no_gold_area` | `examples/flask_example.py`, `examples/peewee_example.py` |
| `marshmallow-code__marshmallow-234` | *(empty)* | `fields` | `no_gold_area` | `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-262` | `docs-build` | `fields` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `MANIFEST.in`, `docs/extending.rst`, `marshmallow/__init__.py`, `marshmallow/fields.py`, `marshmallow/marshalling.py`, `marshmallow/schema.py`, `marshmallow/validate.py`, `tasks.py` |
| `marshmallow-code__marshmallow-2715` | `docs-build`, `hooks`, `schema` | `utils-registry` | `miss` | `CHANGELOG.rst`, `docs/extending.rst`, `examples/peewee_example.py`, `src/marshmallow/decorators.py`, `src/marshmallow/schema.py` |
| `marshmallow-code__marshmallow-2800` | `docs-build`, `validate` | `fields` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `src/marshmallow/validate.py` |
| `marshmallow-code__marshmallow-293` | `docs-build` | `schema` | `miss` | `docs/quickstart.rst`, `marshmallow/marshalling.py`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-299` | *(empty)* | `validate` | `no_gold_area` | `marshmallow/marshalling.py` |
| `marshmallow-code__marshmallow-691` | *(empty)* | `fields` | `no_gold_area` | `AUTHORS.rst`, `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-700` | *(empty)* | `fields` | `no_gold_area` | `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-744` | *(empty)* | `hooks` | `no_gold_area` | `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-750` | *(empty)* | `hooks` | `no_gold_area` | `marshmallow/decorators.py`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-756` | `docs-build` | `fields` | `miss` | `docs/quickstart.rst`, `docs/upgrading.rst`, `marshmallow/fields.py`, `marshmallow/marshalling.py` |
| `marshmallow-code__marshmallow-819` | `docs-build` | `schema` | `miss` | `CHANGELOG.rst`, `docs/upgrading.rst`, `marshmallow/schema.py`, `marshmallow/utils.py` |
| `marshmallow-code__marshmallow-821` | *(empty)* | `fields` | `no_gold_area` | `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-823` | `docs-build` | `fields` | `miss` | `docs/nesting.rst`, `docs/upgrading.rst`, `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-826` | `docs-build` | `schema` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-872` | *(empty)* | `schema` | `no_gold_area` | `marshmallow/marshalling.py`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-945` | `docs-build` | `schema` | `miss` | `CHANGELOG.rst`, `docs/upgrading.rst`, `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-950` | `docs-build` | `schema` | `miss` | `CHANGELOG.rst`, `marshmallow/schema.py` |
| `marshmallow-code__marshmallow-959` | `docs-build` | `validate` | `miss` | `CHANGELOG.rst`, `marshmallow/marshalling.py` |
| `marshmallow-code__marshmallow-960` | *(empty)* | `fields` | `no_gold_area` | `marshmallow/fields.py` |
| `marshmallow-code__marshmallow-968` | `docs-build` | `fields` | `miss` | `AUTHORS.rst`, `CHANGELOG.rst`, `marshmallow/utils.py` |

## Skipped / Failed Calls

None — all 61 tasks completed on the pinned model.

## Complete Task Results (61 tasks)

| Instance ID | Gold Area Set | Predicted Area | Verdict |
|---|---|---|---|
| `marshmallow-code__marshmallow-1002` | `docs-build` | `fields` | **miss** |
| `marshmallow-code__marshmallow-1036` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-1063` | *(empty)* | `schema` | **no_gold_area** |
| `marshmallow-code__marshmallow-1067` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-1078` | `docs-build` | `fields` | **miss** |
| `marshmallow-code__marshmallow-1079` | `docs-build` | `utils-registry` | **miss** |
| `marshmallow-code__marshmallow-1142` | `docs-build`, `fields`, `utils-registry` | `fields` | hit |
| `marshmallow-code__marshmallow-1164` | `fields`, `schema` | `fields` | hit |
| `marshmallow-code__marshmallow-1220` | `docs-build`, `hooks`, `schema` | `validate` | **miss** |
| `marshmallow-code__marshmallow-1229` | `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1238` | `schema` | `schema` | hit |
| `marshmallow-code__marshmallow-1249` | `utils-registry` | `fields` | **miss** |
| `marshmallow-code__marshmallow-1252` | `utils-registry` | `validate` | **miss** |
| `marshmallow-code__marshmallow-1278` | `docs-build`, `fields`, `utils-registry` | `fields` | hit |
| `marshmallow-code__marshmallow-1304` | `docs-build`, `fields`, `schema` | `errors` | **miss** |
| `marshmallow-code__marshmallow-1322` | `docs-build`, `schema` | `schema` | hit |
| `marshmallow-code__marshmallow-1343` | `docs-build`, `schema` | `validate` | **miss** |
| `marshmallow-code__marshmallow-1359` | `docs-build`, `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1379` | `docs-build`, `schema` | `fields` | **miss** |
| `marshmallow-code__marshmallow-1382` | `docs-build`, `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1385` | `docs-build`, `fields`, `schema` | `schema` | hit |
| `marshmallow-code__marshmallow-1401` | `docs-build`, `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1405` | `docs-build`, `fields` | `schema` | **miss** |
| `marshmallow-code__marshmallow-1503` | `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1524` | `validate` | *(parse_failure)* | **parse_failure** |
| `marshmallow-code__marshmallow-168` | `docs-build` | `errors` | **miss** |
| `marshmallow-code__marshmallow-1682` | `docs-build`, `fields`, `schema`, `utils-registry` | `fields` | hit |
| `marshmallow-code__marshmallow-1702` | `docs-build`, `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1745` | `docs-build`, `schema` | `schema` | hit |
| `marshmallow-code__marshmallow-1777` | `docs-build`, `fields`, `utils-registry`, `validate` | `validate` | hit |
| `marshmallow-code__marshmallow-1810` | `docs-build`, `fields` | `fields` | hit |
| `marshmallow-code__marshmallow-1896` | `docs-build`, `fields`, `schema`, `utils-registry` | `schema` | hit |
| `marshmallow-code__marshmallow-1989` | `docs-build`, `schema`, `utils-registry` | `schema` | hit |
| `marshmallow-code__marshmallow-2123` | `fields`, `validate` | `fields` | hit |
| `marshmallow-code__marshmallow-2150` | `schema` | `fields` | **miss** |
| `marshmallow-code__marshmallow-2229` | `docs-build` | `unknown` | **miss** |
| `marshmallow-code__marshmallow-2252` | *(empty)* | `unknown` | **no_gold_area** |
| `marshmallow-code__marshmallow-2264` | `fields`, `utils-registry` | `fields` | hit |
| `marshmallow-code__marshmallow-2271` | `schema` | `schema` | hit |
| `marshmallow-code__marshmallow-234` | *(empty)* | `fields` | **no_gold_area** |
| `marshmallow-code__marshmallow-262` | `docs-build` | `fields` | **miss** |
| `marshmallow-code__marshmallow-2708` | `docs-build`, `fields`, `validate` | `validate` | hit |
| `marshmallow-code__marshmallow-2715` | `docs-build`, `hooks`, `schema` | `utils-registry` | **miss** |
| `marshmallow-code__marshmallow-2800` | `docs-build`, `validate` | `fields` | **miss** |
| `marshmallow-code__marshmallow-293` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-299` | *(empty)* | `validate` | **no_gold_area** |
| `marshmallow-code__marshmallow-691` | *(empty)* | `fields` | **no_gold_area** |
| `marshmallow-code__marshmallow-700` | *(empty)* | `fields` | **no_gold_area** |
| `marshmallow-code__marshmallow-744` | *(empty)* | `hooks` | **no_gold_area** |
| `marshmallow-code__marshmallow-750` | *(empty)* | `hooks` | **no_gold_area** |
| `marshmallow-code__marshmallow-756` | `docs-build` | `fields` | **miss** |
| `marshmallow-code__marshmallow-819` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-821` | *(empty)* | `fields` | **no_gold_area** |
| `marshmallow-code__marshmallow-823` | `docs-build` | `fields` | **miss** |
| `marshmallow-code__marshmallow-826` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-872` | *(empty)* | `schema` | **no_gold_area** |
| `marshmallow-code__marshmallow-945` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-950` | `docs-build` | `schema` | **miss** |
| `marshmallow-code__marshmallow-959` | `docs-build` | `validate` | **miss** |
| `marshmallow-code__marshmallow-960` | *(empty)* | `fields` | **no_gold_area** |
| `marshmallow-code__marshmallow-968` | `docs-build` | `fields` | **miss** |
