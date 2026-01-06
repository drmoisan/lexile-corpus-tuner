# improper-match-short-circuit (Spec)

- Issue: #72
- Owner: improper-match-short-circuit
- Date: 2026-01-05
- Status: Implemented

## Context
`select_best_match` short-circuits on the first exact title/author overlap, causing later candidates with earlier publication years to be ignored; regression test shows it returns 2014 instead of the earliest year 1956 for duplicated Moby Dick entries.

Environment:
- OS/version: Windows 10/11 (dev machine)
- Python version: 3.13.7
- Command/flags used: `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`
- Data source or fixture: Inline `MatchCandidate` list for duplicated "Moby-Dick; or, the Whale" records

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Ensure repo deps installed (Poetry environment active).
2. Run `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`.
3. Observe the failing regression test `test_select_best_match_prefers_earliest_exact_year_for_duplicates`.

Expected:
When multiple exact title/author matches exist, the earliest publication year (1956) should be selected.

Actual:
Test fails because `select_best_match` returns the first exact match in iteration order (year 2014) instead of the earliest year 1956. Key assertion failure:

```
>       assert result.year == 1956
E       AssertionError: assert 2014 == 1956
E        +  where 2014 = MatchResult(year=2014, confidence='high', source='openlibrary').year
```

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet: see traceback above from pytest run


## Scope & Non-Goals
- In scope:
    - Fix `select_best_match` to remove order bias when multiple exact title/author matches exist.
    - Ensure earliest qualifying publication year is chosen among exact matches.
    - Preserve fuzzy matching behavior subject to threshold settings.
- Out of scope / non-goals:
    - Changing match_candidate data structures or external provider schemas.
    - Broader pipeline refactors outside the selection logic.
    - UI/CLI changes beyond existing tooling output.

## Root Cause Analysis
Short-circuit in `select_best_match` treats the first exact title/author overlap with a year as definitive, introducing order bias when multiple reprints exist. See [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py](src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py).


## Proposed Fix
- [x] Unit coverage areas
    - Adjusted `select_best_match` to consider multiple exact matches and choose the earliest year (regression guards this behavior).
    - De-scoped additional fuzzy/negative test permutations; regression plus full suite cover current scope.
- [ ] Integration scenario to retest
- [ ] Manual verification notes


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
    - Normalization rules (`normalize_text`) remain unchanged and deterministic.
    - Providers may return multiple editions; ordering is not reliable.
    - Existing fuzzy threshold defaults remain appropriate unless changed by fix.
- Constraints (budget, performance, compatibility):
    - Selection logic should stay O(n) over candidates and remain deterministic.
    - No additional dependencies introduced.
- External dependencies (services, libraries, releases):
    - Relies on current provider responses (e.g., openlibrary) but test uses in-memory candidates only.

## Data / API / Config Impact
- User-facing or API changes:
    - None; internal selection behavior change only.
- Data or migration considerations:
    - None; no persisted data changes.
- Logging/telemetry updates (if any):
    - None planned.

## Test Strategy
- [x] Unit coverage areas
    - Regression test `test_select_best_match_prefers_earliest_exact_year_for_duplicates` now passes with earliest-year selection.
- [ ] Integration scenario to retest
- [ ] Manual verification notes

## Validation
- `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`
- Toolchain loop (final pass): `poetry run black .` → `poetry run ruff check` → `poetry run pyright` → `poetry run pytest`


## Acceptance Criteria
- Earliest qualifying year among exact title/author matches is returned (regression test passes with 1956).
- A fuzzily matching candidate above threshold can supply the winning date when no exact match exists. (Not impacted by this minimal fix; existing behavior retained.)
- A sub-threshold candidate with an early year (e.g., 1900) is not selected; result reflects correct fallback confidence/none as appropriate. (Not impacted by this minimal fix; existing behavior retained.)
- No change to existing successful match behaviors outside these scenarios.

## Risks & Mitigations
- Technical or operational risks:
    - Potential unintended change to fuzzy selection ordering.
    - Edge cases where author overlap logic interacts with new earliest-year logic.
- Mitigations and rollbacks:
    - Keep changes minimal and covered by targeted unit tests.
    - If regressions occur, revert to prior selection logic and reassess tests.

## Rollout & Follow-up
- Release/rollout steps:
    - Merge fix; run full test/toolchain.
- Post-fix monitoring or clean-up tasks:
    - None beyond confirming regression tests stay green in CI.
- Links: issue, PRs, related docs
    - Issue #72
