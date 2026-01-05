# improper-match-short-circuit (Issue #72)

- Date captured: 2026-01-05
- Author: Dan Moisan
- Status: Fixed in feature/original-publication-date-71 (Issue #72)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #72
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/72
- Last Updated: 2026-01-05 (validated fix)
## Summary

`select_best_match` short-circuits on the first exact title/author overlap, causing later candidates with earlier publication years to be ignored; regression test shows it returns 2014 instead of the earliest year 1956 for duplicated Moby Dick entries.

## Environment

- OS/version: Windows 10/11 (dev machine)
- Python version: 3.13.7
- Command/flags used: `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`
- Data source or fixture: Inline `MatchCandidate` list for duplicated “Moby-Dick; or, the Whale” records

## Steps to Reproduce

1. Ensure repo deps installed (Poetry environment active).
2. Run `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`.
3. Observe the failing regression test `test_select_best_match_prefers_earliest_exact_year_for_duplicates`.

## Expected Behavior

When multiple exact title/author matches exist, the earliest publication year (1956) should be selected.

## Actual Behavior

Test fails because `select_best_match` returns the first exact match in iteration order (year 2014) instead of the earliest year 1956. Key assertion failure:

```
>       assert result.year == 1956
E       AssertionError: assert 2014 == 1956
E        +  where 2014 = MatchResult(year=2014, confidence='high', source='openlibrary').year
```

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet: see traceback above from pytest run

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Short-circuit in `select_best_match` treats the first exact title/author overlap with a year as definitive, introducing order bias when multiple reprints exist. See [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py](src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py).

## Proposed Fix / Validation Ideas

- [x] Unit coverage areas
	- Adjust `select_best_match` to consider multiple matches and choose earliest year; extend/parametrize regression test if needed.
    - Add another unit test where a candidate that is not a perfect match, but is above the match threshhold carries the winning date
    - Add a negative test where a candidate that is just below the matching threshhold has a date of 1900. This would be the earliest publication date but would be rejected since it is below threshhold  
- [ ] Integration scenario to retest
- [ ] Manual verification notes

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [x] Move to active fix folder / branch

## Resolution

- Updated `select_best_match` to track the earliest exact title/author match instead of short-circuiting on the first candidate.
- Validation:
    - `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`
    - Toolchain: `poetry run black .` → `poetry run ruff check` → `poetry run pyright` → `poetry run pytest`