# lexile-v2-adapter-python-3-10 (Issue #23)

- Date captured: 2025-12-08
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/lexile-v2-adapter-python-3-10_Potential_Bug/ (Issue #23)
- Issue: #23
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/23
- Last Updated: 2025-12-09

## Summary

Pyright type checking fails on Python 3.10 for `tests/test_lexile_v2_adapter.py` due to missing type arguments for NumPy `ndarray` generic class. The errors do not occur on Python 3.11, 3.12, or 3.13.

## Environment

- OS/version: Ubuntu (GitHub Actions runner)
- Python version: 3.10.19
- Command/flags used: `poetry run pyright`
- Data source or fixture: N/A (type checking error, not runtime)

## Steps to Reproduce

1. Check out the repository on a Python 3.10 environment.
2. Run `poetry install --with dev --extras lexile-v2`
3. Run `poetry run pyright`
4. Observe 6 type errors in `tests/test_lexile_v2_adapter.py`

## Expected Behavior

Pyright should pass without errors on all supported Python versions (3.10, 3.11, 3.12, 3.13), consistent with the CI matrix in `.github/workflows/ci.yml`.

## Actual Behavior

Pyright reports 6 errors on Python 3.10 only:

```
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:10:9 - error: Return type, "ndarray[Unknown, Unknown]", is partially unknown (reportUnknownParameterType)
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:10:52 - error: Expected type arguments for generic class "ndarray" (reportMissingTypeArgument)
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:15:9 - error: Return type, "ndarray[Unknown, Unknown]", is partially unknown (reportUnknownParameterType)
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:15:40 - error: Expected type arguments for generic class "ndarray" (reportMissingTypeArgument)
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:20:9 - error: Return type, "ndarray[Unknown, Unknown]", is partially unknown (reportUnknownParameterType)
/home/runner/work/lexile-corpus-tuner/lexile-corpus-tuner/tests/test_lexile_v2_adapter.py:20:50 - error: Expected type arguments for generic class "ndarray" (reportMissingTypeArgument)
```

Errors occur at lines 10, 15, and 20 (fixture return types using bare `np.ndarray` without type arguments).

## Logs / Screenshots

- [X] Attached minimal logs or screenshot

- Snippet: See "Actual Behavior" section above

## Impact / Severity

- [ ] Blocker
- [X] High
- [ ] Medium
- [ ] Low

**Rationale:** Breaks CI on Python 3.10, preventing merges to protected branches. Does not affect runtime behavior or other Python versions.

## Suspected Cause / Notes

- NumPy's type stubs evolved between Python 3.10 and 3.11+.
- Python 3.10 may require explicit type arguments for `np.ndarray` (e.g., `np.ndarray[Any, np.dtype[np.float64]]`) while later versions infer them.
- Pyright strictness combined with Python 3.10's older type stub definitions exposes this inconsistency.
- Affects test fixtures at lines 10, 15, and 20 in `tests/test_lexile_v2_adapter.py`.

## Proposed Fix / Validation Ideas

- [ ] Add explicit type arguments to `np.ndarray` return annotations in affected fixtures (lines 10, 15, 20).
- [ ] Use `np.ndarray[Any, np.dtype[np.float64]]` or similar fully-specified generic form.
- [ ] Alternatively, use `npt.NDArray[np.float64]` from `numpy.typing` (cleaner, more idiomatic).
- [ ] Unit coverage areas: Verify existing tests still pass after annotation changes.
- [ ] Integration scenario to retest: Run full CI matrix (Python 3.10–3.13) after fix.
- [ ] Manual verification notes: Run `poetry run pyright` locally on Python 3.10 environment before/after fix.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch
