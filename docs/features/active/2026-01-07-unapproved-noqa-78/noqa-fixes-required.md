# Noqa Suppression Fixes Required

## Summary

Based on comprehensive repository audit, the following fixes are required to bring all `# noqa` suppressions into compliance with the updated python-suppressions policy.

---

## ✅ Policy Updated

**Added 8 new pre-authorized patterns:**
1. ARG002 - Unused method arguments in test mocks
2. B008 - Function call in Typer CLI defaults
3. TCH002/TCH003 - Runtime + type hint dual-use imports
4. S310 - urllib accessing trusted HTTPS endpoints
5. S314 - XML parsing of trusted local files
6. BLE001 - Blind except at CLI entry points only
7. S301 - Pickle loading from hardcoded trusted paths
8. S108/S105 - Hardcoded paths/literals in tests only

**Added 5 non-authorized patterns with workarounds:**
1. TID252 - Use absolute imports instead
2. S607 - Use shutil.which() validation
3. D401 - Fix docstring to imperative mood
4. F401 - Remove or use imports
5. UP017 - Use timezone-aware datetime

---

## 🔧 Fixes Required

### 1. TID252: Convert to absolute imports (3 instances)

**Files:**
- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/gutenberg_query_builder_ui/widgets.py`
- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/gutenberg_query_builder_ui/app.py` (2 instances)

**Example Fix:**
```python
# Before:
from ..gutenberg_query_core import QueryGroupModel  # noqa: TID252

# After:
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_core import (
    QueryGroupModel,
)
```

---

### 2. S607: Use shutil.which() validation (1 instance)

**File:**
- `scripts/dev_tools/collect_commit_context.py`

**Current Code:**
```python
result = subprocess.run(
    ["git", *args],  # noqa: S607
    ...
)
```

**Required Fix:**
```python
import shutil

git_exe = shutil.which("git")
if not git_exe:
    raise FileNotFoundError("git executable not found on PATH")

result = subprocess.run(
    [git_exe, *args],  # noqa: S603 - static analysis can't verify runtime validation
    ...
)
```

---

### 3. D401: Fix docstrings to imperative mood (2 instances)

**Files:**
- `tests/scripts/dev_tools/test_resolve_execute_plan_prompt.py`
- `tests/scripts/dev_tools/test_new_active_feature_folder.py` (2 methods)

**Current Code:**
```python
def copy(self, text: str) -> None:  # noqa: D401
    """Mock clipboard copy."""
```

**Required Fix:**
```python
def copy(self, text: str) -> None:
    """Copy text to mock clipboard."""
```

---

### 4. F401: Remove or document unused import (1 instance)

**File:**
- `scripts/dev_tools/pr_context/collector.py`

**Action:** Examine line 38 and either:
- Use the import if it's needed
- Remove the import if it's not needed
- Add to `__all__` if it's a re-export

---

### 5. UP017: Use timezone-aware datetime (1 instance)

**File:**
- `scripts/dev_tools/pr_context/summary_helpers.py`

**Action:** Examine line 382 and convert to timezone-aware datetime:
```python
# Instead of:
now = datetime.now()

# Use:
from datetime import timezone
now = datetime.now(timezone.utc)
```

---

### 6. ANN401: Investigate Any type annotations (3 instances)

**Files:**
- `tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/conftest.py` (2 instances)
- `tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_enrich_original_pub_year.py` (1 instance)

**Action:** Case-by-case analysis:
- Context manager protocol methods may be justified (pre-authorize if pattern emerges)
- Generic mock constructors may need better type design
- Each needs individual evaluation

---

## 📊 Fix Priority

| Priority | Fix Type | Count | Estimated Effort |
|----------|----------|-------|------------------|
| High | TID252 (absolute imports) | 3 | 10 min |
| High | S607 (shutil.which) | 1 | 5 min |
| Medium | D401 (docstrings) | 2 | 5 min |
| Medium | F401 (unused import) | 1 | 2 min |
| Medium | UP017 (timezone) | 1 | 5 min |
| Low | ANN401 (investigate) | 3 | 30 min |

**Total Estimated Time:** ~1 hour

---

## ✅ Already Compliant

The following suppressions are already compliant with the updated policy:

- **S603**: All instances (15+) use proper shutil.which() validation
- **S110**: Single instance for optional pyperclip import (justified)
- **ARG002**: All instances (15+) are test mocks matching known APIs
- **B008**: All instances (20+) are Typer CLI declarations
- **TCH002/TCH003**: All instances (10+) are dual runtime + type hint imports
- **S310**: All instances (6) access trusted HTTPS endpoints
- **S314**: All instances (5) parse trusted local files
- **BLE001**: All instances (6) are CLI top-level error handlers
- **S301**: Single instance loading from hardcoded path
- **S108/S105**: All instances (2) are test fixtures

---

## 🚀 Execution Plan

1. **Create fixes branch** (if not already on feature branch)
2. **Apply fixes in priority order:**
   - TID252: Convert 3 relative imports to absolute
   - S607: Add shutil.which() validation
   - D401: Fix 2 docstrings
   - F401: Remove or use unused import
   - UP017: Convert to timezone-aware datetime
3. **Run toolchain:**
   - Black formatting
   - Ruff linting (should pass without new noqa)
   - Pyright type checking
   - Pytest (all tests must pass)
4. **Investigate ANN401 cases**
5. **Commit with clear message**
6. **Update audit report with results**

---

## 📝 Commit Message Template

```
fix: bring all noqa suppressions into policy compliance

- Convert 3 TID252 relative imports to absolute paths
- Add shutil.which() validation for git subprocess (S607)
- Fix 2 D401 docstrings to use imperative mood  
- Remove/fix F401 unused import in pr_context/collector
- Convert UP017 datetime to timezone-aware
- All remaining noqa suppressions match pre-authorized patterns

Closes: Part of noqa policy enforcement initiative
```

---

## 🎯 Success Criteria

- [ ] All TID252 suppressions removed (absolute imports used)
- [ ] All S607 suppressions removed (shutil.which() used)
- [ ] All D401 suppressions removed (docstrings fixed)
- [ ] All F401 suppressions removed (imports fixed)
- [ ] All UP017 suppressions removed (timezone-aware datetime)
- [ ] ANN401 cases documented with recommendations
- [ ] Full toolchain passes (Black, Ruff, Pyright, Pytest)
- [ ] No new noqa suppressions added during fixes
- [ ] Audit report updated with final status
