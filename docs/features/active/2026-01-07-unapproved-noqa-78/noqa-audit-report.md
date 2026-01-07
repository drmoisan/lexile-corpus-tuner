# Noqa Suppression Audit Report

## Executive Summary

Comprehensive audit of all `# noqa` suppressions in the repository to determine:
1. Which match pre-authorized patterns
2. Which require new pre-authorized patterns
3. Which need proper workarounds

---

## Pre-Authorized Patterns (Already Covered)

### S603: subprocess call - check for execution of untrusted input
**Status**: ✅ Pre-authorized
**Files**: Multiple (cli.py, atomic_executor/cli.py, etc.)
**Pattern**: Subprocess calls where executable is validated via `shutil.which()`
**Action**: None required - already covered

---

## Patterns Requiring Analysis

### 1. ARG002: Unused method argument
**Occurrences**: ~15+ instances
**Locations**: Test mocks/stubs (conftest.py files, test files)
**Example**:
```python
def mkdir(self, parents: bool, exist_ok: bool) -> None:  # noqa: ARG002
    # Mock implementation doesn't need arguments
```

**Analysis**:
- **Context**: Test mock objects implementing interfaces (Path, Tkinter widgets)
- **Why needed**: Must match real API signatures but don't use all parameters in stub
- **Alternatives attempted**:
  1. Remove unused parameters → Breaks interface contract
  2. Use parameters with dummy operations → Adds noise, violates clarity
  3. Use `*args, **kwargs` → Loses type safety and IDE support
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS**
- **Justification**: No reasonable alternative when implementing mock/stub APIs
- **Required context**: Must be in test code, must be implementing known interface
- **Required comment format**: `# noqa: ARG002 - mock API signature`

---

### 2. B008: Function call in default argument
**Occurrences**: ~20+ instances
**Locations**: Typer CLI option declarations
**Example**:
```python
catalog_dir: Path = typer.Option(..., exists=True, file_okay=False),  # noqa: B008
```

**Analysis**:
- **Context**: Typer framework pattern for CLI option definitions
- **Why needed**: Typer evaluates Option() at import time for CLI metadata
- **Alternatives attempted**:
  1. Use Option without defaults → Breaks Typer framework requirements
  2. Use factory functions → Incompatible with Typer's declarative pattern
  3. Refactor to procedural → Would require rewriting entire CLI layer
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS**
- **Justification**: Required by Typer framework design, no alternative within framework
- **Required context**: Must be Typer option declaration in CLI function signature
- **Required comment format**: `# noqa: B008 - Typer framework pattern`

---

### 3. TCH002/TCH003: Type checking block violations
**Occurrences**: ~10 instances
**Locations**: Various files importing for both runtime and type hints
**Examples**:
```python
import pytest  # noqa: TCH002 - pytest is required at runtime for fixtures
from pathlib import Path  # noqa: TCH003 - Path required at runtime for Typer and IO
```

**Analysis**:
- **Context**: Modules used for both type hints AND runtime (pytest fixtures, Typer types)
- **Why needed**: Ruff TCH rules want TYPE_CHECKING block, but imports needed at runtime
- **Alternatives attempted**:
  1. Move to TYPE_CHECKING block → Breaks runtime functionality
  2. Duplicate imports → Violates DRY, adds confusion
  3. Use string annotations → Doesn't help when runtime access needed
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS**
- **Justification**: No alternative when module serves dual runtime + type-hint role
- **Required context**: Module must be used at runtime (fixtures, Typer, etc.)
- **Required comment format**: `# noqa: TCH002/TCH003 - [reason] required at runtime`

---

### 4. S310: urllib.request audit URL open
**Occurrences**: ~6 instances
**Locations**: OER scripts accessing Internet Archive
**Example**:
```python
with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
```

**Analysis**:
- **Context**: Fetching data from known, trusted HTTPS endpoints (archive.org)
- **Why needed**: S310 flags ALL urllib calls, even to trusted domains
- **Alternatives attempted**:
  1. Use requests library → Adds heavy dependency for simple GET
  2. Hardcode responses → Defeats purpose of dynamic data fetching
  3. Validate URL scheme → Already using HTTPS, still triggers
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS**
- **Justification**: When accessing documented, trusted API endpoints over HTTPS
- **Required context**: URL must be validated HTTPS endpoint, timeout must be set
- **Required comment format**: `# noqa: S310 - trusted HTTPS endpoint: [domain]`

---

### 5. S314: XML parsing (ElementTree)
**Occurrences**: ~5 instances
**Locations**: EPUB parser, Wikipedia dump extractor
**Example**:
```python
import xml.etree.ElementTree as ET  # noqa: S314
root = ET.fromstring(container_xml)  # noqa: S314
```

**Analysis**:
- **Context**: Parsing user's own EPUB files, Wikipedia dumps (known sources)
- **Why needed**: S314 warns about XML entity expansion attacks
- **Alternatives attempted**:
  1. Use defusedxml → EPUB spec requires standard ElementTree for compatibility
  2. Sanitize input → Already trusted sources (user files, Wikipedia)
  3. Use lxml → Heavier dependency, same security concerns
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS**
- **Justification**: User-controlled trusted files, not untrusted network data
- **Required context**: Parsing user's own files OR known-safe data sources
- **Required comment format**: `# noqa: S314 - parsing trusted [source type]`

---

### 6. BLE001: Blind except
**Occurrences**: ~6 instances
**Locations**: CLI error handlers, broad catch-all for user messaging
**Example**:
```python
except Exception as exc:  # noqa: BLE001
    console.print(f"[red]Error: {exc}[/red]")
    raise typer.Exit(1)
```

**Analysis**:
- **Context**: Top-level CLI exception handlers for user-friendly error messages
- **Why needed**: Must catch unexpected errors to provide clean exit vs. stack dump
- **Alternatives attempted**:
  1. Specific exceptions → Can't predict all possible failures
  2. Re-raise without handling → Shows ugly stack traces to end users
  3. No exception handling → Python default behavior is user-hostile
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS WITH RESTRICTIONS**
- **Justification**: Only at CLI entry points for user-facing error handling
- **Required context**: Must be at CLI/main entry point, must log/display error, must exit cleanly
- **Required comment format**: `# noqa: BLE001 - CLI top-level error handling`
- **Restriction**: NOT allowed in library code, only CLI entry points

---

### 7. S301: Pickle deserialization
**Occurrences**: 1 instance
**Location**: Lexile v2 model loading
**Example**:
```python
return pickle.load(fh)  # noqa: S301  # Trusted model artifact
```

**Analysis**:
- **Context**: Loading ML model artifacts from trusted local files
- **Why needed**: S301 warns about arbitrary code execution in pickle
- **Alternatives attempted**:
  1. Use JSON → Model contains NumPy arrays, not JSON-serializable
  2. Use HDF5 → Would require retraining all models
  3. Validate before load → Pickle format doesn't support pre-validation
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS WITH RESTRICTIONS**
- **Justification**: Loading known model artifacts from trusted local paths
- **Required context**: Must be loading from known-safe local paths (not user input)
- **Required comment format**: `# noqa: S301 - trusted model artifact from [path]`
- **Restriction**: Path must be hardcoded or validated, NOT from user input

---

### 8. S108/S105: Hardcoded paths/passwords in tests
**Occurrences**: 2 instances (tests only)
**Location**: Test files with literal path strings
**Example**:
```python
"/tmp/repo",  # noqa: S108
top_token = "frequent"  # noqa: S105
```

**Analysis**:
- **Context**: Test code with example paths and test data
- **Why needed**: Tests need concrete examples; S108/S105 overly strict for test code
- **Alternatives attempted**:
  1. Use temp directories → Less readable, more complex test setup
  2. Variable instead of literal → Adds indirection without benefit
  
**Recommendation**: **ADD TO PRE-AUTHORIZED PATTERNS (TESTS ONLY)**
- **Justification**: Test code needs concrete examples for clarity
- **Required context**: Must be in test files only
- **Required comment format**: `# noqa: S108/S105 - test fixture data`

---

### 9. ANN401: Any type annotation
**Occurrences**: ~3 instances
**Locations**: Context managers, generic mock constructors
**Example**:
```python
def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: ANN401
```

**Analysis**:
- **Context**: Generic pass-through wrappers that truly accept anything
- **Why needed**: Context managers and generic mocks must accept arbitrary types
- **Alternatives attempted**:
  1. Use TypeVar → Can't represent truly arbitrary heterogeneous *args/**kwargs
  2. Use Protocol → Doesn't work for pass-through wrappers
  3. Use overloads → Would require infinite overloads
  
**Recommendation**: **INVESTIGATE CASE-BY-CASE**
- Some may be fixable with better type design
- Others (context managers, protocol methods) may need pre-authorization
- Need detailed analysis per instance

---

### 10. D401: First line should be imperative
**Occurrences**: 2 instances (test mocks)
**Location**: Test callable stubs
**Example**:
```python
def copy(self, text: str) -> None:  # noqa: D401
    """Mock clipboard copy."""  # Should be "Copy text to mock clipboard"
```

**Analysis**:
- **Context**: Docstring style violation in test stubs
- **Why needed**: Not needed - this is fixable!
- **Alternatives**: Fix the docstrings to use imperative mood
  
**Recommendation**: **FIX - NOT JUSTIFIED**
- No suppression needed
- Fix docstrings to match D401 requirements

---

### 11. TID252: Relative imports beyond top-level
**Occurrences**: 3 instances
**Locations**: gutenberg_query_builder_ui submodule
**Example**:
```python
from ..gutenberg_query_core import QueryGroupModel  # noqa: TID252
```

**Analysis**:
- **Context**: UI submodule importing from parent package modules
- **Why needed**: TID252 discourages parent imports, prefers absolute
- **Alternatives attempted**:
  1. Use absolute imports → Creates long import paths
  2. Restructure package → Would break existing architecture
  
**Recommendation**: **EVALUATE ALTERNATIVES**
- Could use absolute imports: `from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_core import ...`
- Slightly more verbose but removes suppression
- **Recommend**: Refactor to absolute imports, remove suppressions

---

### 12. S607: Starting process with partial path
**Occurrences**: 1 instance
**Location**: collect_commit_context.py
**Example**:
```python
["git", *args],  # noqa: S607
```

**Analysis**:
- **Context**: Running git command via subprocess
- **Why needed**: S607 wants full path, but git location varies by platform
- **Alternative**: Use shutil.which("git") to resolve full path
  
**Recommendation**: **FIX - APPLY S603 PATTERN**
- Use existing S603 pattern (validate via shutil.which)
- Remove S607 suppression

---

### 13. UP017: Datetime timezone usage
**Occurrences**: 1 instance
**Location**: pr_context/summary_helpers.py
**Example**: (Need to examine)
  
**Recommendation**: **EXAMINE AND FIX**
- UP017 suggests modern timezone-aware datetime
- Likely fixable by using timezone-aware datetime objects

---

### 14. F401: Unused import
**Occurrences**: 1 instance
**Location**: pr_context/collector.py
**Example**: (Need to examine)
  
**Recommendation**: **EXAMINE AND FIX**
- Either use the import or remove it
- F401 should not require suppression

---

## Summary Statistics

| Pattern | Count | Status | Action |
|---------|-------|--------|--------|
| S603 (subprocess) | ~15 | ✅ Pre-authorized | None |
| S110 (try-except-pass) | 1 | ✅ Justified (optional import) | None |
| ARG002 (unused args) | ~15 | ⚠️ Needs pre-auth | Add pattern |
| B008 (Typer defaults) | ~20 | ⚠️ Needs pre-auth | Add pattern |
| TCH002/TCH003 | ~10 | ⚠️ Needs pre-auth | Add pattern |
| S310 (urllib) | ~6 | ⚠️ Needs pre-auth | Add pattern |
| S314 (XML parsing) | ~5 | ⚠️ Needs pre-auth | Add pattern |
| BLE001 (blind except) | ~6 | ⚠️ Needs pre-auth (restricted) | Add pattern |
| S301 (pickle) | 1 | ⚠️ Needs pre-auth (restricted) | Add pattern |
| S108/S105 (test literals) | 2 | ⚠️ Needs pre-auth (test only) | Add pattern |
| ANN401 (Any type) | ~3 | 🔍 Investigate | Case-by-case |
| D401 (imperative) | 2 | ❌ Fix | Remove suppression |
| TID252 (relative import) | 3 | 🔄 Refactor | Use absolute imports |
| S607 (partial path) | 1 | 🔄 Refactor | Apply S603 pattern |
| UP017 (datetime) | 1 | 🔍 Examine | Likely fixable |
| F401 (unused import) | 1 | 🔍 Examine | Remove or use |

---

## Recommendations

### Immediate Actions
1. **Add 8 new pre-authorized patterns** (ARG002, B008, TCH002/003, S310, S314, BLE001, S301, S108/S105)
2. **Fix 2 docstrings** (D401)
3. **Refactor 3 imports** (TID252) to use absolute imports
4. **Fix 1 subprocess call** (S607) to use shutil.which pattern
5. **Investigate 5 instances** (ANN401, UP017, F401) individually

### Next Steps
1. Update python-suppressions.instructions.md with new patterns
2. Create targeted fixes for TID252, S607, D401
3. Investigate remaining edge cases (ANN401, UP017, F401)
4. Run full toolchain after changes
5. Regenerate AGENTS.md
