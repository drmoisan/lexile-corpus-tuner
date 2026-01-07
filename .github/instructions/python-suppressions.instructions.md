---
applyTo: "**/*.py"
name: python-suppressions-policy
description: "Pre-authorized patterns for # noqa and # type: ignore suppressions in Python code"
---

# Pre-Authorized Suppression Patterns

This policy defines the **only** patterns of `# noqa` and `# type: ignore` suppressions that are pre-authorized for use in Python code without explicit user approval.

**Authorization requirement:**
- All `# noqa` and `# type: ignore` suppressions must either:
  1. **Match a pre-authorized pattern** defined in this file, OR
  2. **Have explicit user approval** for that specific suppression

**If you encounter an error that seems to require a suppression not matching a pre-authorized pattern:**
1. First, attempt to resolve it without a suppression (refactor, restructure, use approved patterns)
2. If that fails, try at least five more distinct approaches
3. Continue iterating until you solve the problem or demonstrate why each approach fails
4. Only after multiple documented failed attempts may you request user approval, providing:
   - The specific rule/error and diagnostic code
   - Each approach you tried and why it failed
   - Why a suppression is the only remaining option

---

## Ruff Suppressions

### S603: subprocess call - check for execution of untrusted input

**When pre-authorized:**  
Subprocess calls where the executable is validated via `shutil.which()` before use.

**Required pattern:**
```python
# Validate executable exists and resolve full path
exe = shutil.which("tool_name")
if not exe:
    raise FileNotFoundError("Required executable not found on PATH: tool_name")

# Use validated executable in subprocess call
subprocess.run([exe, ...])  # noqa: S603 - static analysis can't verify runtime validation
```

**Required comment format:**  
`# noqa: S603 - static analysis can't verify runtime validation`

**Justification:**  
Cross-platform compatibility requires runtime PATH resolution via `shutil.which()`. Static analysis cannot trace the runtime validation, but the code is safe because:
1. The executable path is resolved from PATH (not user input)
2. We verify it exists before use
3. Hardcoding platform-specific paths like `/usr/bin/git` or `C:\\Program Files\\Git\\bin\\git.exe` would break portability

**Examples:**
- Git operations: `git_exe = shutil.which("git")`
- Clipboard commands: `clip_exe = shutil.which("pbcopy")`
- Any system tool resolved from PATH

---

## Pyright Suppressions

### import-untyped: Cannot access member for module with unknown type

**When pre-authorized:**  
Optional third-party dependencies that lack type stubs or `py.typed` marker.

**Required pattern:**
```python
try:
    import untyped_library  # type: ignore[import-untyped]
    # Use library...
except ImportError:
    # Graceful fallback when library not installed
    pass
```

**Required context:**
- Import must be in a try/except ImportError block
- Library must be optional (not in core dependencies)
- No type stubs available (checked via typeshed or types-* packages)
- Library lacks `py.typed` marker (required by PEP 561)

**Justification:**  
Optional dependencies may not have type stubs or proper PEP 561 type markers. Rather than exclude entire files from type checking, we use targeted suppressions on the import line while wrapping usage in properly typed adapter functions.

**Examples:**
- `pyperclip` (has inline type hints but lacks `py.typed` marker)
- `tkinter` (stdlib but excluded from type checking, no stubs)
- Platform-specific optional libraries

---

### ARG002: Unused method argument

**When pre-authorized:**  
Test mock/stub implementations that must match interface signatures but don't use all parameters.

**Required pattern:**
```python
class MockPath:
    def mkdir(self, parents: bool, exist_ok: bool) -> None:  # noqa: ARG002 - mock API signature
        """Mock mkdir that doesn't need arguments."""
        self._created = True
```

**Required context:**
- Must be in test code (tests/ directory)
- Must be implementing a known interface (Path, Tkinter widgets, etc.)
- Cannot use the parameters without defeating the purpose of the mock

**Required comment format:**  
`# noqa: ARG002 - mock API signature` or `# noqa: ARG002 - match [InterfaceName] API`

**Justification:**  
Test mocks must match real API signatures for type safety and IDE support, but stub implementations often don't need all parameters. Alternatives (removing parameters, using *args/**kwargs) break type safety.

**Examples:**
- Mock Path.mkdir(parents, exist_ok)
- Mock Tkinter widget constructors
- Protocol method stubs in tests

---

### B008: Function call in default argument

**When pre-authorized:**  
Typer CLI option declarations where Option() must be evaluated at import time.

**Required pattern:**
```python
def cli_command(
    input_file: Path = typer.Option(..., exists=True),  # noqa: B008 - Typer framework pattern
    verbose: bool = typer.Option(False, help="Enable verbose output"),  # noqa: B008 - Typer framework pattern
) -> None:
    """CLI command using Typer options."""
```

**Required context:**
- Must be Typer option declaration in CLI function signature
- Typer framework requires evaluation at import time for CLI metadata
- No alternative within Typer's declarative pattern

**Required comment format:**  
`# noqa: B008 - Typer framework pattern`

**Justification:**  
Typer's declarative CLI pattern evaluates Option() at import time to build CLI metadata. This is framework design, not a code smell. Alternative (procedural approach) would require rewriting entire CLI layer.

**Examples:**
- typer.Option() in function signatures
- typer.Argument() in function signatures

---

### TCH002/TCH003: Type checking block violations

**When pre-authorized:**  
Modules used for both runtime and type hints (pytest fixtures, Typer type hints, etc.).

**Required pattern:**
```python
import pytest  # noqa: TCH002 - pytest required at runtime for fixtures

from pathlib import Path  # noqa: TCH003 - Path required at runtime for Typer and IO
```

**Required context:**
- Module must be used at runtime (fixtures, Typer CLI, runtime isinstance checks)
- Cannot move to TYPE_CHECKING block without breaking functionality
- Not just for type hints

**Required comment format:**  
`# noqa: TCH002 - [module] required at runtime for [reason]`  
`# noqa: TCH003 - [module] required at runtime for [reason]`

**Justification:**  
Some modules serve dual roles: type hints AND runtime functionality. Moving to TYPE_CHECKING block breaks runtime behavior. Duplicating imports violates DRY.

**Examples:**
- pytest (fixtures, marks, decorators)
- Path (Typer CLI types + file operations)
- collections.abc (runtime Protocol checks + type hints)

---

### S310: Audit URL open with urllib

**When pre-authorized:**  
Accessing documented, trusted HTTPS API endpoints with timeout.

**Required pattern:**
```python
req = urllib.request.Request(  # noqa: S310 - trusted HTTPS endpoint: archive.org
    "https://archive.org/metadata/identifier"
)
with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - trusted HTTPS endpoint: archive.org
    data = resp.read()
```

**Required context:**
- URL must be validated HTTPS endpoint
- Domain must be documented trusted source (archive.org, pypi.org, etc.)
- Timeout must be set
- Not user-provided URLs

**Required comment format:**  
`# noqa: S310 - trusted HTTPS endpoint: [domain]`

**Justification:**  
S310 flags ALL urllib calls indiscriminately. When accessing well-known, documented HTTPS APIs with timeouts, the security risk is minimal. Using requests library adds heavy dependency for simple GETs.

**Examples:**
- Internet Archive API
- PyPI JSON API
- GitHub API with known endpoints

---

### S314: XML parsing with ElementTree

**When pre-authorized:**  
Parsing user's own files or known-safe data sources (not untrusted network data).

**Required pattern:**
```python
import xml.etree.ElementTree as ET  # noqa: S314 - parsing trusted user EPUB files

root = ET.fromstring(container_xml)  # noqa: S314 - parsing trusted user EPUB files
```

**Required context:**
- Parsing user's own local files (EPUB, configuration)
- Parsing known-safe sources (Wikipedia dumps, curated datasets)
- NOT parsing untrusted network data
- EPUB spec requires standard ElementTree for compatibility

**Required comment format:**  
`# noqa: S314 - parsing trusted [source type]`

**Justification:**  
S314 warns about XML entity expansion attacks from untrusted sources. User's own files and curated datasets are trusted. EPUB spec requires standard ElementTree. defusedxml incompatible with EPUB parsing requirements.

**Examples:**
- EPUB file parsing (user's own books)
- Wikipedia XML dump processing
- Configuration file parsing

---

### BLE001: Blind except (CLI entry points ONLY)

**When pre-authorized:**  
Top-level CLI exception handlers for user-friendly error messages and clean exits.

**Required pattern:**
```python
def main() -> None:
    """CLI entry point."""
    try:
        # CLI logic here
        run_pipeline()
    except Exception as exc:  # noqa: BLE001 - CLI top-level error handling
        console.print(f"[red]Error: {exc}[/red]")
        logger.exception("Pipeline failed")
        raise typer.Exit(1)
```

**Required context:**
- Must be at CLI entry point (main, CLI command function)
- Must log or display error with context
- Must exit cleanly (not re-raise without handling)
- NOT allowed in library/internal code

**Required comment format:**  
`# noqa: BLE001 - CLI top-level error handling`

**Justification:**  
CLI tools must provide user-friendly error messages instead of stack traces. Cannot predict all possible exception types. This is ONLY for user-facing CLI, NOT library code.

**Restriction:**  
- **ONLY at CLI entry points**
- NOT in internal/library functions
- NOT in test code
- Must include error logging/display

**Examples:**
- Typer command entry points
- Script main() functions
- CLI error wrapper functions

---

### S301: Pickle deserialization (restricted)

**When pre-authorized:**  
Loading known model artifacts from hardcoded trusted local paths.

**Required pattern:**
```python
# Model path is hardcoded, not from user input
MODEL_PATH = Path(__file__).parent / "artifacts" / "lexile_model.pkl"

def load_model() -> Model:
    """Load pre-trained model from trusted artifact."""
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - trusted model artifact from hardcoded path
```

**Required context:**
- Path must be hardcoded or validated (not from user input/CLI args)
- Loading known model artifacts from trusted local paths
- Not deserializing user-provided pickle files

**Required comment format:**  
`# noqa: S301 - trusted model artifact from hardcoded path`

**Justification:**  
ML models contain NumPy arrays not serializable to JSON. HDF5 would require retraining all models. Pickle format doesn't support pre-validation. Safe when loading from known paths.

**Restriction:**  
- Path MUST be hardcoded or validated before use
- NOT from user input/command-line arguments
- Only for ML model/artifact loading

**Examples:**
- Loading pre-trained ML models
- Loading tokenizer artifacts
- Loading vocabulary caches

---

### S108/S105: Hardcoded paths/passwords (tests ONLY)

**When pre-authorized:**  
Test fixtures with example paths and test data literals.

**Required pattern:**
```python
def test_path_handling():
    """Test with concrete example path."""
    result = process_path("/tmp/test_repo")  # noqa: S108 - test fixture path
    assert result.exists()

def test_token_parsing():
    """Test with literal test data."""
    token = "sample_token_string"  # noqa: S105 - test fixture data
    assert parse_token(token) == expected
```

**Required context:**
- Must be in test code only
- Literal paths/strings for test clarity
- Not actual secrets or production paths

**Required comment format:**  
`# noqa: S108 - test fixture path`  
`# noqa: S105 - test fixture data`

**Justification:**  
Test code needs concrete examples for readability. Using temp directories or variables adds complexity without benefit. These are not real paths or secrets.

**Restriction:**  
- **ONLY in test files** (tests/ directory)
- Not in production code

**Examples:**
- Example paths in test assertions
- Test token/string literals
- Mock credential strings in tests

---

## Non-authorized Patterns (Explicitly Prohibited - with Workarounds)

Beyond the S110 pattern documented earlier, the following patterns are **NOT** pre-authorized. Use the documented workarounds instead.

### TID252: Relative imports beyond top-level package - NOT AUTHORIZED

**Why NOT authorized:**  
Parent-relative imports (`from ..module import`) reduce code clarity and create coupling.

**Recommended alternative pattern:**
```python
# Instead of:
from ..gutenberg_query_core import QueryGroupModel  # noqa: TID252

# Use absolute imports:
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_core import (
    QueryGroupModel,
)
```

**Why this is better:**
- Explicit full path shows exact module location
- Works regardless of execution context
- Better IDE support and refactoring tools
- Clearer for code readers

---

### S607: Starting process with partial executable path - NOT AUTHORIZED

**Why NOT authorized:**  
Using partial paths like `"git"` instead of full paths creates security risks.

**Recommended alternative pattern:**
```python
import shutil
import subprocess

# Validate executable exists and get full path
git_exe = shutil.which("git")
if not git_exe:
    raise FileNotFoundError("git not found on PATH")

# Use validated full path
result = subprocess.run(
    [git_exe, "status"],  # noqa: S603 - static analysis can't verify runtime validation
    capture_output=True,
    text=True,
    check=True,
)
```

**Why this is better:**
- Validates executable exists before use
- Uses full path from PATH resolution
- Clear error if executable not found
- Follows S603 pre-authorized pattern

---

### D401: First line should be in imperative mood - NOT AUTHORIZED

**Why NOT authorized:**  
Docstring style rules are not technical limitations, just formatting preferences.

**Recommended alternative pattern:**
```python
# Instead of:
def copy(self, text: str) -> None:  # noqa: D401
    """Mock clipboard copy."""

# Use imperative mood:
def copy(self, text: str) -> None:
    """Copy text to mock clipboard."""
```

**Why this is better:**
- Follows PEP 257 docstring conventions
- More readable and consistent
- No technical reason for suppression

---

### F401: Unused import - NOT AUTHORIZED

**Why NOT authorized:**  
Unused imports should be removed or used, not suppressed.

**Recommended alternative pattern:**
```python
# Instead of:
from typing import Optional  # noqa: F401

# Either use it:
def func() -> Optional[str]:
    return None

# Or remove it:
# (deleted)

# For re-exports, use __all__:
from .module import Symbol  # Used for re-export
__all__ = ["Symbol"]
```

**Why this is better:**
- Cleaner code
- Faster import times
- Clear signal of what's actually used

---

### UP017: Datetime without timezone - NOT AUTHORIZED

**Why NOT authorized:**  
Modern Python supports timezone-aware datetime; naive datetime causes bugs.

**Recommended alternative pattern:**
```python
from datetime import datetime, timezone

# Instead of:
now = datetime.now()  # noqa: UP017

# Use timezone-aware datetime:
now = datetime.now(timezone.utc)
```

**Why this is better:**
- Avoids timezone-related bugs
- Explicit about timezone handling
- Modern Python best practice

---

## Policy Enforcement

### Pre-authorized pattern checklist:

Before using a suppression, verify:
- [ ] Pattern **exactly** matches a pre-authorized pattern above
- [ ] Required comment format is used verbatim
- [ ] All contextual requirements are met (validation, fallback chain, try/except, etc.)
- [ ] Code structure matches the documented safe pattern

### Requesting new pre-authorized patterns:

If you encounter a recurring pattern that should be pre-authorized:
1. Document the pattern with full justification
2. Show why it's deterministic and can be codified
3. Propose the required comment format
4. Request user approval to add to this file

### Audit checklist:

When reviewing code:
- [ ] All suppressions either match pre-authorized patterns OR have documented user approval
- [ ] Comment format matches required format exactly
- [ ] No suppressions are broader than necessary (file-level vs. line-level)
- [ ] Justifications are clear and reference this policy

---

## Non-authorized Patterns (Explicitly Prohibited)

The following are **NOT** pre-authorized and require case-by-case approval:
- File-level suppressions (e.g., adding paths to `pyproject.toml` ignores)
- Broad exception catching without validation (`subprocess.run([user_input, ...])  # noqa: S603`)
- Disabling security rules for convenience without justification
- Using `# noqa` or `# type: ignore` as a shortcut to avoid fixing legitimate issues

### S110: try-except-pass fallback chains - NOT AUTHORIZED

**Why NOT authorized:**  
Try-except-pass fallback chains often hide lazy design. If you know the correct method at design time (platform detection, environment variables, shutil.which() validation), you should implement explicit detection instead of relying on exception-based control flow.

**Recommended alternative pattern:**
```python
import shutil
import sys
from functools import lru_cache

@lru_cache(maxsize=1)
def get_clipboard_command() -> str | None:
    """
    Detect the correct clipboard command for the current platform.
    
    Returns:
        Command name if available, None if no clipboard support detected.
    
    Side Effects:
        Caches result after first call for performance.
    """
    # Detect platform
    if sys.platform == "win32":
        candidates = ["clip"]
    elif sys.platform == "darwin":
        candidates = ["pbcopy"]
    else:  # Linux/Unix
        # Check for WSL (reports linux but needs Windows clipboard)
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    candidates = ["clip.exe", "pbcopy", "xclip", "wl-copy"]
                else:
                    candidates = ["xclip", "wl-copy"]
        except FileNotFoundError:
            candidates = ["xclip", "wl-copy"]
    
    # Validate candidates exist on PATH
    for cmd in candidates:
        if shutil.which(cmd):
            return cmd
    
    return None

# Usage
clip_cmd = get_clipboard_command()
if clip_cmd:
    subprocess.run([clip_cmd], input=text, ...)  # noqa: S603 - validated above
else:
    raise RuntimeError("No clipboard command available")
```

**Why this is better:**
- Explicit platform detection makes behavior predictable
- shutil.which() validates availability before use
- Caching avoids repeated detection overhead
- Clear failure mode (exception) instead of silent fallback
- No try-except-pass control flow

**When exception-based fallback IS acceptable:**
- Optional library imports where the library truly may or may not be installed
- Cases where explicit detection is genuinely impossible (not just inconvenient)

**These cases still require explicit user approval with justification.**
