# Agent 4: Utilities & Gap Filling

**Mission:** Complete test coverage for utility modules and fill gaps in existing tests to reach 90%+

**Assigned Modules:**
- `src/lexile_corpus_tuner/textutils.py` (44% → 90%+)
- `src/lexile_corpus_tuner/frequency_loader.py` (50% → 90%+)
- `src/lexile_corpus_tuner/config.py` (84% → 95%+)
- `src/lexile_corpus_tuner/corpus/cli.py` (68% → 90%+)
- `src/lexile_corpus_tuner/analyzer/cli.py` (46% → 90%+)
- `src/lexile_corpus_tuner/text_difficulty_pipeline.py` (0% → 90%+ if applicable)

**Test Files to Create/Extend:**
- `tests/test_textutils.py` (NEW)
- `tests/test_frequency_loader.py` (NEW)
- `tests/test_config_advanced.py` (NEW - extends existing config tests)
- `tests/test_corpus_cli.py` (NEW)
- `tests/test_analyzer_cli.py` (NEW)

---

## Loading Prompt

```
You are Agent 4 in a coordinated test coverage expansion effort for the lexile-corpus-tuner project. Your mission is to complete test coverage for utility modules and fill gaps in well-tested modules, bringing overall coverage to 90%+.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1. **Standards Compliance (NON-NEGOTIABLE):**
   - Every test MUST follow the unit-test-policy.md exactly
   - Every test MUST follow code-change.instructions.md exactly
   - NO EXCEPTIONS without explicit user approval
   - If you cannot meet a standard, STOP and ask for guidance

2. **Before Writing ANY Code:**
   - Read docs/test-coverage-expansion-plan.md
   - Read docs/agent-4-instructions.md (this file)
   - Read .github/unit-test-policy.md
   - Read docs/code-change.instructions.md
   - Review existing test patterns in tests/

3. **Your Assignment:**
   - Modules: textutils.py, frequency_loader.py, config.py (gaps), corpus/cli.py, analyzer/cli.py
   - Create: 5 new test files with comprehensive coverage
   - Target: 90%+ coverage for each assigned module
   - PRIORITY: Complete textutils.py FIRST (other agents depend on it)

4. **Work Process (MANDATORY):**
   a. Start with textutils.py (highest priority)
   b. Read the source module thoroughly
   c. Identify all functions, classes, and code paths
   d. Design test cases covering:
      - Happy path (valid inputs)
      - Edge cases (None, empty, invalid types)
      - Type validation
      - Unicode and special character handling
   e. Write tests following AAA pattern (Arrange-Act-Assert)
   f. Run quality checks IN ORDER:
      - black .
      - ruff check
      - pyright
      - pytest (all tests must pass)
      - pytest --cov=src/lexile_corpus_tuner (verify coverage increase)
   g. Fix any failures and repeat step f until ALL pass
   h. Document progress in test-coverage-expansion-plan.md

5. **Coordination (CRITICAL):**
   - Complete textutils.py tests FIRST (others depend on it)
   - Notify when textutils.py is complete so others can proceed
   - Your modules are dependencies for other agents

6. **Quality Enforcement:**
   - If ANY check fails, you MUST fix it before proceeding
   - Do NOT skip checks or commit failing code
   - Do NOT create incomplete tests
   - Every test must be deterministic and fast (<1s per test)

7. **Failure Modes to AVOID:**
   - Completing low-priority modules before textutils.py
   - Writing tests that don't cover edge cases
   - Missing type validation tests
   - Not testing error conditions

**Start by confirming:**
1. You've read all required documents
2. You understand textutils.py is the top priority
3. You're ready to begin with textutils.py

Do NOT write any code until you confirm understanding.
```

---

## Detailed Task List

### Phase 1: Foundation Tests (PRIORITY ORDER)

#### Task 4.1: `test_textutils.py` 🔴 HIGHEST PRIORITY
**Module:** `textutils.py`  
**Current Coverage:** 44%  
**Missing Lines:** 15-20, 25-27

**⚠️ COMPLETE THIS FIRST - OTHER AGENTS DEPEND ON IT**

**Test Cases Required:**

1. **Test `normalize_text` - Type Coercion**
   - Input: string → returns normalized string
   - Input: int → converts to string
   - Input: float → converts to string
   - Input: None → returns empty string
   - Input: object with __str__ → converts correctly

2. **Test `normalize_text` - Whitespace Normalization**
   - Multiple spaces → single space
   - Tabs and newlines → single space
   - Leading/trailing whitespace → stripped
   - Unicode whitespace characters → normalized

3. **Test `normalize_text` - Unicode Handling**
   - Non-ASCII characters → preserved correctly
   - Emojis → handled gracefully
   - Mixed encodings → no errors

4. **Test `normalize_text` - Edge Cases**
   - Empty string → returns empty string
   - String with only whitespace → returns empty string
   - Very long string → performance acceptable

5. **Test `iter_tokens` - Token Iteration**
   - Simple text → yields words
   - Text with punctuation → tokens correct
   - Empty text → yields nothing
   - Unicode text → tokens correct

6. **Test `iter_tokens` - Generator Behavior**
   - Verify it's a generator (not list)
   - Verify lazy evaluation
   - Verify can be consumed multiple times

**Mocking Strategy:**
- None required (pure functions)

**Success Criteria:**
- 100% coverage for both functions
- All edge cases tested
- Type validation verified
- Unicode handling verified
- Tests run fast (<0.5s total)

---

#### Task 4.2: `test_frequency_loader.py`
**Module:** `frequency_loader.py`  
**Current Coverage:** 50%  
**Missing Lines:** 25-43

**Test Cases Required:**

1. **Test `load_frequency_table` - Successful Load**
   - Create mock Parquet file (use tmp_path)
   - Verify table loaded as dict
   - Verify word → frequency mapping correct

2. **Test `load_frequency_table` - File Formats**
   - Test with Parquet file
   - Test with CSV file (if supported)
   - Test format detection

3. **Test `load_frequency_table` - Error Handling**
   - Missing file → exception
   - Corrupted file → exception
   - Empty file → empty dict or exception
   - Invalid schema → exception

4. **Test `load_frequency_table` - Data Validation**
   - Verify frequencies are positive integers
   - Verify words are strings
   - Test handling of duplicates

**Mocking Strategy:**
```python
@pytest.fixture
def sample_frequency_table(tmp_path):
    """Create sample frequency table file."""
    import pandas as pd
    
    df = pd.DataFrame({
        "word": ["the", "a", "test"],
        "frequency": [100000, 50000, 1000]
    })
    
    freq_file = tmp_path / "frequencies.parquet"
    df.to_parquet(freq_file)
    return freq_file
```

**Success Criteria:**
- Coverage ≥90%
- File loading verified
- Error cases handled
- Tests use tmp_path
- Tests run fast (<1s total)

---

#### Task 4.3: `test_config_advanced.py`
**Module:** `config.py`  
**Current Coverage:** 84%  
**Missing Lines:** 65-70, 75-77, 83, 92, 98, 104

**Test Cases Required:**

1. **Test `_build_kwargs` - Edge Cases**
   - Test with all None values
   - Test with empty dict
   - Test with invalid keys → filtered out
   - Test with type mismatches → handled

2. **Test `_build_openai_settings` - Settings Construction**
   - Test with all fields provided
   - Test with minimal fields
   - Test with invalid settings → exception

3. **Test `config_from_dict` - Error Handling**
   - Test with invalid dict structure → exception
   - Test with missing required fields → defaults
   - Test with extra fields → ignored

4. **Test `config_from_yaml` - Error Paths**
   - Test file read errors
   - Test YAML parse errors
   - Test schema validation errors

5. **Test `load_config` - Path Resolution**
   - Test with absolute path
   - Test with relative path
   - Test with non-existent path → exception

**Mocking Strategy:**
```python
@pytest.fixture
def invalid_config_yaml(tmp_path):
    """Create invalid config YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: [yaml: content")  # Malformed
    return config_file

@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dict."""
    return {
        "window_size": 500,
        "stride": 250,
    }
```

**Success Criteria:**
- Coverage ≥95%
- All error paths tested
- Edge cases covered
- Tests run fast (<1s total)

---

#### Task 4.4: `test_corpus_cli.py`
**Module:** `corpus/cli.py`  
**Current Coverage:** 68%  
**Missing Lines:** 22-25, 38, 44

**Test Cases Required:**

1. **Test `corpus_group` Command Group**
   - Verify subcommands registered
   - Test `--help` output

2. **Test `corpus_download` Command**
   - Use CliRunner to invoke
   - Mock download functions
   - Verify exit code 0
   - Test error handling → exit code 1

3. **Test `corpus_normalize` Command**
   - Use CliRunner to invoke
   - Mock normalization functions
   - Use tmp_path for input/output
   - Verify command completes

4. **Test `corpus_frequencies` Command**
   - Use CliRunner to invoke
   - Mock frequency computation
   - Verify output file created

**Mocking Strategy:**
```python
from typer.testing import CliRunner

@pytest.fixture
def corpus_cli_runner():
    """CLI runner for corpus commands."""
    return CliRunner()

@pytest.fixture
def mock_corpus_operations(monkeypatch):
    """Mock corpus download/normalize/frequencies."""
    def fake_download(*args, **kwargs):
        pass  # No-op
    
    monkeypatch.setattr(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download.download_gutenberg_subset",
        fake_download
    )
    # Mock other operations
```

**Success Criteria:**
- Coverage ≥90%
- All CLI commands tested
- Tests use CliRunner
- Tests run fast (<2s total)

---

#### Task 4.5: `test_analyzer_cli.py`
**Module:** `analyzer/cli.py`  
**Current Coverage:** 46%  
**Missing Lines:** 41-85

**Test Cases Required:**

1. **Test `analyze_group` Command Group**
   - Verify subcommands registered
   - Test `--help` output

2. **Test `analyze_text` Command - Success Path**
   - Use CliRunner to invoke
   - Mock feature computation
   - Mock model loading
   - Verify Lexile output printed

3. **Test `analyze_text` Command - Error Handling**
   - Test missing model file → error
   - Test invalid text → error
   - Test missing frequency table → error

4. **Test CLI Argument Validation**
   - Test required arguments
   - Test optional arguments
   - Test invalid argument combinations

**Mocking Strategy:**
```python
@pytest.fixture
def mock_analyzer(monkeypatch):
    """Mock analyzer components."""
    def fake_compute_features(text, freq_table):
        return Mock(word_count=100, unique_words=50)
    
    def fake_estimate(features):
        return 350.0  # Mock Lexile
    
    monkeypatch.setattr(
        "lexile_corpus_tuner.lexile_scoring_model.analyzer.features.compute_document_features",
        fake_compute_features
    )
    monkeypatch.setattr(
        "lexile_corpus_tuner.lexile_scoring_model.analyzer.model.estimate_lexile_from_features",
        fake_estimate
    )
```

**Success Criteria:**
- Coverage ≥90%
- All CLI commands tested
- Error paths covered
- Tests use CliRunner
- Tests run fast (<2s total)

---

## Phase 2: Gap Filling

### Task 4.6: Review Overall Coverage
- Run coverage report for entire project
- Identify modules still below 90%
- Prioritize gaps in critical path modules

### Task 4.7: Add Missing Tests
- Focus on uncovered error handling paths
- Add edge case tests to existing modules
- Improve fixture reusability

### Task 4.8: Cross-Module Integration
- Create shared fixtures in conftest.py
- Document fixture usage patterns
- Ensure fixtures are efficient

---

## Phase 3: Polish

### Task 4.9: Documentation
- Add comprehensive docstrings to all tests
- Document any complex mocking strategies
- Update test README if needed

### Task 4.10: Final Verification
- Run full test suite
- Verify ≥90% overall coverage
- Verify all quality gates pass
- Verify no flaky tests

---

## Shared Fixtures to Create (in `tests/conftest.py`)

```python
@pytest.fixture
def sample_text():
    """Simple text for testing utilities."""
    return "The quick brown fox jumps over the lazy dog."

@pytest.fixture
def unicode_text():
    """Text with Unicode characters for testing."""
    return "Café résumé 日本語 emoji: 🎉"

@pytest.fixture
def mock_frequency_dict():
    """Small frequency dictionary for tests."""
    return {
        "the": 100000,
        "quick": 5000,
        "brown": 3000,
        "fox": 1000,
    }
```

---

## Success Checklist

Before marking any task complete:
- [ ] Read source module completely
- [ ] Identified all code paths
- [ ] Wrote tests for happy path
- [ ] Wrote tests for edge cases (None, empty, invalid)
- [ ] Wrote tests for error cases
- [ ] Wrote tests for type validation
- [ ] All tests have docstrings
- [ ] Ran `black .` (passes)
- [ ] Ran `ruff check` (passes)
- [ ] Ran `pyright` (passes)
- [ ] Ran `pytest` (all pass)
- [ ] Ran `pytest --cov` (coverage increased)
- [ ] Verified coverage ≥90% for module
- [ ] Verified tests run fast (<1s per test)
- [ ] Updated test-coverage-expansion-plan.md
- [ ] Notified other agents if completing dependency

---

## Reporting Template

```markdown
### Agent 4 Update - YYYY-MM-DD

**Phase:** [1/2/3]  
**Tasks Completed:**
- test_textutils.py ✅ (PRIORITY COMPLETE - notified other agents)
- test_frequency_loader.py ✅
- test_config_advanced.py 🔄

**Coverage Changes:**
- textutils.py: 44% → 100% ✅
- frequency_loader.py: 50% → 94%
- config.py: 84% → 89% (in progress)

**Blockers:**
- None

**Next Steps:**
- Complete test_config_advanced.py
- Begin test_corpus_cli.py
```

---

## Priority Matrix

1. **URGENT (Day 1):** `textutils.py` - Blocks other agents
2. **HIGH (Day 2):** `frequency_loader.py` - Used by multiple modules
3. **MEDIUM (Day 3):** `config.py` - Already well-tested, just filling gaps
4. **MEDIUM (Day 4-5):** CLI modules - Independent work
5. **LOW (Day 6+):** Gap filling and polish

---

## Common Pitfalls to Avoid

1. **Not prioritizing textutils.py**
   - ❌ Starting with easier modules first
   - ✅ Complete textutils.py before anything else

2. **Incomplete edge case testing**
   - ❌ Only testing happy path
   - ✅ Test None, empty, invalid types

3. **Not testing type coercion**
   - ❌ Assuming inputs are always correct type
   - ✅ Test with various input types

4. **CLI tests without proper mocking**
   - ❌ CLI tests that call real functions
   - ✅ All operations mocked

5. **Slow tests due to unnecessary setup**
   - ❌ Creating large fixtures for simple tests
   - ✅ Minimal fixtures for each test

---

## Emergency Contacts

If blocked or uncertain:
1. Review existing utility tests for patterns
2. Consult `unit-test-policy.md` for specific guidance
3. Check `code-change.instructions.md` for code standards
4. Report blocker in test-coverage-expansion-plan.md
5. Request guidance from coordinator

**Remember: You're completing the foundation. Other agents depend on you. Prioritize accordingly.**


