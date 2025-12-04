# Agent 1: Core Lexile Pipeline Testing

**Mission:** Expand test coverage for analyzer and core calibration modules from ~35% to 90%+

**Assigned Modules:**
- `src/lexile_corpus_tuner/analyzer/features.py` (40% → 90%+)
- `src/lexile_corpus_tuner/analyzer/slices.py` (22% → 90%+)
- `src/lexile_corpus_tuner/analyzer/model.py` (40% → 90%+)
- `src/lexile_corpus_tuner/analyzer/adjustments.py` (25% → 90%+)
- `src/lexile_corpus_tuner/calibration/train.py` (38% → 90%+)
- `src/lexile_corpus_tuner/calibration/featureset.py` (33% → 90%+)

**Test Files to Create/Extend:**
- `tests/test_analyzer_features.py` (NEW)
- `tests/test_analyzer_slices.py` (NEW)
- `tests/test_analyzer_model.py` (NEW)
- `tests/test_analyzer_adjustments.py` (NEW)
- `tests/test_calibration_train.py` (NEW)
- `tests/test_calibration_featureset.py` (NEW)

---

## Loading Prompt

```
You are Agent 1 in a coordinated test coverage expansion effort for the lexile-corpus-tuner project. Your mission is to create comprehensive unit tests for the analyzer and core calibration modules, bringing coverage from ~35% to 90%+.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1. **Standards Compliance (NON-NEGOTIABLE):**
   - Every test MUST follow the unit-test-policy.md exactly
   - Every test MUST follow code-change.instructions.md exactly
   - NO EXCEPTIONS without explicit user approval
   - If you cannot meet a standard, STOP and ask for guidance

2. **Before Writing ANY Code:**
   - Read docs/test-coverage-expansion-plan.md
   - Read docs/agent-1-instructions.md (this file)
   - Read .github/unit-test-policy.md
   - Read docs/code-change.instructions.md
   - Review existing test patterns in tests/

3. **Your Assignment:**
   - Modules: analyzer/features.py, analyzer/slices.py, analyzer/model.py, analyzer/adjustments.py, calibration/train.py, calibration/featureset.py
   - Create: 6 new test files with comprehensive coverage
   - Target: 90%+ coverage for each assigned module

4. **Work Process (MANDATORY):**
   a. Read the source module thoroughly
   b. Identify all functions, classes, and code paths
   c. Design test cases covering:
      - Happy path (valid inputs)
      - Edge cases (boundaries, empty inputs)
      - Error cases (invalid inputs, exceptions)
      - State transitions (where applicable)
   d. Write tests following AAA pattern (Arrange-Act-Assert)
   e. Run quality checks IN ORDER:
      - black .
      - ruff check
      - pyright
      - pytest (all tests must pass)
      - pytest --cov=src/lexile_corpus_tuner (verify coverage increase)
   f. Fix any failures and repeat step e until ALL pass
   g. Document progress in test-coverage-expansion-plan.md

5. **Mocking Strategy:**
   - Mock external dependencies (filesystem, network, ML models)
   - Use pytest fixtures for reusable test data
   - Keep test data minimal but representative
   - Ensure tests are deterministic and fast (<1s per test)

6. **Quality Enforcement:**
   - If ANY check fails, you MUST fix it before proceeding
   - Do NOT skip checks or commit failing code
   - Do NOT create placeholder or incomplete tests
   - Every test must have a clear docstring explaining its purpose

7. **Coordination:**
   - You own your test files exclusively (no conflicts with other agents)
   - If you need shared fixtures, coordinate through conftest.py
   - Update test-coverage-expansion-plan.md with daily progress
   - Report blockers immediately

8. **Failure Modes to AVOID:**
   - Writing tests that don't actually test the code (false coverage)
   - Copying tests without understanding them
   - Skipping quality checks
   - Proceeding when coverage hasn't increased
   - Creating flaky or non-deterministic tests

**Start by confirming:**
1. You've read all required documents
2. You understand the standards are mandatory
3. You're ready to begin with analyzer/features.py

Do NOT write any code until you confirm understanding.
```

---

## Detailed Task List

### Phase 1: Foundation Tests

#### Task 1.1: `test_analyzer_features.py`
**Module:** `analyzer/features.py`  
**Current Coverage:** 40%  
**Missing Lines:** 33, 34, 36-57, 63, 73, 76, 78-79, 82, 88, 98-101

**Test Cases Required:**

1. **Test `compute_document_features` - Happy Path**
   - Input: Document with known text and simple frequency table
   - Verify: SliceFeatures and DocumentFeatures computed correctly
   - Check: All fields populated (word_count, unique_words, unseen_ratio, etc.)

2. **Test `compute_document_features` - Edge Cases**
   - Empty document → verify handling
   - Document with all unknown words → verify unseen_ratio = 1.0
   - Document with all known words → verify unseen_ratio = 0.0
   - Very long document → verify performance and correctness

3. **Test `_compute_unseen_floor` - Boundary Conditions**
   - Test with different slice_size values
   - Test with slices containing 0, 1, many unseen words
   - Verify floor calculation logic

4. **Test Feature Integration**
   - Verify SliceFeatures aggregation into DocumentFeatures
   - Check mathematical correctness (means, mins, maxes)

**Mocking Strategy:**
- Mock `frequency_loader.load_frequency_table()` to return controlled data
- Create fixture with small, predictable frequency table
- Use simple text with known word frequencies

**Success Criteria:**
- All lines in compute_document_features covered
- Coverage ≥90%
- Tests run in <2 seconds total
- No external file dependencies

---

#### Task 1.2: `test_analyzer_slices.py`
**Module:** `analyzer/slices.py`  
**Current Coverage:** 22%  
**Missing Lines:** 21-40, 45-100

**Test Cases Required:**

1. **Test `split_into_sentences` - Various Delimiters**
   - Simple sentences with periods
   - Questions and exclamations
   - Abbreviations (e.g., "Dr.", "Mr.")
   - Edge case: empty string, single word
   - Edge case: no punctuation

2. **Test `build_slices` - Windowing Logic**
   - Document with exact slice_size → verify 1 slice
   - Document with 2x slice_size → verify overlapping windows
   - Document shorter than slice_size → verify single slice
   - Document with 10 sentences, slice_size=5, stride=2 → verify overlap math

3. **Test `build_slices` - Token Aggregation**
   - Verify char_start and char_end boundaries
   - Verify text extraction matches original
   - Test with multi-word tokens

4. **Test Slice Data Integrity**
   - Verify all Slice fields populated correctly
   - Check that slices don't have gaps or overlaps (unless intended)

**Mocking Strategy:**
- Create fixture texts with known sentence boundaries
- Use simple, predictable input strings
- No external dependencies to mock

**Success Criteria:**
- Both functions fully covered
- Coverage ≥90%
- Edge cases handled gracefully
- No off-by-one errors in indexing

---

#### Task 1.3: `test_analyzer_model.py`
**Module:** `analyzer/model.py`  
**Current Coverage:** 40%  
**Missing Lines:** 18-19, 23, 28-42

**Test Cases Required:**

1. **Test `_load_model` - Success Path**
   - Mock file system to provide valid model file
   - Verify model loaded correctly
   - Check model type and properties

2. **Test `_load_model` - Error Handling**
   - File not found → verify exception raised
   - Invalid model file → verify exception raised
   - Corrupted data → verify exception raised

3. **Test `estimate_lexile_from_features` - Prediction**
   - Mock model to return predictable values
   - Provide DocumentFeatures → verify Lexile output
   - Test with edge case features (all zeros, very large values)

4. **Test Adjustment Logic**
   - Verify adjustments applied correctly
   - Test with and without adjustments
   - Check adjustment edge cases

**Mocking Strategy:**
- Mock `joblib.load()` to return a fake sklearn model
- Use a simple mock model that returns deterministic predictions
- Mock file existence checks

**Success Criteria:**
- All functions covered
- Coverage ≥90%
- Model loading and prediction logic verified
- Error cases handled

---

#### Task 1.4: `test_analyzer_adjustments.py`
**Module:** `analyzer/adjustments.py`  
**Current Coverage:** 25%  
**Missing Lines:** 11-16

**Test Cases Required:**

1. **Test `adjust_for_special_cases` - Adjustment Application**
   - Test with features that trigger adjustments
   - Test with features that don't trigger adjustments
   - Verify adjustment magnitude and direction

2. **Test Boundary Conditions**
   - Features at threshold values
   - Extreme feature values

**Mocking Strategy:**
- None required (pure logic)

**Success Criteria:**
- Full coverage of adjustment logic
- Coverage = 100%
- Logic correctness verified

---

#### Task 1.5: `test_calibration_train.py`
**Module:** `calibration/train.py`  
**Current Coverage:** 38%  
**Missing Lines:** 34-42, 47-85

**Test Cases Required:**

1. **Test `compute_metrics` - Regression Metrics**
   - Provide y_true, y_pred arrays
   - Verify MAE, RMSE, R² computed correctly
   - Test edge cases: perfect predictions, random predictions

2. **Test `train_regression_model` - Training Workflow**
   - Mock data loading
   - Verify model training completes
   - Check that trained model has expected properties
   - Test with various hyperparameters

3. **Test Model Selection Logic**
   - Test different model types (if supported)
   - Verify best model selected based on metrics

4. **Test Error Handling**
   - Insufficient training data → verify exception
   - Invalid features → verify exception

**Mocking Strategy:**
- Mock sklearn models for speed
- Use small, synthetic training data
- Mock joblib.dump for model saving

**Success Criteria:**
- Training workflow covered
- Metrics calculation verified
- Coverage ≥90%
- Tests run fast (mocked ML)

---

#### Task 1.6: `test_calibration_featureset.py`
**Module:** `calibration/featureset.py`  
**Current Coverage:** 33%  
**Missing Lines:** 13, 20, 22-24, 27, 31-35

**Test Cases Required:**

1. **Test `make_regression_features` - Feature Engineering**
   - Input: List of DocumentFeatures
   - Verify: Correct numpy array shape
   - Check: Feature ordering and scaling

2. **Test Edge Cases**
   - Empty input → verify handling
   - Single document → verify shape
   - Many documents → verify stacking

**Mocking Strategy:**
- Use simple DocumentFeatures fixtures
- No external dependencies

**Success Criteria:**
- Full function coverage
- Coverage ≥90%
- Correct array shapes verified

---

## Phase 2: Integration Tests

### Task 2.1: End-to-End Analyzer Workflow
- Test: Load doc → compute features → estimate Lexile
- Verify: Entire pipeline works together
- Check: Intermediate outputs are correct

### Task 2.2: Edge Case Deep Dive
- Malformed input handling
- Boundary conditions for all functions
- Performance with large inputs

---

## Phase 3: Polish

### Task 3.1: Documentation Review
- Ensure every test has clear docstring
- Add comments for complex assertions
- Update any outdated test documentation

### Task 3.2: Coverage Gap Analysis
- Run coverage report
- Identify any remaining gaps
- Add targeted tests to fill gaps

### Task 3.3: Final Verification
- Run full test suite
- Verify ≥90% coverage on all assigned modules
- Check all quality gates pass

---

## Fixtures to Create (in `tests/conftest.py` or local file)

```python
@pytest.fixture
def sample_frequency_table() -> dict[str, int]:
    """Small frequency table for testing."""
    return {
        "the": 100000,
        "a": 50000,
        "test": 1000,
        "word": 500,
    }

@pytest.fixture
def simple_document_text() -> str:
    """Simple text for testing feature computation."""
    return "The test word is a simple test. This is another test."

@pytest.fixture
def sample_document_features() -> DocumentFeatures:
    """Sample DocumentFeatures for training tests."""
    # ... create realistic DocumentFeatures instance
```

---

## Success Checklist

Before marking any task complete:
- [ ] Read source module completely
- [ ] Identified all code paths
- [ ] Wrote tests for happy path
- [ ] Wrote tests for edge cases
- [ ] Wrote tests for error cases
- [ ] All tests have docstrings
- [ ] Ran `black .` (passes)
- [ ] Ran `ruff check` (passes)
- [ ] Ran `pyright` (passes)
- [ ] Ran `pytest` (all pass)
- [ ] Ran `pytest --cov` (coverage increased)
- [ ] Verified coverage ≥90% for module
- [ ] Updated test-coverage-expansion-plan.md

---

## Reporting Template

Copy and paste into test-coverage-expansion-plan.md:

```markdown
### Agent 1 Update - YYYY-MM-DD

**Phase:** [1/2/3]  
**Tasks Completed:**
- test_analyzer_features.py ✅
- test_analyzer_slices.py 🔄
- (etc.)

**Coverage Changes:**
- analyzer/features.py: 40% → 92%
- analyzer/slices.py: 22% → 85% (in progress)

**Blockers:**
- None / [describe blocker]

**Next Steps:**
- Complete test_analyzer_slices.py
- Begin test_analyzer_model.py
```

---

## Common Pitfalls to Avoid

1. **Writing tests that don't actually test anything**
   - ❌ `assert result` (always True if no exception)
   - ✅ `assert result == expected_value`

2. **Not mocking external dependencies**
   - ❌ Tests that read real files or call real APIs
   - ✅ All external calls mocked

3. **Tests that depend on execution order**
   - ❌ Test B assumes Test A ran first
   - ✅ Every test independent and isolated

4. **Vague test names and missing docstrings**
   - ❌ `def test_function():`
   - ✅ `def test_compute_features_with_empty_document_returns_zero_values():`

5. **Not running quality checks**
   - ❌ "I'll check later"
   - ✅ Run after EVERY test file

---

## Emergency Contacts

If blocked or uncertain:
1. Review existing tests in `tests/` for patterns
2. Consult `unit-test-policy.md` for specific guidance
3. Check `code-change.instructions.md` for code standards
4. Report blocker in test-coverage-expansion-plan.md
5. Request guidance from coordinator

**Remember: Quality over speed. No compromises on standards.**
