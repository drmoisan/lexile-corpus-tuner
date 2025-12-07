# Agent 3: Calibration CLI & Preprocessing Testing

**Mission:** Expand test coverage for calibration CLI and text preprocessing modules from ~25% to 90%+

**Assigned Modules:**
- `src/lexile_corpus_tuner/calibration/cli.py` (22% → 90%+)
- `src/lexile_corpus_tuner/calibration/model_store.py` (55% → 90%+)
- `src/lexile_corpus_tuner/estimators/lexile_v2_preprocessing.py` (23% → 90%+)

**Test Files to Create/Extend:**
- `tests/test_calibration_cli.py` (NEW)
- `tests/test_calibration_model_store.py` (NEW)
- `tests/test_lexile_v2_preprocessing.py` (NEW - or extend existing if present)

---

## Loading Prompt

```
You are Agent 3 in a coordinated test coverage expansion effort for the lexile-corpus-tuner project. Your mission is to create comprehensive unit tests for calibration CLI and preprocessing modules, bringing coverage from ~25% to 90%+.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1. **Standards Compliance (NON-NEGOTIABLE):**
   - Every test MUST follow the unit-test-policy.md exactly
   - Every test MUST follow code-change.instructions.md exactly
   - NO EXCEPTIONS without explicit user approval
   - If you cannot meet a standard, STOP and ask for guidance

2. **Before Writing ANY Code:**
   - Read docs/test-coverage-expansion-plan.md
   - Read docs/agent-3-instructions.md (this file)
   - Read .github/unit-test-policy.md
   - Read docs/code-change.instructions.md
   - Review existing test patterns in tests/

3. **Your Assignment:**
   - Modules: calibration/cli.py, calibration/model_store.py, estimators/lexile_v2_preprocessing.py
   - Create: 3 new test files with comprehensive coverage
   - Target: 90%+ coverage for each assigned module

4. **Work Process (MANDATORY):**
   a. Read the source module thoroughly
   b. Identify all functions, classes, and code paths
   c. Design test cases covering:
      - Happy path (valid inputs)
      - Edge cases (boundaries, empty inputs)
      - Error cases (missing files, invalid formats)
      - CLI argument validation
   d. Write tests following AAA pattern (Arrange-Act-Assert)
   e. Run quality checks IN ORDER:
      - black .
      - ruff check
      - pyright
      - pytest (all tests must pass)
      - pytest --cov=src/lexile_corpus_tuner (verify coverage increase)
   f. Fix any failures and repeat step e until ALL pass
   g. Document progress in test-coverage-expansion-plan.md

5. **Mocking Strategy (CRITICAL):**
   - Mock ALL CLI invocations (use CliRunner or invoke directly)
   - Mock ALL file system operations (use tmp_path)
   - Mock NLTK downloads and data
   - Mock sklearn model training (use fake models)
   - Ensure tests run without internet or large downloads

6. **Dependencies:**
   - Wait for Agent 4 to complete config.py tests before starting
   - Coordinate if config fixtures are needed

7. **Quality Enforcement:**
   - If ANY check fails, you MUST fix it before proceeding
   - Do NOT skip checks or commit failing code
   - Do NOT create tests that require NLTK data downloads
   - Every test must be deterministic and fast (<2s per test)

8. **Failure Modes to AVOID:**
   - Tests that call real CLI commands that modify global state
   - Tests that download NLTK data
   - Tests that train real ML models
   - Incomplete mocking of external dependencies

**Start by confirming:**
1. You've read all required documents
2. You understand the standards are mandatory
3. You're ready to begin with calibration/model_store.py (easiest first)

Do NOT write any code until you confirm understanding.
```

---

## Detailed Task List

### Phase 1: Foundation Tests

#### Task 3.1: `test_calibration_model_store.py`
**Module:** `calibration/model_store.py`  
**Current Coverage:** 55%  
**Missing Lines:** 17-18, 25-26, 31

**Test Cases Required:**

1. **Test `save_model` - Model Serialization**
   - Create mock sklearn model
   - Use tmp_path for output directory
   - Verify model file created (.pkl or similar)
   - Verify metadata file created
   - Test model and metadata can be loaded back

2. **Test `save_model` - Directory Creation**
   - Use non-existent directory path
   - Verify directories created automatically
   - Test permission errors → exception

3. **Test `load_model_spec` - Model Loading**
   - Create mock model file (use tmp_path)
   - Verify model loaded correctly
   - Test missing file → exception
   - Test corrupted file → exception

4. **Test Model Spec Format**
   - Verify spec contains expected fields (version, features, etc.)
   - Test spec JSON is valid and can be parsed

**Mocking Strategy:**
```python
@pytest.fixture
def mock_trained_model():
    """Create a mock sklearn model."""
    from sklearn.dummy import DummyRegressor
    model = DummyRegressor()
    model.fit([[1], [2]], [100, 200])  # Trivial training
    return model

@pytest.fixture
def model_output_dir(tmp_path):
    """Create a temporary model output directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
```

**Success Criteria:**
- All save/load functions covered
- Coverage = 100% (module is small)
- File I/O uses tmp_path
- Tests run fast (<1s total)

---

#### Task 3.2: `test_lexile_v2_preprocessing.py`
**Module:** `estimators/lexile_v2_preprocessing.py`  
**Current Coverage:** 23%  
**Missing Lines:** 26-35, 48-49, 53-54, 58-72, 76-88, 98-117

**Test Cases Required:**

1. **Test `load_stopwords` - Stopword Loading**
   - Create mock stopwords file (use tmp_path)
   - Verify stopwords loaded as list
   - Test missing file → exception
   - Test empty file → empty list
   - Test file with whitespace → stripped correctly

2. **Test `vectorize_with_lexile_pipeline` - Vectorization**
   - Mock `_load_pickle` to return fake vectorizer
   - Provide sample text
   - Verify vectorized output shape and type
   - Mock NLTK dependencies

3. **Test `_load_pickle` - Pickle Loading**
   - Create mock pickle file (use tmp_path)
   - Verify object loaded correctly
   - Test missing file → exception
   - Test corrupted pickle → exception

4. **Test `_segment_text` - Text Segmentation**
   - Mock stopwords list
   - Provide text with known words
   - Verify segments created correctly
   - Test edge cases: empty text, all stopwords, no stopwords

5. **Test `_lemmatize_segments` - Lemmatization**
   - Mock NLTK WordNetLemmatizer
   - Provide segments
   - Verify lemmatized output
   - Test with various POS tags

6. **Test `_ensure_nltk_dependencies` - Dependency Check**
   - Mock NLTK data path checking
   - Verify downloads triggered (mocked)
   - Test when data already exists → no download
   - Test download failure → exception

**Mocking Strategy:**
```python
@pytest.fixture
def mock_nltk_data(monkeypatch):
    """Mock NLTK data and downloads."""
    # Mock nltk.data.find to prevent actual downloads
    def mock_find(resource_path):
        return "/fake/path/to/nltk/data"
    
    monkeypatch.setattr("nltk.data.find", mock_find)
    
    # Mock lemmatizer
    class FakeLemmatizer:
        def lemmatize(self, word, pos='n'):
            return word.lower()  # Simple mock
    
    monkeypatch.setattr("nltk.stem.WordNetLemmatizer", lambda: FakeLemmatizer())

@pytest.fixture
def sample_stopwords_file(tmp_path):
    """Create sample stopwords file."""
    stopwords = tmp_path / "stopwords.txt"
    stopwords.write_text("the\na\nan\nand\n")
    return stopwords

@pytest.fixture
def mock_vectorizer():
    """Create mock scikit-learn vectorizer."""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(["sample text"])
    return vectorizer
```

**Success Criteria:**
- All preprocessing functions covered
- Coverage ≥90%
- NO real NLTK downloads during tests
- Tests run fast (<2s total)
- Lemmatization logic verified

---

#### Task 3.3: `test_calibration_cli.py`
**Module:** `calibration/cli.py`  
**Current Coverage:** 22%  
**Missing Lines:** 34-79, 97-154, 170-184, 192-194, 198-216, 220-228, 232-234, 238-253

**Test Cases Required:**

1. **Test `calibration_group` Command Group**
   - Verify subcommands registered
   - Test `--help` output

2. **Test `fetch_texts` Command - Success Path**
   - Use CliRunner to invoke command
   - Mock `_read_catalog` to return test data
   - Mock `_fetch_gutenberg_text` or `_fetch_http_text`
   - Use tmp_path for output directory
   - Verify files written to output
   - Verify metadata recorded

3. **Test `fetch_texts` Command - Error Handling**
   - Mock fetch failure for some texts
   - Verify partial success (continues processing)
   - Test invalid catalog file → exception
   - Test missing output directory → created automatically

4. **Test `build_dataset` Command - Success Path**
   - Mock input files (use tmp_path)
   - Mock feature computation
   - Use tmp_path for output dataset
   - Verify dataset file created (Parquet/CSV)
   - Verify dataset format correct

5. **Test `build_dataset` Command - Error Handling**
   - Test missing input directory → exception
   - Test malformed input files → skip with warning
   - Test invalid Lexile values → filtered out

6. **Test `fit` Command - Training Flow**
   - Mock dataset loading
   - Mock model training (use DummyRegressor)
   - Use tmp_path for model output
   - Verify model saved
   - Verify metrics reported

7. **Test `fit` Command - Validation**
   - Test with validation split
   - Verify metrics computed on validation set
   - Test insufficient data → exception

8. **Test Helper Functions**
   - `_read_catalog` with mock CSV/JSON
   - `_fetch_gutenberg_text` with mocked requests
   - `_fetch_http_text` with mocked requests
   - `_strip_html` with sample HTML
   - `_parse_lexile_value` with various formats

**Mocking Strategy:**
```python
from typer.testing import CliRunner

@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()

@pytest.fixture
def mock_catalog_file(tmp_path):
    """Create mock catalog CSV."""
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("id,lexile,url\n1,350L,http://example.com/1.txt\n")
    return catalog

@pytest.fixture
def mock_http_fetch(monkeypatch):
    """Mock HTTP fetching."""
    def fake_fetch(url):
        return "Sample fetched text content."
    
    monkeypatch.setattr(
        "lexile_corpus_tuner.lexile_scoring_model.calibration.cli._fetch_http_text",
        fake_fetch
    )

@pytest.fixture
def sample_dataset_file(tmp_path):
    """Create mock dataset for training."""
    import pandas as pd
    
    df = pd.DataFrame({
        "text": ["sample text 1", "sample text 2"],
        "lexile": [300, 400]
    })
    dataset_file = tmp_path / "dataset.parquet"
    df.to_parquet(dataset_file)
    return dataset_file
```

**Success Criteria:**
- All CLI commands covered
- Coverage ≥90%
- Tests use CliRunner or direct function calls
- NO real HTTP requests
- Tests use tmp_path for all I/O
- Tests run fast (<5s total)

---

## Phase 2: Integration Tests

### Task 3.4: End-to-End Calibration Workflow
- Test: Fetch texts → Build dataset → Fit model
- Use small, mocked data throughout
- Verify trained model can make predictions

### Task 3.5: Error Recovery
- Network failures during fetch
- Corrupted files during dataset building
- Training failures → proper error messages

### Task 3.6: CLI Argument Validation
- Test missing required arguments
- Test invalid argument combinations
- Test help messages

---

## Phase 3: Polish

### Task 3.7: Documentation and Cleanup
- Add comprehensive docstrings
- Document mocking strategies
- Clean up test fixtures

### Task 3.8: Coverage Gap Analysis
- Run coverage report
- Add tests for remaining gaps

### Task 3.9: Final Verification
- Run full test suite
- Verify ≥90% coverage on all assigned modules
- Check all quality gates pass

---

## Fixtures to Create

```python
# In tests/conftest.py or test_calibration_cli.py
@pytest.fixture
def mock_nltk_setup(monkeypatch):
    """Prevent NLTK downloads during tests."""
    def mock_download(resource_name):
        return True  # Pretend success
    
    monkeypatch.setattr("nltk.download", mock_download)
    
    # Mock data.find to always return a path
    monkeypatch.setattr(
        "nltk.data.find",
        lambda path: "/mock/nltk/data"
    )

@pytest.fixture
def mock_sklearn_training(monkeypatch):
    """Speed up sklearn model training in tests."""
    from sklearn.dummy import DummyRegressor
    
    class FastModel:
        def fit(self, X, y):
            self.model_ = DummyRegressor()
            self.model_.fit(X, y)
            return self
        
        def predict(self, X):
            return self.model_.predict(X)
    
    # Patch relevant model classes if needed
```

---

## Success Checklist

Before marking any task complete:
- [ ] Read source module completely
- [ ] Identified all code paths
- [ ] Wrote tests for happy path
- [ ] Wrote tests for edge cases
- [ ] Wrote tests for error cases
- [ ] Mocked all external dependencies (NLTK, HTTP, file I/O)
- [ ] All tests have docstrings
- [ ] Tests use tmp_path for file operations
- [ ] Tests use CliRunner for CLI testing
- [ ] Ran `black .` (passes)
- [ ] Ran `ruff check` (passes)
- [ ] Ran `pyright` (passes)
- [ ] Ran `pytest` (all pass)
- [ ] Ran `pytest --cov` (coverage increased)
- [ ] Verified coverage ≥90% for module
- [ ] Verified tests run fast (<2s per test)
- [ ] Updated test-coverage-expansion-plan.md

---

## Reporting Template

```markdown
### Agent 3 Update - YYYY-MM-DD

**Phase:** [1/2/3]  
**Tasks Completed:**
- test_calibration_model_store.py ✅
- test_lexile_v2_preprocessing.py 🔄

**Coverage Changes:**
- calibration/model_store.py: 55% → 100%
- estimators/lexile_v2_preprocessing.py: 23% → 82% (in progress)

**Blockers:**
- Waiting for Agent 4 to complete config.py tests [if applicable]

**Next Steps:**
- Complete test_lexile_v2_preprocessing.py
- Begin test_calibration_cli.py
```

---

## Common Pitfalls to Avoid

1. **Not mocking NLTK**
   - ❌ Tests that download NLTK data
   - ✅ All NLTK operations mocked

2. **Not using CliRunner**
   - ❌ Calling CLI functions directly without proper setup
   - ✅ Using CliRunner for CLI command testing

3. **Slow model training**
   - ❌ Training real sklearn models with large datasets
   - ✅ Using DummyRegressor or mocked models

4. **Not using tmp_path**
   - ❌ Writing to fixed paths
   - ✅ All file I/O through tmp_path

5. **Tests with side effects**
   - ❌ Tests that modify global NLTK state
   - ✅ Tests that clean up after themselves

---

## Emergency Contacts

If blocked or uncertain:
1. Review existing CLI tests in tests/test_cli.py
2. Consult `unit-test-policy.md` for specific guidance
3. Check `code-change.instructions.md` for code standards
4. Report blocker in test-coverage-expansion-plan.md
5. Request guidance from coordinator

**Remember: Quality over speed. No compromises on standards. Mock everything external.**

