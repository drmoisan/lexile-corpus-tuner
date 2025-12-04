# Agent 2: Corpus Pipeline Testing

**Mission:** Expand test coverage for corpus processing modules from ~18% to 90%+

**Assigned Modules:**
- `src/lexile_corpus_tuner/corpus/download.py` (18% → 90%+)
- `src/lexile_corpus_tuner/corpus/normalize.py` (16% → 90%+)
- `src/lexile_corpus_tuner/corpus/frequencies.py` (21% → 90%+)

**Test Files to Create/Extend:**
- `tests/test_corpus_download.py` (NEW)
- `tests/test_corpus_normalize.py` (NEW)
- `tests/test_corpus_frequencies.py` (NEW)

---

## Loading Prompt

```
You are Agent 2 in a coordinated test coverage expansion effort for the lexile-corpus-tuner project. Your mission is to create comprehensive unit tests for the corpus processing pipeline, bringing coverage from ~18% to 90%+.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1. **Standards Compliance (NON-NEGOTIABLE):**
   - Every test MUST follow the unit-test-policy.md exactly
   - Every test MUST follow code-change.instructions.md exactly
   - NO EXCEPTIONS without explicit user approval
   - If you cannot meet a standard, STOP and ask for guidance

2. **Before Writing ANY Code:**
   - Read docs/test-coverage-expansion-plan.md
   - Read docs/agent-2-instructions.md (this file)
   - Read .github/unit-test-policy.md
   - Read docs/code-change.instructions.md
   - Review existing test patterns in tests/

3. **Your Assignment:**
   - Modules: corpus/download.py, corpus/normalize.py, corpus/frequencies.py
   - Create: 3 new test files with comprehensive coverage
   - Target: 90%+ coverage for each assigned module

4. **Work Process (MANDATORY):**
   a. Read the source module thoroughly
   b. Identify all functions, classes, and code paths
   c. Design test cases covering:
      - Happy path (valid inputs)
      - Edge cases (boundaries, empty inputs)
      - Error cases (network failures, file I/O errors)
      - Multi-source integration
   d. Write tests following AAA pattern (Arrange-Act-Assert)
   e. Run quality checks IN ORDER:
      - black .
      - ruff check
      - pyright
      - pytest (all tests must pass)
      - pytest --cov=src/lexile_corpus_tuner (verify coverage increase)
   f. Fix any failures and repeat step e until ALL pass
   g. Document progress in test-coverage-expansion-plan.md

5. **Mocking Strategy (CRITICAL for this agent):**
   - Mock ALL network calls (requests library)
   - Mock ALL file system operations where possible (use tmp_path fixture)
   - Mock external data sources (Gutenberg, Wikipedia dumps, etc.)
   - Ensure tests run without internet access
   - Keep test data small (use representative samples)

6. **Quality Enforcement:**
   - If ANY check fails, you MUST fix it before proceeding
   - Do NOT skip checks or commit failing code
   - Do NOT create tests that require network access
   - Every test must be deterministic and fast (<5s total per file)

7. **Coordination:**
   - You own your test files exclusively (no conflicts with other agents)
   - Wait for Agent 4 to complete textutils.py tests if needed
   - Update test-coverage-expansion-plan.md with daily progress
   - Report blockers immediately

8. **Failure Modes to AVOID:**
   - Tests that download real files from the internet
   - Tests that depend on specific file system state
   - Tests that are slow (>1s per test)
   - Incomplete mocking leading to flaky tests

**Start by confirming:**
1. You've read all required documents
2. You understand the standards are mandatory
3. You're ready to begin with corpus/download.py

Do NOT write any code until you confirm understanding.
```

---

## Detailed Task List

### Phase 1: Foundation Tests

#### Task 2.1: `test_corpus_download.py`
**Module:** `corpus/download.py`  
**Current Coverage:** 18%  
**Missing Lines:** 31-34, 41-63, 70-82, 87-128, 132-147, 152-165, 169-177, 181-184

**Test Cases Required:**

1. **Test `ensure_dirs` - Directory Creation**
   - Mock pathlib.Path.mkdir
   - Verify directories created with correct structure
   - Test when dirs already exist → no error
   - Test permission errors → exception propagated

2. **Test `download_gutenberg_subset` - Main Download Flow**
   - Mock `_iter_gutenberg_ids` to return small list
   - Mock `_resolve_gutenberg_url` to return test URLs
   - Mock `_download_file` to simulate successful downloads
   - Verify all IDs processed
   - Test progress tracking (if applicable)

3. **Test `download_gutenberg_subset` - Error Recovery**
   - Mock download failure for some IDs
   - Verify partial success (other downloads continue)
   - Test network timeout scenarios
   - Verify logging of failures

4. **Test `download_simple_wiki_dump` - Download Flow**
   - Mock HTTP download of dump file
   - Verify file saved to correct location
   - Test download progress/chunking
   - Test HTTP error responses (404, 500)

5. **Test `download_oer_sources` - Multi-source Download**
   - Mock JSON source file loading
   - Mock download for each source
   - Verify all sources processed
   - Test invalid JSON → exception

6. **Test `_iter_gutenberg_ids` - ID Parsing**
   - Provide text file with IDs (use tmp_path)
   - Verify IDs yielded correctly
   - Test empty file → no IDs
   - Test file with comments/whitespace → IDs parsed

7. **Test `_resolve_gutenberg_url` - URL Construction**
   - Input: Gutenberg ID
   - Verify: Correct URL format
   - Test various ID formats (1, 100, 10000)

8. **Test `_download_file` - HTTP Download**
   - Mock requests.get with successful response
   - Verify file written to correct path
   - Test chunked download (large files)
   - Test HTTP errors (404, 403, 500)
   - Test network timeouts

9. **Test `_copy_local_file` - Local File Copy**
   - Use tmp_path to create source file
   - Verify copied to destination
   - Test missing source → exception

**Mocking Strategy:**
```python
@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mock requests.get for testing downloads."""
    def fake_get(url, *args, **kwargs):
        response = Mock()
        response.status_code = 200
        response.content = b"test content"
        response.iter_content = lambda chunk_size: [b"test content"]
        return response
    
    monkeypatch.setattr("requests.get", fake_get)
    return fake_get
```

**Success Criteria:**
- All download functions covered
- Coverage ≥90%
- NO real network calls
- Tests run fast (<5s total)
- Error handling verified

---

#### Task 2.2: `test_corpus_normalize.py`
**Module:** `corpus/normalize.py`  
**Current Coverage:** 16%  
**Missing Lines:** 36-83, 88-90, 94-156, 160-167, 173-188, 192-198, 202-230

**Test Cases Required:**

1. **Test `normalize_all_sources` - Main Orchestration**
   - Mock `iter_raw_texts` to yield test documents
   - Mock file writing (`_write_shard`)
   - Verify shards created correctly
   - Test shard size logic (when to create new shard)

2. **Test `normalize_all_sources` - Multi-source Integration**
   - Mock multiple sources (gutenberg, simple_wiki, oer)
   - Verify all sources processed
   - Test source-specific normalization rules

3. **Test `iter_raw_texts` - Text Iteration**
   - Mock `_iter_gutenberg_texts`
   - Mock `_iter_simple_wiki_texts`
   - Verify texts yielded in order
   - Test empty sources → no texts

4. **Test `_iter_gutenberg_texts` - Gutenberg Processing**
   - Use tmp_path to create test .txt and .epub files
   - Verify both formats processed
   - Mock epub parsing (use existing epub module tests as guide)
   - Test file read errors → skip and log

5. **Test `_iter_simple_wiki_texts` - Wikipedia Dump Processing**
   - Mock XML dump file with sample articles
   - Verify articles extracted correctly
   - Test malformed XML → skip and continue
   - Test filtering logic (min length, etc.)

6. **Test `_iter_oer_texts` - OER Source Processing**
   - Mock OER source files
   - Verify text extraction
   - Test various OER formats

7. **Test `_classify_gutenberg_path` - Format Detection**
   - Input: Path to .txt file → returns "txt"
   - Input: Path to .epub file → returns "epub"
   - Input: Other format → returns appropriate label

8. **Test `_write_shard` - Shard Writing**
   - Use tmp_path for output
   - Verify parquet file created
   - Verify data integrity (can read back)
   - Test write permissions error

9. **Test `_write_summary` - Summary Generation**
   - Mock shard metadata
   - Verify summary JSON created
   - Check summary contains expected fields

**Mocking Strategy:**
```python
@pytest.fixture
def sample_gutenberg_files(tmp_path):
    """Create sample Gutenberg files for testing."""
    txt_file = tmp_path / "raw" / "gutenberg" / "123.txt"
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    txt_file.write_text("Sample Gutenberg text content.")
    
    # Create more files as needed
    return tmp_path

@pytest.fixture
def mock_xml_dump(tmp_path):
    """Create mock Wikipedia XML dump."""
    xml_content = """
    <mediawiki>
      <page><title>Test</title><text>Article content</text></page>
    </mediawiki>
    """
    dump_file = tmp_path / "wiki_dump.xml"
    dump_file.write_text(xml_content)
    return dump_file
```

**Success Criteria:**
- Normalization pipeline covered
- Coverage ≥90%
- Tests use tmp_path (no global file system pollution)
- Large file processing mocked
- Tests run fast (<5s total)

---

#### Task 2.3: `test_corpus_frequencies.py`
**Module:** `corpus/frequencies.py`  
**Current Coverage:** 21%  
**Missing Lines:** 21-85, 93-95, 99-114

**Test Cases Required:**

1. **Test `compute_global_frequencies` - Frequency Computation**
   - Mock `iter_raw_texts` to provide test documents
   - Verify frequency dict computed correctly
   - Test word counting logic
   - Test case normalization (if applicable)

2. **Test `compute_global_frequencies` - Multi-source Weighting**
   - Mock `_load_source_weights` to provide test weights
   - Verify weighted frequencies computed
   - Test equal weights → simple count
   - Test unequal weights → correct weighting

3. **Test `compute_global_frequencies` - Output Writing**
   - Use tmp_path for output
   - Verify frequency file written
   - Test file format (Parquet/JSON/etc.)
   - Verify can read back and reconstruct

4. **Test `_current_version` - Versioning**
   - Verify version string returned
   - Test version format

5. **Test `_load_source_weights` - Weight Loading**
   - Mock source weight file (use tmp_path)
   - Verify weights loaded correctly
   - Test missing file → default weights
   - Test malformed JSON → exception

**Mocking Strategy:**
```python
@pytest.fixture
def mock_normalized_corpus(tmp_path):
    """Create mock normalized corpus shards."""
    shard_dir = tmp_path / "normalized"
    shard_dir.mkdir()
    
    # Create mock parquet files with text data
    # Use pandas or pyarrow to write
    df = pd.DataFrame({"text": ["sample text", "more text"]})
    df.to_parquet(shard_dir / "shard_0.parquet")
    
    return shard_dir

@pytest.fixture
def sample_source_weights(tmp_path):
    """Create sample source weights file."""
    weights = {"gutenberg": 0.5, "simple_wiki": 0.3, "oer": 0.2}
    weights_file = tmp_path / "source_weights.json"
    weights_file.write_text(json.dumps(weights))
    return weights_file
```

**Success Criteria:**
- Frequency computation covered
- Coverage ≥90%
- Tests use fixtures for data
- Output verification included
- Tests run fast (<5s total)

---

## Phase 2: Integration Tests

### Task 2.4: End-to-End Corpus Pipeline
- Test: Download → Normalize → Compute Frequencies
- Use small, mocked data throughout
- Verify pipeline produces expected output

### Task 2.5: Error Recovery and Edge Cases
- Network failures during download
- Corrupted files during normalization
- Empty corpus → handle gracefully

### Task 2.6: Performance and Scalability
- Test with "large" mocked datasets (100s of documents)
- Verify memory usage stays reasonable
- Check that progress reporting works

---

## Phase 3: Polish

### Task 3.1: Documentation and Cleanup
- Add comprehensive docstrings
- Document any mocking assumptions
- Clean up test fixtures

### Task 3.2: Coverage Gap Analysis
- Run coverage report
- Add tests for any remaining gaps

### Task 3.3: Final Verification
- Run full test suite
- Verify ≥90% coverage on all assigned modules
- Check all quality gates pass

---

## Fixtures to Create

```python
# In tests/test_corpus_download.py
@pytest.fixture
def mock_gutenberg_ids_file(tmp_path):
    """Create a file with Gutenberg IDs."""
    ids_file = tmp_path / "gutenberg_ids.txt"
    ids_file.write_text("1\n100\n1000\n")
    return ids_file

@pytest.fixture
def mock_http_response():
    """Create mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.content = b"mock file content"
    response.iter_content = lambda chunk_size: [b"chunk1", b"chunk2"]
    return response

# In tests/test_corpus_normalize.py
@pytest.fixture
def sample_raw_corpus(tmp_path):
    """Create sample raw corpus files."""
    raw_dir = tmp_path / "raw"
    
    # Gutenberg
    gut_dir = raw_dir / "gutenberg"
    gut_dir.mkdir(parents=True)
    (gut_dir / "123.txt").write_text("Sample Gutenberg text.")
    
    # Simple Wiki
    wiki_dir = raw_dir / "simple_wiki"
    wiki_dir.mkdir(parents=True)
    # Add mock wiki dump
    
    return raw_dir

# In tests/test_corpus_frequencies.py
@pytest.fixture
def mock_normalized_shards(tmp_path):
    """Create mock normalized corpus shards."""
    import pandas as pd
    
    shard_dir = tmp_path / "normalized"
    shard_dir.mkdir()
    
    df = pd.DataFrame({
        "text": ["the quick brown fox", "the lazy dog jumps"],
        "source": ["gutenberg", "simple_wiki"]
    })
    df.to_parquet(shard_dir / "shard_0.parquet")
    
    return shard_dir
```

---

## Success Checklist

Before marking any task complete:
- [ ] Read source module completely
- [ ] Identified all code paths
- [ ] Wrote tests for happy path
- [ ] Wrote tests for edge cases (file errors, network errors)
- [ ] Wrote tests for error cases
- [ ] All external dependencies mocked
- [ ] All tests have docstrings
- [ ] Tests use tmp_path for file operations
- [ ] Ran `black .` (passes)
- [ ] Ran `ruff check` (passes)
- [ ] Ran `pyright` (passes)
- [ ] Ran `pytest` (all pass)
- [ ] Ran `pytest --cov` (coverage increased)
- [ ] Verified coverage ≥90% for module
- [ ] Verified tests run fast (<5s per file)
- [ ] Updated test-coverage-expansion-plan.md

---

## Reporting Template

```markdown
### Agent 2 Update - YYYY-MM-DD

**Phase:** [1/2/3]  
**Tasks Completed:**
- test_corpus_download.py ✅
- test_corpus_normalize.py 🔄

**Coverage Changes:**
- corpus/download.py: 18% → 91%
- corpus/normalize.py: 16% → 78% (in progress)

**Blockers:**
- None / [describe blocker]

**Next Steps:**
- Complete test_corpus_normalize.py
- Begin test_corpus_frequencies.py
```

---

## Common Pitfalls to Avoid

1. **Not mocking network calls**
   - ❌ Tests that call real URLs (even in try/except)
   - ✅ All requests.get calls mocked

2. **Not using tmp_path**
   - ❌ Writing to fixed paths in /tmp or project dir
   - ✅ Using pytest's tmp_path fixture for ALL file I/O

3. **Slow tests**
   - ❌ Downloading/processing large files
   - ✅ Small, representative test data only

4. **Incomplete mocking**
   - ❌ Mocking requests but not file I/O
   - ✅ Mock ALL external dependencies

5. **Tests that depend on external state**
   - ❌ Tests that fail if internet is down
   - ✅ Tests run completely offline

---

## Emergency Contacts

If blocked or uncertain:
1. Review existing corpus tests (test_extract_simple_wiki_dump.py, etc.)
2. Consult `unit-test-policy.md` for specific guidance
3. Check `code-change.instructions.md` for code standards
4. Report blocker in test-coverage-expansion-plan.md
5. Request guidance from coordinator

**Remember: Quality over speed. No compromises on standards. All tests must be fast and deterministic.**
