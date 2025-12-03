# Test Coverage Expansion Plan: 55% → 90%+

**Project:** lexile-corpus-tuner  
**Current Coverage:** 55%  
**Target Coverage:** 90%+  
**Strategy:** Phased, parallel agent development  
**Estimated Agents:** 4 working in parallel

---

## Executive Summary

This document outlines a systematic, phased approach to expanding test coverage for the lexile-corpus-tuner project from 55% to above 90%. The plan divides the work into **4 parallel tracks** handled by specialized agents, each focusing on distinct module groups to minimize conflicts and maximize efficiency.

### Critical Coverage Gaps (by module)

| Module | Current | Missing Lines | Priority |
|--------|---------|---------------|----------|
| `corpus/normalize.py` | 16% | 143 | HIGH |
| `calibration/cli.py` | 22% | 128 | HIGH |
| `corpus/download.py` | 18% | 107 | HIGH |
| `corpus/frequencies.py` | 21% | 59 | MEDIUM |
| `estimators/lexile_v2_preprocessing.py` | 23% | 48 | MEDIUM |
| `analyzer/slices.py` | 22% | 46 | MEDIUM |
| `analyzer/features.py` | 40% | 31 | MEDIUM |
| `analyzer/cli.py` | 46% | 19 | LOW |
| `calibration/train.py` | 38% | 26 | MEDIUM |

---

## Phased Approach

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Cover high-value, independent utility modules  
**Target:** Bring coverage to 70%

- Focus on modules with clear, isolated functionality
- Establish testing patterns and standards
- Build reusable test fixtures and helpers

### Phase 2: Integration (Weeks 3-4)
**Goal:** Cover modules with external dependencies  
**Target:** Bring coverage to 85%

- Focus on corpus processing pipeline
- Cover calibration and analysis workflows
- Mock external dependencies (APIs, file systems)

### Phase 3: Polish (Week 5)
**Goal:** Cover edge cases and CLI interactions  
**Target:** Reach 90%+

- CLI command testing
- Error handling paths
- Boundary conditions
- Edge case scenarios

---

## Parallel Agent Assignment

### Agent 1: Core Lexile Pipeline Testing
**Modules:** `analyzer/*`, `calibration/train.py`, `calibration/featureset.py`  
**Current avg:** ~35%  
**Target:** 90%+  
**Estimated LOC to test:** ~300 lines

**Focus areas:**
- Feature computation (analyzer/features.py)
- Slice generation (analyzer/slices.py)
- Model estimation (analyzer/model.py)
- Regression training (calibration/train.py)

### Agent 2: Corpus Pipeline Testing
**Modules:** `corpus/download.py`, `corpus/normalize.py`, `corpus/frequencies.py`  
**Current avg:** ~18%  
**Target:** 90%+  
**Estimated LOC to test:** ~320 lines

**Focus areas:**
- Download orchestration with mocked HTTP
- Text normalization workflows
- Frequency computation
- Multi-source corpus handling

### Agent 3: Calibration & Model Store Testing
**Modules:** `calibration/cli.py`, `calibration/model_store.py`, `estimators/lexile_v2_preprocessing.py`  
**Current avg:** ~25%  
**Target:** 90%+  
**Estimated LOC to test:** ~200 lines

**Focus areas:**
- CLI commands for calibration
- Model serialization/deserialization
- Text preprocessing pipelines
- NLTK integration (mocked)

### Agent 4: Utilities & Gap Filling
**Modules:** `textutils.py`, `frequency_loader.py`, `config.py`, `corpus/cli.py`, `analyzer/cli.py`  
**Current avg:** ~55%  
**Target:** 90%+  
**Estimated LOC to test:** ~150 lines

**Focus areas:**
- Text normalization utilities
- Frequency table loading
- Config edge cases
- CLI sub-commands
- Missing coverage in well-tested modules

---

## Coordination & Conflict Avoidance

### File Ownership
- Each agent "owns" their assigned test files
- No two agents modify the same test file
- Shared fixtures go in `tests/conftest.py` (coordinated through this document)

### Dependencies Between Agents
- **Agent 4** completes `textutils.py` tests FIRST (used by others)
- **Agent 1** and **Agent 2** work independently (no overlap)
- **Agent 3** depends on Agent 4 completing `config.py` tests

### Integration Points
- All agents update this document with:
  - ✅ Completed tasks
  - 🔄 In-progress tasks
  - ⚠️ Blockers or dependencies
  - 📊 Current coverage numbers

---

## Phase 1 Task Breakdown

### Agent 1 Tasks (Analyzer/Calibration Core)
- [ ] `test_analyzer_features.py` - Feature computation tests
- [ ] `test_analyzer_slices.py` - Sentence splitting & slice building
- [ ] `test_analyzer_model.py` - Model loading & prediction
- [ ] `test_calibration_train.py` - Training workflow tests
- [ ] Coverage checkpoint: Re-run and verify ≥70% for assigned modules

### Agent 2 Tasks (Corpus Pipeline)
- [ ] `test_corpus_download.py` - Download functions with mocked requests
- [ ] `test_corpus_normalize.py` - Text normalization workflows
- [ ] `test_corpus_frequencies.py` - Frequency computation
- [ ] Coverage checkpoint: Re-run and verify ≥70% for assigned modules

### Agent 3 Tasks (Calibration CLI & Preprocessing)
- [ ] `test_calibration_cli.py` - CLI command tests (fetch-texts, build-dataset, fit)
- [ ] `test_calibration_model_store.py` - Model save/load
- [ ] `test_lexile_v2_preprocessing.py` - Text preprocessing with mocked NLTK
- [ ] Coverage checkpoint: Re-run and verify ≥70% for assigned modules

### Agent 4 Tasks (Utilities)
- [x] `test_textutils.py` - Text normalization & tokenization helpers
- [x] `test_frequency_loader.py` - Frequency table loading
- [x] `test_config_advanced.py` - Config edge cases (extends existing tests)
- [x] `test_corpus_cli.py` - Corpus CLI commands
- [x] `test_analyzer_cli.py` - Analyzer CLI commands
- [x] Coverage checkpoint: Re-run and verify ≥90% for assigned modules

---

## Phase 2 Task Breakdown

### Agent 1 Tasks
- [ ] Integration tests for analyzer workflow (features → model → prediction)
- [ ] Edge cases: empty text, malformed input, boundary conditions
- [ ] Adjustment logic (`analyzer/adjustments.py`)

### Agent 2 Tasks
- [ ] Integration tests for full corpus pipeline
- [ ] Error handling: network failures, corrupted files
- [ ] Multi-source normalization edge cases

### Agent 3 Tasks
- [ ] CLI integration tests with temp directories
- [ ] Model versioning and compatibility
- [ ] Preprocessing with various text encodings

### Agent 4 Tasks
- [ ] Cross-module integration helpers
- [ ] Gap-filling in existing tests
- [ ] Shared fixture refinement

---

## Phase 3 Task Breakdown

### All Agents
- [ ] Review coverage reports for gaps
- [ ] Add missing edge case tests
- [ ] Improve test documentation
- [ ] Final coverage verification (≥90%)

---

## Quality Gates

### Before Starting Any Phase
- [ ] Read and understand `unit-test-policy.md`
- [ ] Read and understand `code-change.instructions.md`
- [ ] Review existing test patterns in `tests/`
- [ ] Set up local testing environment

### After Each Test File
- [ ] Run `poetry run black .`
- [ ] Run `poetry run ruff check`
- [ ] Run `poetry run pyright`
- [ ] Run `poetry run pytest` (all tests pass)
- [ ] Run `poetry run pytest --cov` (verify coverage increase)

### Before Moving to Next Phase
- [ ] All phase tasks completed
- [ ] Coverage target met
- [ ] All checks pass (Black, Ruff, Pyright, Pytest)
- [ ] Document any deviations or issues

---

## Success Metrics

### Phase 1 Complete
- Overall coverage: ≥70%
- All assigned modules: ≥70%
- Zero test failures
- All code quality checks pass

### Phase 2 Complete
- Overall coverage: ≥85%
- All assigned modules: ≥85%
- Integration tests pass
- Error handling verified

### Phase 3 Complete
- Overall coverage: ≥90%
- All modules: ≥85% (with documented exceptions)
- Comprehensive edge case coverage
- Full CI/CD pipeline passing

---

## Reporting Template

### Daily Update (to be appended by each agent)

**Date:** YYYY-MM-DD  
**Agent:** [1/2/3/4]  
**Phase:** [1/2/3]

**Completed:**
- List of completed test files
- Coverage increase achieved

**In Progress:**
- Current work

**Blockers:**
- Any issues or dependencies

**Coverage Snapshot:**
- Module X: XX% → YY%
- Module Y: XX% → YY%

### Agent 4 Update - 2025-12-02

**Phase:** 1

**Completed:**
- `tests/test_textutils.py` ✅
- `tests/test_frequency_loader.py` ✅
- `tests/test_config_advanced.py` ✅
- `tests/test_corpus_cli.py` ✅
- `tests/test_analyzer_cli.py` ✅

**Coverage Snapshot:**
- `textutils.py`: 44% → 100%
- `frequency_loader.py`: 50% → 100%
- `config.py`: 84% → 100%
- `corpus/cli.py`: 68% → 100%
- `analyzer/cli.py`: 46% → 100%

**In Progress:**
- Phase 1 complete. Ready for Phase 2 tasks.

**Blockers:**
- None.

### Agent 1 Update - 2025-12-03

**Phase:** 1 (Complete)

**Completed:**
- `tests/test_analyzer_adjustments.py` ✅ (NEW - 9 tests)
- `tests/test_analyzer_slices.py` ✅ (NEW - 30 tests)
- `tests/test_analyzer_features.py` ✅ (NEW - 17 tests)
- `tests/test_analyzer_model.py` ✅ (NEW - 13 tests)
- `tests/test_calibration_featureset.py` ✅ (NEW - 20 tests)
- `tests/test_calibration_train.py` ✅ (NEW - 23 tests)

**Coverage Snapshot:**
- `analyzer/adjustments.py`: 25% → 100%
- `analyzer/slices.py`: 22% → 100%
- `analyzer/features.py`: 40% → 100%
- `analyzer/model.py`: 40% → 100%
- `calibration/train.py`: 38% → 100%
- `calibration/featureset.py`: 33% → 100%

**Overall Coverage:**
- 58% → 65% (112 new tests added)

**Quality Checks:**
- ✅ black formatting
- ✅ ruff linting
- ✅ pyright type checking
- ✅ pytest (247 tests, all passing)

**In Progress:**
- Phase 1 complete. All assigned modules at 100% coverage.

**Blockers:**
- None.

---

## Risk Mitigation

### Risk: Agent conflicts on shared files
**Mitigation:** Clear file ownership, coordinated fixture updates

### Risk: Coverage regression in existing tests
**Mitigation:** Always run full test suite before committing

### Risk: Flaky tests or external dependencies
**Mitigation:** Comprehensive mocking, deterministic test data

### Risk: Deadline pressure leading to poor test quality
**Mitigation:** Enforce quality gates, prioritize correctness over speed

---

## Notes

- This is a living document. Agents should update it as work progresses.
- Weekly sync recommended to address blockers and adjust priorities.
- Coverage numbers are guidelines, not absolute requirements. Quality > quantity.
- Document any modules that cannot feasibly reach 90% with justification.
