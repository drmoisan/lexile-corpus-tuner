# Policy Compliance Audit: [Component Name]

> **Template Usage Instructions:**
> 
> This template is for documenting policy compliance during agent-driven development. Use it to audit any code changes (features, bugfixes, refactors) or test additions.
> 
> **How to use:**
> 1. Copy this template to your working directory (feature folder, PR docs, etc.)
> 2. Replace all `[placeholders]` with actual values
> 3. Use `[✅/❌/N/A]` and `[PASS/FAIL/N/A]` to mark status
> 4. Delete inapplicable sections (e.g., delete Python sections for PowerShell work)
> 5. Fill in evidence based on actual test runs and code inspection
> 6. Complete before submitting PR or marking work as done
> 
> **When to use:**
> - After completing any code changes (required by policy)
> - Before submitting PRs
> - When adding or updating tests
> - During bugfix implementation (see Bugfix Playbook)
> - For feature development (see Feature Playbook)
> 
> **Delete this instruction block before finalizing the audit.**

---

**Audit Date:** [YYYY-MM-DD]  
**Test File:** `[path/to/test_file]` (if testing)  
**Code Under Test:** `[path/to/code_file]`  
**Total Tests:** [N] (if testing)  
**Test Result:** [✅/❌] [N] Passed, [N] Failed (if testing)

---

## Executive Summary

[Summarize overall compliance status and key findings. List which policy documents were evaluated. Provide a brief overview of what was tested and the outcome of toolchain execution.]

**Policy documents evaluated:**
- [✅/❌] `general-code-change.instructions.md` (if applicable)
- [✅/❌] `python-code-change.instructions.md` OR `powershell-code-change.instructions.md` (language-specific)
- [✅/❌] `general-unit-test.instructions.md` (if testing)
- [✅/❌] `python-unit-test.instructions.md` OR `powershell-unit-test.instructions.md` (if testing)

[Brief summary of test coverage, toolchain results, and compliance status.]

---

## 1. General Unit Test Policy Compliance

### 1.1 Core Principles

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Independence** - Tests run in any order | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how tests demonstrate independence. Do they share state? Can they run in parallel? Do they use proper setup/teardown?] |
| **Isolation** - Each test targets single behavior | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how each test targets a single unit. List test organization structure. Explain how tests are grouped.] |
| **Fast Execution** - Tests complete quickly | [✅/❌/N/A] [PASS/FAIL/N/A] | [Report total execution time, average per test, discovery time. Identify any slow tests.] |
| **Determinism** - Consistent results | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how tests avoid randomness, time dependencies, and external I/O. List mocking strategy.] |
| **Readability & Maintainability** - Clear structure | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test naming conventions, organization patterns, and documentation approach.] |

### 1.2 Coverage and Scenarios

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Comprehensive Coverage** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all functions/classes/modules tested with line ranges and test counts. Report function coverage percentage and line coverage percentage. Note any untested code and justification.] |
| **Positive Flows** - Valid inputs | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all positive test scenarios. Describe normal operation paths tested. Show that happy paths are covered.] |
| **Negative Flows** - Invalid inputs | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all negative test scenarios. Show that invalid inputs are handled correctly. Describe error conditions tested.] |
| **Edge Cases** - Boundary conditions | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all edge case scenarios. Show that boundary conditions are tested (empty strings, nulls, limits, special characters, etc.).] |
| **Error Handling** - Error paths | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all error handling scenarios. Show that exceptions and error conditions are properly tested.] |
| **Concurrency** - If applicable | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe concurrency testing or explain why it's not applicable. If applicable, describe how race conditions and thread safety are tested.] |
| **State Transitions** - If applicable | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe state transition testing or explain why it's not applicable. If applicable, show that all state transitions are tested.] |

### 1.3 Test Structure and Diagnostics

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Clear Failure Messages** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe assertion strategy and failure message quality. Show examples of diagnostic output from failing tests.] |
| **Arrange-Act-Assert Pattern** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how tests follow AAA pattern. Explain setup, execution, and assertion phases.] |
| **Document Intent** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Show test naming conventions and documentation approach. Provide examples of test names and describe how tests are grouped.] |

### 1.4 External Dependencies and Environment

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Avoid External Dependencies** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List any external dependencies (databases, networks, APIs, processes, filesystem). If none, explicitly state this.] |
| **Use Mocks/Stubs** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all mocked/stubbed components and why they were mocked. Explain mocking strategy.] |
| **Environment Stability** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how tests avoid environment dependencies. Note any global state, config files, or temporary files. Confirm no prohibited temporary file creation.] |

### 1.5 Policy Audit Requirement

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Pre-submission Review** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm that this audit document serves as the required policy review. Note any outstanding review items.] |

---

## 2. General Code Change Policy Compliance

### 2.1 Before Making Changes

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Clarify the objective** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe what objective was being pursued. Reference issue numbers or feature specs if applicable.] |
| **Read existing change plans** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Note whether existing change plans were reviewed. List any relevant planning documents.] |
| **Document the plan** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Reference where the plan was documented (commit messages, planning docs, etc.).] |

### 2.2 Design Principles

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Simplicity first** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how the code demonstrates simplicity. Note any complexity and its justification.] |
| **Reusability** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe what was reused and how. Note any shared patterns or helpers used.] |
| **Extensibility** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how the design supports future extension. Note any extension points.] |
| **Separation of concerns** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how different concerns are separated. Note the layering strategy.] |

### 2.3 Module & File Structure

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Cohesive modules** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe the purpose and cohesion of each module/file. Note organization strategy.] |
| **Under 500 lines** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Report line counts for all files. Note any exceptions and justifications.] |
| **Public vs internal** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe the public API surface. Note how internals are hidden.] |
| **No circular dependencies** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe the dependency graph. Confirm no circular dependencies exist.] |

### 2.4 Naming, Docs, and Comments

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Descriptive names** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe naming conventions used. Provide examples of well-named elements.] |
| **Docs/docstrings** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe documentation approach. Note coverage of public APIs.] |
| **Comment why, not what** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe commenting approach. Provide examples of good comments explaining rationale.] |

### 2.5 After Making Changes - Toolchain Execution

| Requirement | Status | Evidence |
|------------|--------|----------|
| **1. Formatting** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `[formatting command]`<br>**Result:** [Describe result - no changes needed, or changes applied] |
| **2. Linting** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `[linting command]`<br>**Result:** [Describe result - no findings, or findings fixed] |
| **3. Type checking** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `[type checking command]`<br>**Result:** [Describe result - no errors, or errors fixed, or N/A for PowerShell] |
| **4. Testing** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `[test command]`<br>**Result:** [Describe result - all tests passing] |
| **Full toolchain loop** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm all steps completed in single pass, or describe how many iterations were needed.] |
| **Explicit reporting** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm commands and results are documented in this audit and/or commit messages.] |

### 2.6 Summarize and Document

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Summarize changes** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Summarize what was changed. Reference PR description and commit messages.] |
| **Design choices explained** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Explain key design decisions. Note any alternatives considered.] |
| **Update supporting documents** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List any documentation updated (README, specs, runbooks, etc.).] |
| **Provide next steps** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe what should happen next. Note completion status and readiness for merge/deployment.] |

---

## 3. Language-Specific Code Change Policy Compliance

**Note:** Use Section 3A for Python OR Section 3B for PowerShell. Delete the section that doesn't apply.

---

### Section 3A: Python Code Change Policy Compliance

#### 3A.1 Tooling & Baseline

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Formatting with Black** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `poetry run black .`<br>**Result:** [Describe result] |
| **Linting with Ruff** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `poetry run ruff check`<br>**Result:** [Describe result] |
| **Type checking with Pyright** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `poetry run pyright`<br>**Result:** [Describe result] |
| **Testing with Pytest** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `poetry run pytest`<br>**Result:** [Describe result] |

#### 3A.2 Python Design & Typing

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Strong typing** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe type annotation coverage. Note any use of `Any` and justification.] |
| **Dataclasses for value objects** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe use of dataclasses. Note any value objects.] |
| **Protocols/ABCs for interfaces** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe use of protocols or abstract base classes. Note any interfaces.] |
| **Avoid utility classes** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm no static-method-only utility classes. Note use of modules with functions instead.] |

#### 3A.3 Python Error Handling

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Specific exceptions** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe exception handling approach. Confirm no broad catches.] |
| **Logging over print** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm use of logging module. Note any print statements and justification.] |
| **Invariants at construction** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how invariants are enforced in `__init__` or `__post_init__`.] |

---

### Section 3B: PowerShell Code Change Policy Compliance

#### 3B.1 Tooling & Baseline

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Formatting with Invoke-Formatter** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `Invoke-PoshQCFormat -Root .`<br>**Result:** [Describe result] |
| **Linting with PSScriptAnalyzer** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `Invoke-PoshQCAnalyze -Root .`<br>**Result:** [Describe result] |
| **Fix all findings** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe any findings and how they were resolved.] |
| **PowerShell 5.1 & 7.5+ compatible** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe compatibility testing. Note any version-specific features.] |

#### 3B.2 PowerShell Design & Safety

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Advanced functions** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe use of CmdletBinding and parameter attributes. N/A for test files.] |
| **Parameter validation** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe parameter validation attributes used. N/A for test files.] |
| **Avoid global state** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how global state is avoided. Note any legitimate global usage.] |
| **Error handling** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe error handling approach. Note use of try/catch, -ErrorAction, etc.] |

#### 3B.3 Structure, Naming, and Comments

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Cohesive and under 500 lines** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Report file line counts. Confirm all under limit.] |
| **Approved verbs** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all function names and confirm verbs are approved.] |
| **Comment why** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe commenting approach. Confirm focus on rationale.] |

#### 3B.4 Running the Toolchain

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Step 1: Format** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe execution and result.] |
| **Step 2: Analyze** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe execution and result.] |
| **Step 3: Type check** | N/A | Not applicable for PowerShell. |
| **Step 4: Test** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe execution and result.] |
| **Rerun loop if needed** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how many iterations were needed.] |

---

## 4. Language-Specific Unit Test Policy Compliance

**Note:** Use Section 4A for Python OR Section 4B for PowerShell. Delete the section that doesn't apply.

---

### Section 4A: Python Unit Test Policy Compliance

#### 4A.1 Framework and Scope

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Use Pytest** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe use of Pytest. Note any fixtures or plugins used.] |
| **Coverage expectation** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Report coverage metrics. Confirm new code has >= 90% coverage and repo-wide >= 80%.] |

#### 4A.2 Test Style and Structure

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Focused unit tests** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test focus. Confirm each test exercises single behavior.] |
| **Mocking sparingly** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe mocking strategy. List what is mocked and why.] |
| **Organization** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test organization. Confirm tests mirror code structure.] |

#### 4A.3 Naming and Readability

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Naming conventions** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test naming. Provide examples of descriptive test names.] |
| **Docstrings/comments** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe documentation approach for tests.] |

#### 4A.4 Running the Toolchain

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Use Pytest** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `poetry run pytest`<br>**Result:** [Describe test results] |
| **No Alternative Test Runners** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm only Pytest is used.] |

---

### Section 4B: PowerShell Unit Test Policy Compliance

#### 4B.1 Framework and Scope

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Use Pester v5.x** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe Pester v5 features used (BeforeAll, BeforeEach, Describe/Context/It, modern Should syntax).] |
| **Use PoshQC Configuration** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `Invoke-PoshQCTest -Root .`<br>**Config:** `scripts/powershell/PoshQC/settings/pester.runsettings.psd1`<br>[Describe configuration updates if any.] |
| **PowerShell 5.1 & 7.5+ Compatible** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe compatibility testing. Note any version-specific features.] |

#### 4B.2 Test Style and Structure

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Focused Unit Tests** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test focus. Report test distribution across functions.] |
| **Test Behavior Over Implementation** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe what behaviors are tested vs. implementation details.] |
| **Mocking Used Sparingly** | [✅/❌/N/A] [PASS/FAIL/N/A] | [List all mocked components and justification for each.] |
| **Organization** | [✅/❌/N/A] [PASS/FAIL/N/A] | **CRITICAL:** Test file location must mirror code file location.<br>**Test file:** `[path]`<br>**Code file:** `[path]`<br>[Confirm structure mirrors code location.] |

#### 4B.3 Naming and Readability

| Requirement | Status | Evidence |
|------------|--------|----------|
| **File Naming** - *.Tests.ps1 | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm file named correctly: `[name].Tests.ps1`] |
| **Describe/Context/It Structure** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe test structure: # Describe blocks, # Context blocks, # It blocks.] |
| **Logical Grouping** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe how tests are grouped. Show grouping strategy.] |
| **Docstrings/Comments** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Describe documentation approach. Note that test names should be self-documenting.] |

#### 4B.4 Running the Toolchain

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Use PoshQCTest Command** | [✅/❌/N/A] [PASS/FAIL/N/A] | **Command:** `Invoke-PoshQCTest -Root .`<br>**Result:** [Describe test results] |
| **No Alternative Test Runners** | [✅/❌/N/A] [PASS/FAIL/N/A] | [Confirm only Pester is used through PoshQC.] |

---

## 5. Test Coverage Detail

**Instructions:** Create one subsection for each function/class/module under test. Use tables to show test names, scenario types, lines covered, and status.

### [Function/Class/Module Name] ([N] tests)

| Test Name | Scenario Type | Lines Covered | Status |
|-----------|--------------|---------------|--------|
| [test name] | [Positive/Negative/Edge Case/Error Handling] | [line ranges] | [✅/❌] |
| [test name] | [Positive/Negative/Edge Case/Error Handling] | [line ranges] | [✅/❌] |
| [test name] | [Positive/Negative/Edge Case/Error Handling] | [line ranges] | [✅/❌] |

**Coverage:** [X]% of [function/class] (lines [start]-[end])

**Detailed line-by-line coverage (optional):**
- Line [N]: [✅/❌] Covered ([describe what's tested])
- Lines [N-M]: [✅/❌] Covered ([describe what's tested])
- Line [N]: ❌ Not covered ([explain why or plan to cover])

**Not covered:** [List any untested code and justification, or state "None"]

---

### [Additional Function/Class/Module] ([N] tests)

[Repeat structure above for each component tested]

---

## 6. Test Execution Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Tests | [N] | [✅/❌] |
| Tests Passed | [N] ([X]%) | [✅/❌] |
| Tests Failed | [N] | [✅/❌] |
| Execution Time | [X]s total | [✅/❌] Fast/Slow |
| Average Time per Test | [X]ms | [✅/❌] Fast/Slow |
| Discovery Time | [X]ms | [✅/❌] |
| Functions/Classes Tested | [N]/[M] ([X]%) | [✅/❌] |
| Test File Size | [N] lines | [✅/❌] Maintainable |
| Code Coverage (if applicable) | [X]% lines, [Y]% branches | [✅/❌] |

---

## 7. Code Quality Checks

**For Python:**

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| Black Formatting | `poetry run black .` | [result] | [✅/❌] |
| Ruff Linting | `poetry run ruff check` | [result] | [✅/❌] |
| Pyright Type Checking | `poetry run pyright` | [result] | [✅/❌] |
| Pytest Tests | `poetry run pytest` | [result] | [✅/❌] |

**For PowerShell:**

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| Invoke-Formatter | `Invoke-PoshQCFormat -Root .` | [result] | [✅/❌] |
| PSScriptAnalyzer | `Invoke-PoshQCAnalyze -Root .` | [result] | [✅/❌] |
| Pester Tests | `Invoke-PoshQCTest -Root .` | [result] | [✅/❌] |

**Notes:**
[Document any pre-existing failures unrelated to this work. Explain any deviations from clean runs.]

---

## 8. Gaps and Exceptions

### Identified Gaps
[List any policy requirements that are not fully met. If none, state "**None.** All policy requirements are met."]

**Example:**
- [Requirement name]: [Description of gap and plan to address]

### Approved Exceptions
[List any approved exceptions to policy requirements with justification. If none, state "**None.** No exceptions needed."]

**Example:**
- [Requirement name]: [Justification for exception and approval source]

### Removed/Skipped Tests
[List any tests that were planned but removed or skipped, with justification. If none, state "**None.** All planned tests implemented."]

**Example:**
1. **"[test name]"** - Removed in commit [hash]
   - **Reason:** [Why was it removed?]
   - **Impact:** [What coverage is lost?]
   - **Justification:** [Why is this acceptable?]

---

## 9. Summary of Changes

### Commits in This PR/Branch

[List all commits with hashes and brief descriptions]

**Example:**
1. **[hash]** - [commit message]
2. **[hash]** - [commit message]
3. **[hash]** - [commit message]

### Files Modified

[List all files that were created, modified, or deleted]

**Example:**

1. **[path/to/file]** (NEW/MODIFIED/DELETED)
   - [Description of changes]
   - [Key points]

2. **[path/to/file]** (NEW/MODIFIED/DELETED)
   - [Description of changes]
   - [Key points]

3. **[path/to/file]** (NEW/MODIFIED/DELETED)
   - [Description of changes]
   - [Key points]

---

## 10. Compliance Verdict

### Overall Status: [✅ FULLY COMPLIANT / ⚠️ PARTIALLY COMPLIANT / ❌ NON-COMPLIANT]

[Provide a brief summary of overall compliance status and any major findings.]

---

### Policy-by-Policy Summary

#### General Code Change Policy (Section 2)
- [✅/⚠️/❌] Before Making Changes: [Brief status]
- [✅/⚠️/❌] Design Principles: [Brief status]
- [✅/⚠️/❌] Module & File Structure: [Brief status]
- [✅/⚠️/❌] Naming, Docs, Comments: [Brief status]
- [✅/⚠️/❌] Toolchain Execution: [Brief status]
- [✅/⚠️/❌] Summarize & Document: [Brief status]

#### Language-Specific Code Change Policy (Section 3)

**For Python:**
- [✅/⚠️/❌] Tooling & Baseline: [Brief status]
- [✅/⚠️/❌] Python Design & Typing: [Brief status]
- [✅/⚠️/❌] Error Handling: [Brief status]

**For PowerShell:**
- [✅/⚠️/❌] Tooling & Baseline: [Brief status]
- [✅/⚠️/❌] PowerShell Design & Safety: [Brief status]
- [✅/⚠️/❌] Structure & Naming: [Brief status]
- [✅/⚠️/❌] Toolchain: [Brief status]

#### General Unit Test Policy (Section 1)
- [✅/⚠️/❌] Core Principles: [Brief status]
- [✅/⚠️/❌] Coverage & Scenarios: [Brief status]
- [✅/⚠️/❌] Test Structure: [Brief status]
- [✅/⚠️/❌] External Dependencies: [Brief status]
- [✅/⚠️/❌] Policy Audit: [Brief status]

#### Language-Specific Unit Test Policy (Section 4)

**For Python:**
- [✅/⚠️/❌] Framework & Scope: [Brief status]
- [✅/⚠️/❌] Test Style & Structure: [Brief status]
- [✅/⚠️/❌] Naming & Readability: [Brief status]
- [✅/⚠️/❌] Toolchain: [Brief status]

**For PowerShell:**
- [✅/⚠️/❌] Framework & Scope: [Brief status]
- [✅/⚠️/❌] Test Style & Structure: [Brief status]
- [✅/⚠️/❌] Naming & Readability: [Brief status]
- [✅/⚠️/❌] Toolchain: [Brief status]

---

### Metrics Summary

[Provide a bulleted summary of key metrics from Section 6]

**Example:**
- [✅/❌] [N]/[M] tests passing ([X]%)
- [✅/❌] [N]/[M] functions/classes tested ([X]%)
- [✅/❌] [X]% line coverage
- [✅/❌] Proper file organization: [describe]
- [✅/❌] All code quality checks passing
- [✅/❌] Test execution time: [X] seconds ([fast/slow])

---

### Recommendation

**[Ready for merge / Needs revision / Blocked]**

[Provide clear recommendation and any next steps. If not ready for merge, list specific items that must be addressed.]

---

## Appendix A: Test Inventory

### Complete Test List

[List all tests in a hierarchical structure that matches the test organization (Describe/Context/It or test class hierarchy)]

**Example format:**

1. [Describe/Class] › [Context/Method] › [test name]
2. [Describe/Class] › [Context/Method] › [test name]
3. [Describe/Class] › [Context/Method] › [test name]
...

**Alternative flat format:**

- [test_module.py::TestClassName::test_method_name]
- [test_module.py::TestClassName::test_method_name]
- [test_module.py::test_function_name]
...

---

## Appendix B: Toolchain Commands Reference

[Provide quick reference of all commands used in this audit]

**For Python:**
```bash
# Formatting
poetry run black .

# Linting
poetry run ruff check
poetry run ruff check --fix  # auto-fix

# Type checking
poetry run pyright

# Testing
poetry run pytest
poetry run pytest --cov=src/[package] --cov-report=term-missing
```

**For PowerShell:**
```powershell
# Formatting
Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root .

# Linting
Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root .

# Testing
Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root .
```

---

**Audit Completed By:** [Agent Name / Human Name]  
**Audit Date:** [YYYY-MM-DD]  
**Policy Version:** Current (as of audit date)
