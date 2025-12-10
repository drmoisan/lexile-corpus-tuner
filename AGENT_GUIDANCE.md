# Agent Guidance: Ensuring Complete Policy Compliance

**Date Created:** 2025-12-10  
**Context:** Lessons learned from fix-all.ps1 unit test implementation  
**Audience:** Future GitHub Copilot agents (cloud and local)

---

## Problem Identified

During the fix-all.ps1 unit test implementation, the initial audit **missed two critical policy documents**:
- ❌ `general-code-change.instructions.md` 
- ❌ `powershell-code-change.instructions.md`

The agent only audited against:
- ✅ `general-unit-test.instructions.md`
- ✅ `powershell-unit-test.instructions.md`

**This was a significant oversight** because the code change policies contain **mandatory requirements** including:
- Required toolchain execution (format → lint → type check → test)
- Explicit validation and reporting requirements
- "After Making Changes" checklist requirements

---

## Root Cause Analysis

### Why This Happened

1. **Narrow focus on "unit test" policies**: The agent interpreted "write unit tests" as only requiring unit test policy compliance
2. **Incomplete policy discovery**: Did not systematically check **all** `.github/instructions/*.instructions.md` files
3. **Missing critical requirement**: The general-code-change policy states **"When implementing any code, tests, tasks, or scripts, you **must** adhere to these repo policies"**
4. **Cloud agent behavior**: As noted by the user, cloud agents appear to violate policies more frequently than local agents

---

## Required Future Behavior

### STEP 1: Always Start with Complete Policy Discovery

**BEFORE starting any work**, you MUST:

```bash
# List ALL policy files that may apply
find .github/instructions -name "*.instructions.md" -type f

# For this repo, that yields:
# - general-code-change.instructions.md           ← ALWAYS applies to ANY code
# - powershell-code-change.instructions.md        ← Applies to .ps1/.psm1 files
# - python-code-change.instructions.md            ← Applies to .py files
# - general-unit-test.instructions.md             ← Applies when writing tests
# - powershell-unit-test.instructions.md          ← Applies when writing .ps1 tests
# - python-unit-test.instructions.md              ← Applies when writing .py tests
# - github-actions.instructions.md                ← Applies to .github/workflows/*.yml
```

### STEP 2: Determine Which Policies Apply

For **ANY** code change (including tests), you MUST check:

1. **General Code Change Policy** - ALWAYS APPLIES
   - File: `general-code-change.instructions.md`
   - Applies to: All code, tests, scripts, tasks, modules, packages

2. **Language-Specific Code Change Policy** - IF writing code in that language
   - PowerShell: `powershell-code-change.instructions.md` (for .ps1, .psm1 files)
   - Python: `python-code-change.instructions.md` (for .py files)

3. **General Unit Test Policy** - IF writing tests
   - File: `general-unit-test.instructions.md`
   - Applies to: All test files regardless of language

4. **Language-Specific Unit Test Policy** - IF writing tests in that language
   - PowerShell: `powershell-unit-test.instructions.md` (for .Tests.ps1 files)
   - Python: `python-unit-test.instructions.md` (for test_*.py files)

5. **Special Domain Policies** - IF modifying specific areas
   - GitHub Actions: `github-actions.instructions.md` (for .github/workflows/*.yml)

### STEP 3: Read Policies in Correct Order

Per `general-code-change.instructions.md`, the authority hierarchy is:

1. **First**: General code change policy
2. **Second**: Language-specific code change policy  
3. **Third**: General unit test policy
4. **Fourth**: Language-specific unit test policy
5. **Fifth**: Operational guidance (CI docs, developer tooling)

Read them **in this order** and apply rules as layers.

### STEP 4: Execute the Full Toolchain

The **General Code Change Policy Section 8** requires:

#### Mandatory Toolchain Loop

You **MUST** run in this exact order:

1. **Formatting**
2. **Linting** 
3. **Type checking** (skip if N/A)
4. **Testing**

**Critical Rules:**
- If ANY step fails or changes files → restart from step 1
- Continue looping until all 4 steps pass in a single iteration
- Explicitly report which commands you ran
- Explicitly report that all steps passed

#### For PowerShell Changes

Per `powershell-code-change.instructions.md` Section 4:

```bash
# Step 1: Format
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."

# Step 2: Lint/Analyze
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."

# Step 3: Type check (N/A for PowerShell)

# Step 4: Test
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."
```

#### For Python Changes

Per `python-code-change.instructions.md`:

```bash
# Step 1: Format
poetry run black .

# Step 2: Lint
poetry run ruff check

# Step 3: Type check
poetry run pyright

# Step 4: Test
poetry run pytest --cov=src/lexile_corpus_tuner --cov-report=term-missing
```

### STEP 5: Document Toolchain Execution

In your final report/commit/PR, you MUST explicitly state:

```
## Toolchain Validation

Executed full toolchain in this order:
1. ✅ Formatting: [command] - Result: [no changes needed / changes applied]
2. ✅ Linting: [command] - Result: [no findings / fixed X issues]
3. ✅ Type checking: [command] - Result: [passed / N/A]
4. ✅ Testing: [command] - Result: [X/X tests passing]

All four steps completed successfully in a single pass with no errors.
```

### STEP 6: Create Comprehensive Audit

When asked to audit your work, you MUST:

1. **List ALL applicable policies** at the top
2. **Create sections for EACH policy** with subsections for each policy section
3. **Provide evidence** for each requirement (not just "PASS" - show HOW you passed)
4. **Include toolchain execution details** with actual commands and results
5. **Document any exceptions or gaps** explicitly

---

## Checklist for ANY Code Change

Use this checklist for **every** coding task:

### Pre-Work
- [ ] List all `.instructions.md` files in `.github/instructions/`
- [ ] Identify which policies apply to this change
- [ ] Read applicable policies in correct order (general → specific → test → operational)
- [ ] Document the plan

### During Work
- [ ] Follow design principles from general code change policy
- [ ] Follow language-specific patterns
- [ ] Follow test patterns if writing tests
- [ ] Keep files under 500 lines
- [ ] Use descriptive names

### Post-Work (MANDATORY)
- [ ] Run formatter - document result
- [ ] Run linter - document result  
- [ ] Run type checker (if applicable) - document result
- [ ] Run tests - document result
- [ ] If any step failed/changed files → restart from formatter
- [ ] Continue until all pass in single iteration
- [ ] Explicitly report commands and results

### Documentation (MANDATORY)
- [ ] Create/update audit document if requested
- [ ] Include ALL applicable policies in audit
- [ ] Provide evidence for each requirement
- [ ] Document toolchain execution
- [ ] Update PR description with compliance statement

---

## Example: Writing PowerShell Tests

### What You Must Check

✅ **4 policies apply:**

1. `general-code-change.instructions.md` - Section 8 requires toolchain
2. `powershell-code-change.instructions.md` - Section 4 specifies PowerShell toolchain
3. `general-unit-test.instructions.md` - Core principles, coverage, structure
4. `powershell-unit-test.instructions.md` - Pester v5, file organization

### What You Must Execute

```bash
# 1. Format
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command \
  "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."

# 2. Analyze  
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command \
  "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."

# 3. Test
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command \
  "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."
```

### What You Must Document

In audit/report:
- ✅ Section for General Code Change Policy with toolchain results
- ✅ Section for PowerShell Code Change Policy with toolchain results  
- ✅ Section for General Unit Test Policy with coverage evidence
- ✅ Section for PowerShell Unit Test Policy with Pester evidence
- ✅ Explicit statement: "All four steps completed successfully in a single pass"

---

## Common Mistakes to Avoid

### ❌ DON'T: Focus only on test policies when writing tests
**Why:** Code change policies apply to ALL code, including tests

### ❌ DON'T: Skip toolchain execution
**Why:** Section 8 of general-code-change policy makes it mandatory

### ❌ DON'T: Run tools but not document execution
**Why:** Policy requires explicit reporting of commands and results

### ❌ DON'T: Assume audit only needs test policies
**Why:** Audit must cover ALL applicable policies

### ❌ DON'T: Say "I ran the checks" without specifics
**Why:** Must state exact commands and results

### ✅ DO: Always list ALL applicable policies first
### ✅ DO: Execute full toolchain and document it
### ✅ DO: Create comprehensive audit covering all policies
### ✅ DO: Provide evidence, not just checkmarks

---

## Quick Reference Card

**For ANY code change:**
1. Find all `.instructions.md` files
2. Determine which apply (general code + language code + general test + language test)
3. Read in order: general → language → test
4. Run full toolchain: format → lint → type → test
5. Document execution with commands and results
6. Create audit covering ALL applicable policies

**Toolchain commands for PowerShell:**
```bash
Invoke-PoshQCFormat -Root .
Invoke-PoshQCAnalyze -Root .
Invoke-PoshQCTest -Root .
```

**Toolchain commands for Python:**
```bash
poetry run black .
poetry run ruff check
poetry run pyright
poetry run pytest --cov=...
```

**If any step fails or changes files:**
→ Restart from step 1 (formatting)

**When documenting:**
→ List ALL policies that apply
→ Show evidence for each requirement
→ Include exact commands and results
→ Explicitly state all steps passed in single iteration

---

## Conclusion

The `.github/instructions/*.instructions.md` ecosystem is **comprehensive and mandatory**. Missing any policy document results in incomplete work. Cloud agents must be especially diligent about:

1. **Discovering all applicable policies** before starting work
2. **Executing the complete toolchain** as specified in code change policies
3. **Documenting toolchain execution** with specific commands and results
4. **Auditing against ALL applicable policies**, not just the obvious ones

By following this guidance, future agents will ensure complete policy compliance from the start.

---

**Last Updated:** 2025-12-10  
**Related Issue:** fix-all.ps1 unit tests (initial audit missed 2 of 4 applicable policies)  
**Status:** Active guidance for all future agent work in this repository
