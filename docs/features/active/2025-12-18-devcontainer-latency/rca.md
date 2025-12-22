Below is the same content rewritten in a **Root Cause Analysis (RCA)** format that is commonly used in engineering, SRE, and platform teams. This is suitable for a GitHub issue, post-incident note, or internal engineering log.

---

# Root Cause Analysis (RCA)

## Issue Title

Pytest extremely slow in VS Code dev container due to Task Explorer extension causing filesystem I/O contention

---

## Impact

* Pytest test collection for ~521 tests took **~11–12 seconds** in a VS Code dev container.
* The same test collection took **~1.53 seconds** when run locally on the host.
* Developer productivity was significantly reduced due to long test startup and execution times.
* The issue affected all pytest runs inside the dev container environment.

---

## Detection

* The issue was detected during routine development when pytest runs in the dev container felt significantly slower than expected.
* The slowdown was reproducible and consistent across runs.
* Comparison against local execution confirmed the performance regression was container-specific.

---

## Timeline / Investigation Summary

### Step 1 — Establish Baseline

**Action:** Measured pytest collection time in the dev container using `--collect-only`.

**Result:** ~11–12 seconds.

**Conclusion:** Collection overhead was abnormally high.

---

### Step 2 — Rule Out Pytest Plugin Autoload

**Action:** Disabled pytest plugin autoload (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).

**Result:** ~11.0 seconds (no material change).

**Conclusion:** Pytest plugins were not the primary cause.

---

### Step 3 — Rule Out Excessive Test Discovery Scope

**Action:** Restricted pytest discovery explicitly to the `tests/` directory.

**Result:** ~11.3 seconds.

**Conclusion:** Repo traversal was not the primary cause.

---

### Step 4 — Rule Out Pytest Cache Overhead

**Action:** Redirected pytest cache to `/tmp` via `cache_dir`.

**Result:** ~11.7 seconds.

**Conclusion:** Pytest cache writes were not the bottleneck.

---

### Step 5 — Rule Out `conftest.py` / Fixture Initialization

**Action:** Disabled all `conftest.py` loading (`--noconftest`).

**Result:** ~15.8 seconds (slower).

**Conclusion:** `conftest.py` was not responsible for the slowdown.

---

### Step 6 — Compare Against Local Execution

**Action:** Ran the same pytest collection locally on the host.

**Result:** ~1.53 seconds.

**Conclusion:** The slowdown was isolated to the dev container environment.

---

### Step 7 — Investigate External Workspace Scanning

**Action:** Disabled the **Task Explorer** VS Code extension in the dev container.

**Result:** Pytest performance improved significantly after disabling the extension.

**Conclusion:** External workspace scanning was causing filesystem contention.

---

## Root Cause

The **Task Explorer VS Code extension** was continuously scanning the workspace to discover runnable tasks. In the dev container environment—where the workspace is accessed through a mounted filesystem—this resulted in heavy filesystem I/O contention.

Pytest test collection is highly sensitive to filesystem latency due to large numbers of small file reads and metadata operations. Task Explorer’s background scanning significantly amplified this cost, leading to an ~8× slowdown compared to local execution.

---

## Contributing Factors

* Dev container workspace mounted from the host filesystem.
* Task Explorer performing repeated background scans.
* Pytest’s inherently metadata-heavy test collection process.
* Large repository with many Python files.

---

## Resolution

* Disabled the Task Explorer extension in the dev container environment.
* Pytest collection and execution times returned closer to local performance.

---

## Preventative Measures

* Avoid enabling aggressive workspace-scanning extensions inside dev containers.
* Prefer explicit test runners (CLI or VS Code’s Python test integration).
* Exclude large directories and artifacts from any task or file discovery tools.
* Treat dev container performance issues as potentially caused by **editor extensions**, not just container or tooling configuration.

---

## Lessons Learned

* Dev container performance issues are often caused by **external tooling interacting with slow filesystem mounts**, not by the application or test framework itself.
* Comparing local vs container execution early is a high-signal diagnostic step.
* Systematically ruling out internal causes (plugins, config, cache, fixtures) helps surface external interference quickly.
