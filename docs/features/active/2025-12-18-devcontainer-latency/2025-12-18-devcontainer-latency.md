# devcontainer-latency (Issue #55)

- Date captured: 2025-12-18
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/devcontainer-latency/ (Issue #55)

- Issue: #55
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/55
- Last Updated: 2025-12-19
## Summary

Pytest collection in the VS Code dev container is ~8× slower (11–12s) than on the host (~1.53s) because the Task Explorer extension causes heavy workspace scanning and filesystem I/O contention.

## Environment

- OS/version: VS Code dev container on Linux host (workspace mounted from host)
- Python version: 3.12.x (Poetry environment)
- Command/flags used: `poetry run pytest --collect-only` (also standard pytest runs)
- Data source or fixture: Lexile Corpus Tuner repo workspace

## Steps to Reproduce

1. Open the repo inside the VS Code dev container with the Task Explorer extension enabled.
2. Run `poetry run pytest --collect-only` (or a normal pytest run) from the workspace root.
3. Observe collection taking ~11–12 seconds.
4. Disable the Task Explorer extension and rerun to confirm collection time drops substantially.

## Expected Behavior

Pytest collection and startup in the dev container should be close to host performance (around 1–2 seconds for ~521 tests) without significant extension-induced overhead.

## Actual Behavior

Pytest collection inside the dev container consistently takes ~11–12 seconds for ~521 tests, compared to ~1.53 seconds on the host. Disabling Task Explorer restores normal performance.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:
	- Dev container: `poetry run pytest --collect-only` → ~11–12s
	- Host: `poetry run pytest --collect-only` → ~1.53s
	- With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, restricted discovery, cache_dir=/tmp, and --noconftest: still ~11–12s

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Root cause: Task Explorer extension continuously scans the workspace to discover tasks; on the dev container’s mounted filesystem this creates heavy I/O contention, making pytest’s metadata-heavy collection far slower.

Contributing factors:
- Workspace mounted from host into the dev container (higher FS latency).
- Task Explorer background scans overlapping with pytest file metadata reads.
- Large repository with many Python files.

Timeline (abridged):
- Baseline: ~11–12s collection in container.
- Disabled pytest autoload/plugins/scopes/cache/conftest: no improvement (still ~11s; --noconftest slower).
- Host comparison: ~1.53s.
- Disabling Task Explorer: restores normal performance → confirmed source of contention.

## Proposed Fix / Validation Ideas

Actions/resolution:
- Migrate the *code* to WSL2 (small) 
- Keep *artifacts* on externally blind mounted directory (large)

Validation:
- Re-run `poetry run pytest --collect-only` to confirm collection time returns to ~1–2s.
- Spot-check a full pytest run to ensure end-to-end performance is improved.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [x] Move to active fix folder / branch

---

## Further Investigation

### Prompt 01

>I am using vs code with github copilot. The codex-max agent is editing files, but it will sometimes open a file in the browser, make changes that don't get reflected in the browser, and then generate an error because what is open in the browser conflicts with what is saved. What might be causing this behavior and how would I correct it?

### GPT-5.2 Extended Thinking

>What you’re describing is almost always a **save-conflict (“dirty write”) + stale snapshot** problem: the file on disk is being changed by *something other than the currently-open editor buffer*, and VS Code (correctly) refuses to blindly overwrite it. VS Code explicitly blocks saving in this situation to prevent data loss. ([Visual Studio Code][1])
>
>On top of that, Copilot Agent/Edit workflows can involve **intermediate “pending changes” snapshots**; if the underlying file changes while the agent is still working (or you switch chat sessions), you can end up with edits being applied against an older revision. This has been reported as a VS Code/Copilot bug pattern in longer multi-step sessions. ([GitHub][2])
>
>## Why it happens (most common causes)
>
>1. **The file is “dirty” in VS Code, so VS Code will not auto-refresh it**
>
>   * VS Code won’t refresh an open file from disk if you have unsaved changes in that file. So an agent (or formatter) can write to disk, but your open editor tab stays on an older in-memory version until you resolve the conflict. ([Stack Overflow][3])
>
>2. **A “save participant” rewrites the file after the agent edits it**
>
>   * Format-on-save / fix-on-save / code actions on save can modify the file contents *during* save. VS Code calls these “Save Participants,” and they can change the file contents on save. ([Visual Studio Code][4])
>   * Result: Copilot writes → formatter rewrites → Copilot (or VS Code) now sees a mismatch between what it thought it edited vs what’s on disk.
>
>3. **Two writers are editing the same workspace**
>
>   * Common culprits: a second VS Code window, a devcontainer + host editing the same mounted folder, a build/watch process that rewrites generated files, or cloud sync (OneDrive/Dropbox/rclone/Google Drive). These environments are notorious for repeated “compare file” prompts because the file changes outside the editor. ([GitHub][5])
>   * Remote/devcontainer file systems can also produce “File Modified Since / content is newer” style conflicts under rapid edits. ([GitHub][6])
>
>4. **Copilot agent state management bugs**
>
>   * There are documented issues where “Keep Changes” or switching chat sessions can reapply older diffs, resurrect “ghost” file states, or otherwise desynchronize the file list and actual workspace state. ([GitHub][2])
>   * Related reports include Copilot operating on an older file version and “discard/keep” behaving unexpectedly. ([GitHub][7])
>
>## How to correct it (practical, high-signal steps)
>
>### A. First isolate the trigger (5 minutes)
>
>1. **Turn off Auto Save** (File → Auto Save) temporarily.
>2. Temporarily disable save mutators:
>
>   * `"editor.formatOnSave": false`
>   * `"editor.codeActionsOnSave": {}` (or disable the specific fixers)
>3. Run the agent again.
>   If the issue disappears, you’ve confirmed a formatter/fix-on-save (or similar save participant) is racing the agent. VS Code explicitly allows save participants to change file contents during saving, so this is a common root cause. ([Visual Studio Code][4])
>
>### B. Adopt an “agent-safe” workflow
>
>* **Before starting the agent**: `File: Save All` and make sure the target files are not dirty.
>* **While the agent runs**: do not manually edit the same files, and avoid switching branches.
>* **After the agent finishes**: click Keep/Apply, *then* run formatters/tests as a separate step.
>
>This avoids the exact scenario where VS Code refuses to refresh a dirty buffer while the disk copy changes. ([Stack Overflow][3])
>
>### C. Eliminate multi-writer situations
>
>* Ensure you don’t have the repo open in **two VS Code windows** (or VS Code + another editor).
>* If using **Dev Containers / WSL**: avoid editing the same files from both the container context and the host filesystem at the same time.
>* If the repo is under **cloud sync** (OneDrive/Dropbox/rclone): exclude the repo folder or move the working copy to a non-synced path. Save-conflict spam is a known symptom in synced/remote filesystems. ([GitHub][5])
>
>### D. Mitigate Copilot snapshot/session issues
>
>* Keep agent sessions shorter and “commit” edits (Keep Changes) more frequently.
>* Avoid switching between old Copilot chat sessions while you have pending edits; there are reports this can revert/restore file states. ([GitHub][8])
>* If you suspect you hit the “superseded edits” bug: start a **new chat session** and re-run the request on the current file state. ([GitHub][2])
>
>### E. Optional “reduce friction” setting (not a real fix)
>
>If you understand the risk and just want VS Code to stop blocking saves, you can set:
>
>```json
>"files.saveConflictResolution": "overwriteFileOnDisk"
>```
>
>VS Code documents this as a way to bypass the save-conflict flow (default is `askUser`). This can prevent interruptions, but it can also overwrite real external changes, so I treat it as a workaround, not a cure. ([Visual Studio Code][4])
>
>## If you want a fast diagnosis from your environment
>
>If you tell me whether you’re editing in **local Windows**, **WSL**, **Dev Container**, or a **cloud-synced folder**, and whether you have **format-on-save / fix-on-save** enabled, I can narrow this to the top 1–2 most likely causes and give you a minimal `settings.json` profile to validate the fix.
>
>[1]: https://code.visualstudio.com/docs/getstarted/tips-and-tricks "Visual Studio Code tips and tricks"
>[2]: https://github.com/microsoft/vscode/issues/265794 "Copilot Agent applies superseded edits from conversation history, corrupting file state · Issue #265794 · microsoft/vscode · GitHub"
>[3]: https://stackoverflow.com/questions/30078077/visual-studio-code-auto-refresh-file-changes "Visual Studio Code: Auto-refresh file changes - Stack Overflow"
>[4]: https://code.visualstudio.com/updates/v1_42 "January 2020 (version 1.42)"
>[5]: https://github.com/microsoft/vscode/issues/89886 "The VS Code asks to compare files all time. · Issue #89886 · microsoft/vscode · GitHub"
>[6]: https://github.com/microsoft/vscode-remote-release/issues/2256 "The content of the file is newer when on remote container · Issue #2256 · microsoft/vscode-remote-release · GitHub"
>[7]: https://github.com/microsoft/vscode-copilot-release/issues/3492 "Copilot using old file version and discarding resetting the whole file · Issue #3492 · microsoft/vscode-copilot-release · GitHub"
>[8]: https://github.com/microsoft/vscode/issues/258001 "VS Code Copilot Recently Edited Files List Causes Performance Issues and File Restoration Problems · Issue #258001 · microsoft/vscode · GitHub"

### Prompt 02

>I am in a devcontainer that is blind mounted to my local harddrive. All files were saved and closed before the start of the session. The agent then opened the files that created the "dirty" file

### GPT5.2 Extended Thinking

>Given your setup (Dev Container + **Windows-host bind mount**), the most likely root cause is **missed / delayed file-watcher events across the Windows → WSL2 → container mount boundary**. In that failure mode, Copilot Agent (or another process) updates the file **on disk**, but the VS Code editor buffer that just opened the file does **not** get a timely “file changed” notification, so it continues operating on a stale snapshot. When a save happens later, VS Code detects the on-disk content is newer and raises the “prevent dirty write” / “content is newer” conflict. This exact watcher impairment for **Windows-sourced folders mounted into devcontainers** is called out in the VS Code Remote Containers issue tracker. ([GitHub][1])
>
>Separately (and sometimes compounding the above), VS Code explicitly allows **Save Participants** (format-on-save / fix-on-save) to rewrite a file during save, which can also produce “newer on disk than your buffer” conflicts if something else saves immediately after the agent’s edit. ([Visual Studio Code][2])
>The Copilot extension itself has also had “content is newer” save-failure reports. ([GitHub][3])
>
>## What it means that “the agent opened the file and created the dirty file”
>
>That part is normal: the agent must open an editor buffer and apply edits, which marks it dirty. The problem is that **something else is modifying the underlying file after the buffer is opened**, and your environment isn’t reliably telling VS Code “reload this,” so you get a conflict.
>
>## The most reliable correction (recommended)
>
>### 1) Stop editing the repo from a Windows bind mount
>
>Put the *source* in a filesystem with reliable Linux file notifications, then reopen in the container:
>
>**Option A — Store the repo inside WSL2, then “Reopen in Container”**
>VS Code explicitly documents opening a WSL2 folder and then reopening it in a container. ([Visual Studio Code][4])
>This avoids the “Windows folder mounted into WSL into container” watcher path that is known to be problematic. ([GitHub][1])
>
>**Option B — Clone the repo into a container volume (no bind mount)**
>VS Code’s devcontainer docs support cloning into a volume, and the advanced config docs show how to switch from bind mounts to named volumes using `workspaceMount`. ([Visual Studio Code][5])
>
>### 2) Keep your huge artifacts on the host, but mount them separately
>
>If your concern is terabyte-scale artifacts, split “code” from “data”:
>
>* Keep **code** in WSL2 or a container volume (reliable watchers).
>* Keep **artifacts** on the host and bind-mount only that directory into the container using `mounts`.
>
>VS Code documents adding additional bind mounts via the `mounts` property. ([Visual Studio Code][6])
>
>This gives you correctness for editing while still accessing large host-side data.
>
>## If you need a tactical workaround (not a true fix)
>
>### 3) Make conflicts less disruptive
>
>You can tell VS Code to overwrite on-disk content instead of prompting (risk: you can clobber legitimate external changes):
>
>```json
>"files.saveConflictResolution": "overwriteFileOnDisk"
>```
>
>VS Code documents this setting as the way to bypass the conflict prompt. ([Visual Studio Code][2])
>
>## “Quick isolate” checks (to confirm the cause)
>
>1. **Disable save mutators** in the devcontainer workspace settings for one run:
>
>* `editor.formatOnSave = false`
>* disable any `editor.codeActionsOnSave` fixers
>  Save participants can change content during save. ([Visual Studio Code][2])
>
>2. Verify you don’t have a second writer:
>
>* another VS Code window on the same repo
>* host-side formatters/watchers (prettier/eslint/ruff/black “watch” modes)
>* cloud sync / backup tooling touching the repo
>
>If none of those are present, the remaining likely culprit is the bind-mount watcher reliability issue described above. ([GitHub][1])
>
>---
>
>If you want the fastest path to “never see this again” while keeping host-side artifacts: move the repo into **WSL2 storage** and then **Dev Containers: Reopen in Container**, and mount the artifacts folder separately via `mounts`. That aligns with the documented WSL2→container workflow ([Visual Studio Code][4]) and avoids the known watcher impairment for Windows-mounted workspaces. ([GitHub][1])
>
>[1]: https://github.com/microsoft/vscode-remote-release/issues/9399 "Source Control in devcontainers does not auto-refresh; requires manually pressing refresh button · Issue #9399 · microsoft/vscode-remote-release · GitHub"
>[2]: https://code.visualstudio.com/updates/v1_42 "January 2020 (version 1.42)"
>[3]: https://github.com/microsoft/vscode-copilot-release/issues/8330 "Fails to save because \"The content is newer\" · Issue #8330 · microsoft/vscode-copilot-release · GitHub"
>[4]: https://code.visualstudio.com/docs/remote/wsl "Developing in WSL"
>[5]: https://code.visualstudio.com/docs/devcontainers/containers "Developing inside a Container"
>[6]: https://code.visualstudio.com/remote/advancedcontainers/add-local-file-mount "Add another local file mount"
