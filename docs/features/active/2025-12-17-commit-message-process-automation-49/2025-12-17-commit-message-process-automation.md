# commit-message-process-automation (Issue #49)

- Date captured: 2025-12-17
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/commit-message-process-automation/ (Issue #49)

- Issue: #49
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/49
- Last Updated: 2025-12-17
## Problem / Why

### Problem Summary
I have a script that generates context for commit messages `scripts/dev-tools/collect-commit-context.ps1`. I also have specialized agent for writing the commit message #file:commit-steward.agent.md. Finally, I have a standard prompt that is accessed by /generate-commit-message. Is there a way to automate these steps and put the results into the messages field on the source countrol tab?

### Thinking Beast Mode - Phase 2: Problem decomposition (Complete)

- **Problem decomposition (Phase 2)**:  
  - *Workflow surface*: add a VS Code chat participant (slash command `/commit-message-generator`) so users trigger it from Copilot Chat.  
  - *Context collection*: command handler runs collect-commit-context.ps1 (PowerShell task or terminal) and captures its report text.  
  - *LLM invocation*: use the Language Model API (`lm.selectChatModels` → `model.sendRequest`) with a prompt built from the collected report plus any user overrides, targeting the specialized commit agent.  
  - *SCM integration*: once the response arrives, set `scm.SourceControl.inputBox.value` (or `scm.createSourceControl(...).inputBox.value`) so the generated commit text lands in the Source Control message field; optionally surface success/failure via notifications.  
  - *Resilience*: wrap each step with error handling, progress indication, and telemetry/logging so timeouts or script failures report cleanly.

- **Todo list**:
  - [x] Phase 2.A — Define extension architecture: activation events, contributes.chatParticipants entry, slash command metadata, and how we’ll locate/run the PowerShell script.
  - [x] Phase 2.B — Design data flow: PowerShell output capture, prompt template for `/commit-message-generator`, model selection and token budgeting, plus fallback UX if the model/tool call fails.
  - [x] Phase 2.C — Plan SCM update + messaging: decide how to acquire the active Source Control input box, confirm formatting (summary vs. body), and specify notifications/progress indicators for success, empty results, or errors.


## Proposed Behavior

### Thinking Beast Mode - Phase 3: Strategy (Complete)

#### Architecture Plan (Phase3.A)

- Create a dedicated VS Code extension under `tools/vscode/commit-steward/` with `package.json`, `tsconfig.json`, `src/extension.ts`, and compiled output in `dist/`.  
- Contribute a chat participant per the [Chat Participant API](https://github.com/microsoft/vscode-docs/blob/main/api/extension-guides/ai/chat.md) with `id` `lexile.commitSteward`, display name `Commit Steward`, description “Generate audit-grade commit text”, and a sticky mention so users can stay inside the participant.  
- Register a slash command `/commit-message-generator` that’s surfaced through `contributes.chatParticipants[].commands[]`; hook it to a VS Code command `lexile.commitSteward.generateCommitMessage`.  
- Activation events: `onCommand:lexile.commitSteward.generateCommitMessage` and `onChatParticipant:lexile.commitSteward` so the extension only loads when the chat participant or command is invoked.  
- Use TypeScript + `@types/vscode` and rely on the Language Model API/Chat API shipped in VS Code (no external runtime). Ship npm scripts for `compile`, `watch`, and `lint`; reuse repo tooling (`npm run lint` can call `eslint --max-warnings=0`). No third-party runtime deps besides `esbuild`/`tslib` if we need bundling.

#### Context + LLM Data Flow (Phase3.B)

- When `/commit-message-generator` (or the VS Code command) is triggered, wrap execution in `vscode.window.withProgress` so users see a cancellable progress UI.  
- Invoke the existing PowerShell script by spawning `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/collect-commit-context.ps1`. Detect Windows-only shells (`powershell.exe`) if `pwsh` is absent. Surface stderr/stdout streaming to the progress notification, and treat non-zero exit codes as actionable errors.  
- After the script finishes, read commit_context.txt (UTF‑8). If the file is missing or empty, show a warning and stop.  
- Load commit-steward.agent.md once and cache it; prepend it as the first `LanguageModelChatMessage.User(...)`. Add a second user message containing the collected commit context + any user freeform prompt (from the chat request or the command’s input box).  
- Select a model with `await vscode.lm.selectChatModels({ vendor: 'copilot', family: 'gpt-4o' })`, honoring the user-selected chat model when invoked from chat (per the Language Model API guide). Handle missing models, consent prompts, and quota errors via `LanguageModelError`.  
- Stream the response: in chat we’ll forward fragments through `ChatResponseStream.markdown`; outside chat we’ll aggregate the final text and show a preview quick-pick so the user can accept or copy.

#### SCM Update & UX (Phase3.C)

- Use the Git extension API (`vscode.extensions.getExtension('vscode.git')?.exports.getAPI(1)`) to obtain repositories; pick the selected repo, or fall back to the first open repo. Then set `repository.inputBox.value = generatedMessage`. This leverages the `InputBox` contract defined in `git.d.ts` and mirrors the Source Control input box semantics described in the Source Control API guide.  
- If no Git repo is available, fall back to copying the message to the clipboard and notify the user.  
- After updating the SCM input box, raise a VS Code notification and append a chat response (if invoked from chat) summarizing what happened, including links to commit_context.txt.  
- Error handling: differentiate script failures, missing context file, LM errors, and SCM update issues; surface actionable notifications and leave the prior message untouched.

### Thinking Beast Mode - Phase 4: Recursive Implementation & Validation (Not Started)

- [ ] Phase 4.A extension scaffolding updates
- [ ] Phase 4.B slash command handler & script bridge
- [ ] Phase 4.C SCM autofill + notifications
- [ ] Phase 4.D validation & documentation


## Acceptance Criteria (early draft)

- [ ] Invoking `/commit-message-generator` (via chat or command palette) runs `scripts/dev-tools/collect-commit-context.ps1`, streams progress, and surfaces any script errors without leaving artifacts in an unknown state.
- [ ] Successful runs cache + reuse `.github/agents/commit-steward.agent.md`, pass commit context + optional user prompt to the Language Model API, and stream the model response back to chat (when invoked there).
- [ ] The generated commit message is automatically inserted into the active Git repository’s Source Control input box using the Git extension API, or copied to the clipboard with a warning when no repo/input box is available.
- [ ] Users receive clear notifications for success, empty context, LM failures, or SCM update issues, and chat responses include actionable summaries with references to the produced `artifacts/commit_context.txt` file.
- [ ] Unit and integration tests cover script invocation plumbing, LM prompt construction, SCM input box updates, and failure fallbacks (missing script, missing file, LM error, SCM unavailable).

## Constraints & Risks

- **Copilot availability & consent**: relies on VS Code’s Language Model API and Copilot entitlement; must degrade gracefully when no approved models are accessible or the user declines consent.
- **PowerShell dependency**: `collect-commit-context.ps1` must run on Windows, macOS, and Linux with `pwsh`; we need shell fallback logic plus clear guidance if PowerShell isn’t installed.
- **Git extension coupling**: automation assumes the built-in Git extension is enabled; failure paths must not corrupt SCM input state or require extra providers.
- **Token/size budgets**: large diffs can exceed LM input limits; the feature needs truncation rules or user warnings to prevent quota waste.
- **Security/privacy**: commit context may include sensitive code; ensure documentation reminds users to review content before sending to Copilot and avoid auto-uploading without opt-in.

## Test Conditions to Consider

- [ ] Unit-test the Node process wrapper to confirm it spawns PowerShell with the right arguments, captures exit codes, and surfaces stderr.
- [ ] Unit-test prompt construction to ensure agent instructions and commit context are merged correctly, including user-provided overrides and truncation.
- [ ] Integration test (VS Code extension test harness) that runs the command in a repo with staged changes, verifies `artifacts/commit_context.txt` is produced, and that SCM input box receives the generated text.
- [ ] Negative test where PowerShell is unavailable or returns non-zero to confirm notifications and chat responses report the failure without invoking the LM.
- [ ] Integration test where Git extension is disabled/no repo open to validate the clipboard fallback and warning path.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/commit-message-process-automation/` folder from the template
