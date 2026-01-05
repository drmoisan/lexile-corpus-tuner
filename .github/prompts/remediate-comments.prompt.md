---
agent: 'comment_remediator'
description: 'Remediate a specified Python scope so docstrings and intent comments comply with self-explanatory-code-commenting rules, following all repo policies.'
tools: ['search/listDirectory', 'search/fileSearch', 'search/codebase', 'search/usages', 'search/changes', 'read/readFile', 'edit/createFile', 'edit/createDirectory', 'edit/editFiles', 'execute/runTask', 'execute/createAndRunTask', 'execute/runInTerminal', 'execute/getTerminalOutput', 'read/problems', 'execute/testFailure', 'todo']
---

# Comment Remediation Loader

Use this prompt to launch `comment_remediator` on any scope (single file, folder, or glob) and enforce `.github/instructions/self-explanatory-code-commenting.instructions.md` while obeying all repo policies.

## Inputs

- **Scope** (required): Path or glob to remediate (e.g., `src/lexile_corpus_tuner/.../enrich_original_pub_year/**`).

## Policy Order

Apply in this sequence:
1. `.github/copilot-instructions.md`
2. `.github/instructions/general-code-change.instructions.md`
3. `.github/instructions/general-unit-test.instructions.md`
4. `.github/instructions/python-code-change.instructions.md`
5. `.github/instructions/python-unit-test.instructions.md`
6. `.github/instructions/self-explanatory-code-commenting.instructions.md`

## Execution Rules

- Remediate the entire scope to add/adjust robust docstrings and intent comments for loops, branches, and multi-step blocks; avoid low-value narration.
- Keep modules cohesive, under 500 lines, strongly typed; avoid `Any` unless justified.
- Maintain encoding/EOL and ASCII unless the file already uses Unicode.
- Do not pause for approval or clarification once scope is set; continue across turns if needed until fully compliant.

## Workflow

1. **Context pass**: Identify files in scope and gaps; capture tasks with `manage_todo_list`.
2. **Remediate**: Add or refine docstrings and intent comments; refactor only when needed for clarity.
3. **Validate (loop until clean)**:
   - `poetry run black .`
   - `poetry run ruff check`
   - `poetry run pyright`
   - `poetry run pytest`
   Restart from Black after any change or failure; resolve all errors before finishing.

## Completion Criteria

- Scope complies with self-explanatory commenting guidance and repo policies.
- Final toolchain pass is green.
- Provide a concise summary and list of commands run.

## Launch Template

```
Scope: <INSERT SCOPE PATH OR GLOB>
```
