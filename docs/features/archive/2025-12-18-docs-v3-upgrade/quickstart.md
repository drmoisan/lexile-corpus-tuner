# Docs v3 Contributor Quickstart

Use this checklist when preparing a PR:

1) Run `poetry run python -m scripts.dev_tools.pr_context.collector --out artifacts/pr_context.txt --appendix-out artifacts/pr_context.appendix.txt`.
2) Fill **PR Intent** in the summary: Primary outcome, User/dev impact, Risks, Author-asserted autoclose issues.
3) Verify **Additional context files** list includes only files you intend to cite; do not reference other files.
4) Confirm **Feature doc excerpts** present the expected Story Statement, Problem/Why, Context, Root Cause, Proposed Fix, Acceptance Criteria, and plan verification notes.
5) If offline/unauthed (GitHub CLI unavailable), ensure GitHub Auto-close remains `None`; do not add `Closes` lines manually.
6) Use the updated PR prompt/agent to generate the body; do not invent issues or PRs beyond those in pr_context.
7) Re-run quality checks: `poetry run black . && poetry run ruff check && poetry run pyright && poetry run pytest`.
