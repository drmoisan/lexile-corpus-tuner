---
applyTo: ".github/workflows/**/*.yml,.github/workflows/**/*.yaml"
---

# GitHub Actions workflow policy

- Treat `.github/workflows/*.yml` files as **CI-critical**:
  - Do not change the overall job structure unless explicitly requested.
  - Preserve existing `on:` triggers, branch filters, and permissions unless change is intentional and documented.

- **Schema & linting**
  - All workflows **must pass `actionlint`**.
  - Before finalizing changes, ensure the YAML is valid with:
    - Local: `scripts/dev-tools/run-actionlint.ps1`
    - CI: job `actionlint` in `.github/workflows/ci.yml`
  - Avoid constructs that are not supported by `actionlint` or GitHub Actions, such as:
    - Misplaced or misspelled keys (e.g. `matrix` at job level instead of under `strategy:`).
    - Unknown named-values or expressions.

- **Best practices**
  - Keep jobs small and focused (quality checks, build, test, deploy).
  - Use the GitHub Actions expression syntax accurately: `\${{ ... }}`.
  - Prefer reusable actions over inlined complex bash scripts when practical.
