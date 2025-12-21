# <bug-name> (Plan)

- Issue: <#id or TBD>
- Owner: <name>
- Date: YYYY-MM-DD
- Status: Draft

**Phase 0 — Context & Inputs**
- [ ] [P0-T1] Link approved spec: <spec link>
- [ ] [P0-T2] Record branch/commit baseline: <branch/commit>
- [ ] [P0-T3] List required environment/fixtures/data: <notes>

**Phase 1 — Preparation**
- [ ] [P1-T1] Confirm scope is locked for this fix (no open spec gaps)
- [ ] [P1-T2] Sync workspace to target branch and ensure tooling is available

**Phase 2 — Regression Test (must fail first)**
- [ ] [P2-T1] Add a small, deterministic regression test in the standard module file (use `tests/bugs/<YYYY>/<issue>-<desc>.py` only if no clear home exists)
- [ ] [P2-T2] Run the regression to confirm it fails and captures the repro

**Phase 3 — Minimal Fix**
- [ ] [P3-T1] Apply the smallest change needed to make the regression test pass; avoid opportunistic refactors

**Phase 4 — Verification Loop**
- [ ] [P4-T1] Re-run repro and regression test to confirm expected behavior
- [ ] [P4-T2] Run formatter → linter → type checker → tests; restart loop if any step changes files or fails

**Phase 5 — Documentation & Status**
- [ ] [P5-T1] Update spec/issue with outcomes, decisions, and any deviations from scope

**Phase 6 — PR & Handoff**
- [ ] [P6-T1] Prepare PR notes (summary, risks, validation performed, links to tests) and request review

**Phase 7 — Rollout / Follow-up**
- [ ] [P7-T1] Capture deployment/rollout notes and post-fix monitoring items
- [ ] [P7-T2] Record links (issue, PRs, related docs) for traceability
