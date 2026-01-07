#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# lexile-corpus-tuner :: Agent MVP
#
# Deterministic CLI orchestrator:
#   plan is implicit in TASK
#   agent executes
#   host enforces QC
# -----------------------------------------------------------------------------

TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
	echo "Usage: tools/agent_mvp.sh \"<task + acceptance criteria>\""
	exit 2
fi

# ---- Canonical QC commands (derived from README / CI) ------------------------

FMT_CMD="poetry run black --check ."
LINT_CMD="poetry run ruff check"
TYPE_CMD="poetry run pyright"
TEST_CMD="poetry run pytest --cov=src/lexile_corpus_tuner --cov-report=xml --cov-report=term-missing"

MAX_ITERS="${MAX_ITERS:-4}"
LOG_DIR=".agent_logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y-%m-%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/agent_${RUN_ID}.log"

say() { printf "%s\n" "$*" | tee -a "${LOG_FILE}"; }

# ---- Safety checks -----------------------------------------------------------

require_clean_tree() {
	if [[ -n "$(git status --porcelain)" ]]; then
		say "ERROR: Working tree is not clean."
		exit 3
	fi
}

refuse_protected_branch() {
	local b
	b="$(git rev-parse --abbrev-ref HEAD)"
	if [[ "${b}" =~ ^(main|master|development)$ ]]; then
		say "ERROR: Refusing to run on protected branch '${b}'."
		exit 4
	fi
}

run_qc() {
	say "== Black =="
	eval "${FMT_CMD}"
	say "== Ruff =="
	eval "${LINT_CMD}"
	say "== Pyright =="
	eval "${TYPE_CMD}"
	say "== Pytest =="
	eval "${TEST_CMD}"
}

# ---- Preconditions -----------------------------------------------------------

require_clean_tree
refuse_protected_branch

say "Task:"
say "  ${TASK}"
say "Branch: $(git rev-parse --abbrev-ref HEAD)"
say "Max iterations: ${MAX_ITERS}"
say "Log file: ${LOG_FILE}"

# ---- Execution loop ----------------------------------------------------------

for ((i = 1; i <= MAX_ITERS; i++)); do
	say ""
	say "================ Iteration ${i}/${MAX_ITERS} ================"

	copilot -p "
You are executing a change in the lexile-corpus-tuner repository.

Authoritative constraints:
- Follow all policies under .github/instructions/.
- Do NOT replan or expand scope.
- Make the minimum change necessary to satisfy the task.

Task:
${TASK}

Hard requirement:
Run and satisfy the following toolchain in this exact order.
If a step fails, fix the root cause and restart the sequence.

1) ${FMT_CMD}
2) ${LINT_CMD}
3) ${TYPE_CMD}
4) ${TEST_CMD}

Stop only when all steps pass or you are blocked by missing information.
" \
		--allow-tool 'write' \
		--allow-tool 'shell(poetry)' \
		--allow-tool 'shell(python)' \
		--allow-tool 'shell(git)' \
		2>&1 | tee -a "${LOG_FILE}"

	say "== Host QC gate =="
	if run_qc 2>&1 | tee -a "${LOG_FILE}"; then
		say "✅ QC PASSED"
		say "Ready to commit."
		exit 0
	else
		say "❌ QC FAILED — remediation required."
	fi
done

say ""
say "ERROR: Reached max iterations (${MAX_ITERS}) without green QC."
say "Inspect ${LOG_FILE} for details."
exit 5
