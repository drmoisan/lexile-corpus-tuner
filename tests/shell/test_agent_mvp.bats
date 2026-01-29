#!/usr/bin/env bats
# ------------------------------------------------------------------------------
# test_agent_mvp.bats
#
# Purpose:
#   Unit tests for scripts/bash/agent_mvp.sh.
#
# Notes:
#   - We stub external executables (git/poetry/copilot/date) via PATH.
#   - Tests disable log file creation using AGENT_MVP_LOG_FILE=/dev/null.
# ------------------------------------------------------------------------------

setup() {
    # Get the repository root (tests/shell is two levels down)
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    SCRIPT_UNDER_TEST="${REPO_ROOT}/scripts/bash/agent_mvp.sh"

    FIXTURE_BIN="${REPO_ROOT}/tests/shell/fixtures/agent_mvp/bin"

    export PATH="${FIXTURE_BIN}:${PATH}"

    # Avoid filesystem writes for agent logs.
    export AGENT_MVP_LOG_FILE="/dev/null"

    # Stable identifiers for assertions.
    export DATE_STUB="2000-01-01_000000"

    # Default: clean tree on a non-protected branch.
    export GIT_STATUS_PORCELAIN=""
    export GIT_BRANCH="feature/test"

    # Default: make all QC tools pass.
    export POETRY_EXIT_BLACK="0"
    export POETRY_EXIT_RUFF="0"
    export POETRY_EXIT_PYRIGHT="0"
    export POETRY_EXIT_PYTEST="0"

    export COPILOT_EXIT="0"

    # Keep tests fast and deterministic.
    export MAX_ITERS="1"
}

@test "agent_mvp.sh prints usage and exits 2 when TASK is missing" {
    run bash "${SCRIPT_UNDER_TEST}"

    [ "$status" -eq 2 ]
    [[ "$output" == Usage:* ]]
}

@test "agent_mvp.sh exits 3 when working tree is dirty" {
    export GIT_STATUS_PORCELAIN=" M scripts/bash/agent_mvp.sh"

    run bash "${SCRIPT_UNDER_TEST}" "do a thing"

    [ "$status" -eq 3 ]
    [[ "$output" == *"ERROR: Working tree is not clean."* ]]
}

@test "agent_mvp.sh exits 4 on protected branch" {
    export GIT_BRANCH="main"

    run bash "${SCRIPT_UNDER_TEST}" "do a thing"

    [ "$status" -eq 4 ]
    [[ "$output" == *"Refusing to run on protected branch"* ]]
}

@test "agent_mvp.sh exits 0 when QC passes (single iteration)" {
    export MAX_ITERS="1"

    run bash "${SCRIPT_UNDER_TEST}" "do a thing"

    [ "$status" -eq 0 ]
    [[ "$output" == *"== Host QC gate =="* ]]
    [[ "$output" == *"QC PASSED"* ]]
    [[ "$output" == *"Ready to commit."* ]]
}

@test "agent_mvp.sh supports log-enabled mode without writing files" {
    # Exercise LOG_ENABLED=1 branches without creating directories/files:
    # - LOG_DIR='.' makes mkdir -p a no-op
    # - LOG_FILE='/dev/fd/1' writes via tee to stdout
    export AGENT_MVP_LOG_DIR="."
    export AGENT_MVP_LOG_FILE="/dev/fd/1"
    export MAX_ITERS="1"

    run bash "${SCRIPT_UNDER_TEST}" "do a thing"

    [ "$status" -eq 0 ]
    [[ "$output" == *"Log file: /dev/fd/1"* ]]
    [[ "$output" == *"QC PASSED"* ]]
}

@test "agent_mvp.sh exits 5 when QC fails until max iterations reached" {
    export MAX_ITERS="2"
    export POETRY_EXIT_BLACK="1"

    run bash "${SCRIPT_UNDER_TEST}" "do a thing"

    [ "$status" -eq 5 ]
    [[ "$output" == *"QC FAILED"* ]]
    [[ "$output" == *"Reached max iterations"* ]]
}
