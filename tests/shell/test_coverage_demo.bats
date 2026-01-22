#!/usr/bin/env bats
# Test suite for coverage_demo.sh - validates the demo script executes correctly.

# Get the repository root (tests/shell is two levels down)
REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
DEMO_SCRIPT="${REPO_ROOT}/scripts/bash/coverage_demo.sh"

@test "coverage_demo.sh runs successfully with default argument" {
    run bash "${DEMO_SCRIPT}"
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, Demo!" ]
}

@test "coverage_demo.sh accepts a custom name argument" {
    run bash "${DEMO_SCRIPT}" "Tester"
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, Tester!" ]
}

@test "coverage_lib.sh greet_user function outputs greeting" {
    # Source the library directly and test the function
    source "${REPO_ROOT}/scripts/bash/coverage_lib.sh"
    run greet_user "World"
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, World!" ]
}
