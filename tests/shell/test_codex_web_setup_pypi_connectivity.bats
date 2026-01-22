#!/usr/bin/env bats
# Bats tests for check_pypi_connectivity function in codex-web-setup.sh
# Requirement coverage: REQ-004 (network connectivity handling), REQ-005 (offline mode)

# Load the script without executing main() by sourcing it
setup() {
    # Source the script to get function definitions
    source "${BATS_TEST_DIRNAME}/../../.github/codex/codex-web-setup.sh"
}

################################################################################
# check_pypi_connectivity tests - REQ-004/REQ-005: network resilience
################################################################################

# [P7-T1] Test that check_pypi_connectivity skips when ALLOW_OFFLINE_INSTALL=1
@test "check_pypi_connectivity skips when ALLOW_OFFLINE_INSTALL=1" {
    # Stub curl to fail if called (should not be called)
    curl() {
        echo "ERROR: curl should not have been called" >&2
        return 1
    }
    export -f curl

    export ALLOW_OFFLINE_INSTALL=1

    run check_pypi_connectivity

    [ "$status" -eq 0 ]
    [[ "$output" == *"ALLOW_OFFLINE_INSTALL=1 set; skipping PyPI connectivity check"* ]]
}

# [P7-T2] Test that check_pypi_connectivity fails when curl fails and offline not allowed
@test "check_pypi_connectivity fails when curl fails and offline not allowed" {
    # Stub curl to always fail
    curl() {
        return 1
    }
    export -f curl

    # Ensure offline mode is disabled
    unset ALLOW_OFFLINE_INSTALL

    run check_pypi_connectivity

    [ "$status" -ne 0 ]
    [[ "$output" == *"ERROR: Unable to reach pypi.org"* ]]
}

# Additional test: check_pypi_connectivity succeeds when curl succeeds
@test "check_pypi_connectivity succeeds when curl succeeds" {
    # Stub curl to succeed
    curl() {
        return 0
    }
    export -f curl

    # Ensure offline mode is disabled
    unset ALLOW_OFFLINE_INSTALL

    run check_pypi_connectivity

    [ "$status" -eq 0 ]
}
