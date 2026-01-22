#!/usr/bin/env bats
# Bats tests for apt helper functions in codex-web-setup.sh
# Requirement coverage: REQ-001 (retries), REQ-002 (timeouts), REQ-003 (pipelining)

# Load the script without executing main() by sourcing it
setup() {
    # Source the script to get function definitions
    source "${BATS_TEST_DIRNAME}/../../.github/codex/codex-web-setup.sh"
}

################################################################################
# apt_with_retries tests - REQ-001: retry logic with exponential backoff
################################################################################

# [P3-T1] Test that apt_with_retries succeeds on first attempt when command succeeds
@test "apt_with_retries succeeds on first attempt when command succeeds" {
    # Create a mock command that succeeds immediately
    mock_success_cmd() {
        echo "success"
        return 0
    }

    # Override apt-specific behavior for test isolation
    export APT_RETRY_ATTEMPTS=3
    export APT_RETRY_DELAY_SECONDS=0

    # Run apt_with_retries with a simple echo (guaranteed to succeed)
    run apt_with_retries echo "test success"

    [ "$status" -eq 0 ]
    [[ "$output" == *"test success"* ]]
}

# [P3-T2] Test that apt_with_retries retries and eventually fails after exhaustion
@test "apt_with_retries exhausts retries and fails with non-zero exit" {
    # Use false command which always fails
    export APT_RETRY_ATTEMPTS=2
    export APT_RETRY_DELAY_SECONDS=0

    run apt_with_retries false

    [ "$status" -ne 0 ]
    [[ "$output" == *"Attempt 1"* ]]
    [[ "$output" == *"Attempt 2"* ]]
    [[ "$output" == *"failed after 2 attempts"* ]]
}

################################################################################
# apt_update tests - REQ-002/REQ-003: timeout and pipelining options
################################################################################

# [P3-T3] Test that apt_update constructs options correctly
@test "apt_update includes timeout and pipelining options when configured" {
    # We can't actually run apt-get, but we can verify option construction
    # by examining what the function would pass. For now, verify the function exists
    # and returns appropriate exit code when apt-get is mocked.

    # Create a mock apt-get that just echoes its arguments
    apt-get() {
        echo "apt-get $*"
        return 0
    }
    export -f apt-get

    export APT_HTTP_TIMEOUT_SECONDS=60
    export APT_DISABLE_PIPELINING=1
    export APT_RETRY_ATTEMPTS=1
    export APT_RETRY_DELAY_SECONDS=0

    run apt_update

    [ "$status" -eq 0 ]
    # Verify timeout option is included
    [[ "$output" == *"-o Acquire::http::Timeout=60"* ]]
    # Verify pipelining disable is included
    [[ "$output" == *"-o Acquire::http::Pipeline-Depth=0"* ]]
}

################################################################################
# apt_install tests - REQ-002: timeout options for install
################################################################################

# [P3-T4] Test that apt_install constructs options correctly
@test "apt_install includes timeout and fix-missing options" {
    # Mock apt-get to capture arguments
    apt-get() {
        echo "apt-get $*"
        return 0
    }
    export -f apt-get

    export APT_HTTP_TIMEOUT_SECONDS=45
    export APT_RETRY_ATTEMPTS=1
    export APT_RETRY_DELAY_SECONDS=0

    run apt_install "package1" "package2"

    [ "$status" -eq 0 ]
    # Verify timeout option is included
    [[ "$output" == *"-o Acquire::http::Timeout=45"* ]]
    # Verify fix-missing is included
    [[ "$output" == *"--fix-missing"* ]]
    # Verify packages are passed
    [[ "$output" == *"package1"* ]]
    [[ "$output" == *"package2"* ]]
}
