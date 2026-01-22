#!/usr/bin/env bats
# ------------------------------------------------------------------------------
# test_codex_web_setup_source_safety.bats
#
# Purpose:
#   Verify that codex-web-setup.sh is safe to source without executing
#   side effects, enabling deterministic Bats unit tests.
#
# Requirement: REQ-006 from plan
# ------------------------------------------------------------------------------

# Path to the script under test (relative to repo root)
SCRIPT_UNDER_TEST=".github/codex/codex-web-setup.sh"

@test "codex-web-setup.sh contains main() guard for source safety" {
    # The script must contain the standard bash source-guard pattern
    # that allows sourcing without executing imperative code.
    #
    # Expected guard line (exact match):
    #   if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi

    run grep -n 'if \[\[ "${BASH_SOURCE\[0\]}" == "$0" \]\]; then main "$@"; fi' "$SCRIPT_UNDER_TEST"

    # Assert: grep should find the guard line (exit code 0)
    [ "$status" -eq 0 ]

    # Assert: output should contain a line number match
    [[ "$output" =~ ^[0-9]+: ]]
}
