#!/bin/bash
# Coverage demo library - provides functions for demonstrating kcov coverage.
# This file is intentionally simple to serve as a minimal coverage target.

# Function that is called by tests (covered)
greet_user() {
	local name="${1:-World}"
	echo "Hello, ${name}!"
}

# Function that is intentionally NOT called (uncovered for demo purposes)
# shellcheck disable=SC2317  # Intentionally unreachable for coverage demo
unused_function() {
	echo "This function is intentionally unused to demonstrate uncovered lines."
	return 1
}
