#!/bin/bash
# Coverage demo entry point - sources the library and invokes the covered function.
# Used by Bats tests to demonstrate non-zero coverage.

set -euo pipefail

# Resolve script directory for relative sourcing
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the library
# shellcheck source=coverage_lib.sh disable=SC1091
source "${SCRIPT_DIR}/coverage_lib.sh"

# Invoke the covered function
greet_user "${1:-Demo}"
