#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TOOLS_DIR="${SCRIPT_DIR}/../tools"
CLOC_EXE="${TOOLS_DIR}/cloc.exe"
CLOC_SCRIPT="${TOOLS_DIR}/cloc"
ARGS=(--vcs=git --quiet --exclude-dir=tools "$ROOT")

if [[ "$OS" == "Windows_NT" && -f "$CLOC_EXE" ]]; then
  "$CLOC_EXE" "${ARGS[@]}"
elif [[ -f "$CLOC_SCRIPT" ]]; then
  if command -v perl >/dev/null 2>&1; then
    perl "$CLOC_SCRIPT" "${ARGS[@]}"
  else
    echo "Perl is required to run the bundled cloc script." >&2
    exit 1
  fi
else
  echo "Bundled cloc binary not found." >&2
  exit 1
fi
