#!/usr/bin/env bash
set -euo pipefail

echo "=== [maintenance] Verifying environment ==="
python --version
poetry --version || true
ls -la
echo "=== [maintenance] Done ==="