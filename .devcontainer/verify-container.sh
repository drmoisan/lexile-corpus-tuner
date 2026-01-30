#!/bin/bash
# verify-container.sh - Verify dev container configuration and environment

set -euo pipefail

echo "========================================="
echo "Dev Container Environment Verification"
echo "========================================="
echo ""

# Detect environment
if [ "${CODESPACES:-}" = "true" ]; then
    echo "🌐 Environment: GitHub Codespaces"
    ENV_TYPE="codespaces"
else
    echo "🐳 Environment: Local Docker"
    ENV_TYPE="local"
fi
echo ""

# OS Check
echo "📦 Operating System:"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "   Name: $PRETTY_NAME"
    echo "   ID: $ID"
    echo "   Version: $VERSION_ID"
    
    if [ "$ID" = "debian" ] && [ "$VERSION_ID" = "12" ]; then
        echo "   ✅ Debian Bookworm detected (expected)"
        OS_OK=true
    else
        echo "   ⚠️  Unexpected OS (expected Debian 12 Bookworm)"
        OS_OK=false
    fi
else
    echo "   ❌ Cannot detect OS"
    OS_OK=false
fi
echo ""

# Python Check
echo "🐍 Python:"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "   Version: $PYTHON_VERSION"
    
    if [[ "$PYTHON_VERSION" == 3.13.* ]]; then
        echo "   ✅ Python 3.13 detected (expected)"
        PYTHON_OK=true
    else
        echo "   ⚠️  Unexpected version (expected 3.13.x)"
        PYTHON_OK=false
    fi
else
    echo "   ❌ Python not found"
    PYTHON_OK=false
fi
echo ""

# Poetry Check
echo "📝 Poetry:"
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version 2>&1 | awk '{print $3}' | tr -d ')')
    echo "   Version: $POETRY_VERSION"
    
    if [[ "$POETRY_VERSION" == 2.2.* ]]; then
        echo "   ✅ Poetry 2.2.x detected (expected)"
        POETRY_OK=true
    else
        echo "   ⚠️  Unexpected version (expected 2.2.1)"
        POETRY_OK=false
    fi
else
    echo "   ❌ Poetry not found"
    POETRY_OK=false
fi
echo ""

# PowerShell Check
echo "⚡ PowerShell:"
if command -v pwsh &> /dev/null; then
    PWSH_VERSION=$(pwsh --version 2>&1 | head -n1)
    echo "   $PWSH_VERSION"
    
    if [[ "$PWSH_VERSION" == *"7."* ]]; then
        echo "   ✅ PowerShell 7+ detected (expected)"
        PWSH_OK=true
    else
        echo "   ⚠️  Unexpected version (expected 7.5+)"
        PWSH_OK=false
    fi
else
    echo "   ❌ PowerShell not found"
    PWSH_OK=false
fi
echo ""

# Shell Tools Check
echo "🔧 Shell Tools:"
declare -a TOOLS=("git" "gh" "shellcheck" "shfmt" "bashdb" "actionlint")
TOOLS_OK=true

for tool in "${TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        VERSION=$($tool --version 2>&1 | head -n1 || echo "installed")
        echo "   ✅ $tool: $VERSION"
    else
        echo "   ❌ $tool: not found"
        TOOLS_OK=false
    fi
done
echo ""

# Configuration File Check
echo "📄 Configuration Files:"
if [ -f "/workspaces/lexile-corpus-tuner/.devcontainer/devcontainer.json" ]; then
    echo "   ✅ .devcontainer/devcontainer.json (Codespaces config)"
fi
if [ -f "/workspaces/lexile-corpus-tuner/.devcontainer/local/devcontainer.json" ]; then
    echo "   ✅ .devcontainer/local/devcontainer.json (Local config)"
fi
echo ""

# Virtual Environment Check
echo "🏗️  Python Virtual Environment:"
if [ -d "/workspaces/lexile-corpus-tuner/.venv" ]; then
    echo "   ✅ .venv directory exists"
    if [ -f "/workspaces/lexile-corpus-tuner/.venv/bin/python" ]; then
        VENV_PYTHON=$(/workspaces/lexile-corpus-tuner/.venv/bin/python --version)
        echo "   ✅ Python in venv: $VENV_PYTHON"
        VENV_OK=true
    else
        echo "   ⚠️  Python not found in .venv"
        VENV_OK=false
    fi
else
    echo "   ⚠️  .venv directory not found"
    VENV_OK=false
fi
echo ""

# Summary
echo "========================================="
echo "Summary"
echo "========================================="
if [ "${OS_OK:-false}" = true ] && \
   [ "${PYTHON_OK:-false}" = true ] && \
   [ "${POETRY_OK:-false}" = true ] && \
   [ "${PWSH_OK:-false}" = true ] && \
   [ "${TOOLS_OK:-false}" = true ] && \
   [ "${VENV_OK:-false}" = true ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Your dev container is correctly configured."
    if [ "$ENV_TYPE" = "codespaces" ]; then
        echo "Using: .devcontainer/devcontainer.json (Codespaces)"
    else
        echo "Using: .devcontainer/local/devcontainer.json (Local)"
    fi
    exit 0
else
    echo "⚠️  Some checks failed!"
    echo ""
    echo "Your environment may not match the expected configuration."
    if [ "$ENV_TYPE" = "codespaces" ]; then
        echo ""
        echo "For Codespaces:"
        echo "  - Try deleting and recreating the Codespace"
        echo "  - Ensure .devcontainer/devcontainer.json exists"
    else
        echo ""
        echo "For Local Docker:"
        echo "  - Try rebuilding the container: F1 → Dev Containers: Rebuild Container"
        echo "  - Ensure you selected .devcontainer/local/devcontainer.json"
        echo "  - See .devcontainer/CONFIG-GUIDE.md for help"
    fi
    echo ""
    echo "See .devcontainer/TROUBLESHOOTING.md for more help."
    exit 1
fi
