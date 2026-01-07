#!/usr/bin/env python3
"""
Atomic task-by-task executor for lexile-corpus-tuner.

COMPATIBILITY SHIM: This file delegates to scripts.dev_tools.atomic_executor
package for backward compatibility. All logic has been refactored into modular
classes under atomic_executor/ to comply with repo policy (500-line limit).

For implementation details, see:
    - scripts/dev_tools/atomic_executor/cli.py (main entry point)
    - scripts/dev_tools/atomic_executor/plan_parser.py
    - scripts/dev_tools/atomic_executor/feature_resolver.py
    - scripts/dev_tools/atomic_executor/qc_runner.py
    - scripts/dev_tools/atomic_executor/prompt_builder.py
"""

import sys

from scripts.dev_tools.atomic_executor import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
