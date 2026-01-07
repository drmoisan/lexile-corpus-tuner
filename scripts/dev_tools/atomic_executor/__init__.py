"""
Atomic task-by-task executor for lexile-corpus-tuner.

This package provides infrastructure for executing feature plans one task at a time,
with automatic QC gates and plan.md checkbox management.

Public API:
    PlanParser: Parse and manipulate atomic execution plans
    FeatureResolver: Resolve feature folders from args/branch
    QCRunner: Execute scoped and full QC toolchains
    PromptBuilder: Build prompts from templates + context
    main: CLI entry point
"""

from scripts.dev_tools.atomic_executor.cli import main
from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver
from scripts.dev_tools.atomic_executor.plan_parser import PlanParser
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_runner import QCRunner

__all__ = [
    "PlanParser",
    "FeatureResolver",
    "QCRunner",
    "PromptBuilder",
    "main",
]
