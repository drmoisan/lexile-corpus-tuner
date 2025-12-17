from __future__ import annotations

import importlib
import runpy
from typing import Any
from unittest.mock import MagicMock


def test_package_main_invokes_app(ui_modules: Any, monkeypatch: Any) -> None:
    module = importlib.import_module(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui"
    )
    app_cls = MagicMock()
    monkeypatch.setattr(module, "QueryBuilderApp", app_cls)
    monkeypatch.setattr(module, "tk", ui_modules.tk)

    module.main()
    app_cls.assert_called_once()


def test_dunder_main_executes_main(ui_modules: Any, monkeypatch: Any) -> None:
    module_path = (
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts."
        "gutenberg_query_builder_ui.__main__"
    )
    mocked_main = MagicMock()
    monkeypatch.setattr(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.main",
        mocked_main,
    )
    runpy.run_module(module_path, run_name="__main__")
    mocked_main.assert_called_once()
