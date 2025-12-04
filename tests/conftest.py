"""Pytest configuration and shared fixtures for test suite.

This file contains shared fixtures and configuration that applies across all tests.
"""

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def restore_pandas_module() -> Iterator[None]:
    """Ensure pandas module is restored after tests that mock it globally.

    Some tests (like test_query_builder_ui.py) mock pandas in sys.modules
    to avoid import errors when tkinter isn't available. This fixture ensures
    the real pandas module is restored so other tests can use it normally.

    Yields:
        None
    """
    # Save the original pandas module if it exists
    original_pandas = sys.modules.get("pandas", None)

    yield

    # After all tests, restore the original pandas module
    if original_pandas is not None:
        sys.modules["pandas"] = original_pandas
    elif "pandas" in sys.modules:
        # If pandas was mocked but wasn't originally imported, remove the mock
        del sys.modules["pandas"]
