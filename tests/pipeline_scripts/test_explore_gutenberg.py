from __future__ import annotations

# ruff: noqa: TC006  # allow explicit typing casts for test doubles
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    explore_gutenberg as eg,
)

if TYPE_CHECKING:
    import pytest

    from .conftest import InMemoryPath


def test_get_canonical_sets_cleans_and_splits() -> None:
    df = pd.DataFrame(
        {
            "subjects": ["Fiction; Drama", " Fiction ; ;None", None],
            "bookshelves": ["Adventure; Classic", "  Poetry  ;nan", ""],
        }
    )

    subjects, shelves = eg.get_canonical_sets(df)

    assert subjects == {"Fiction", "Drama"}
    assert shelves == {"Adventure", "Classic", "Poetry"}


def test_get_canonical_sets_missing_columns() -> None:
    df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})

    subjects, shelves = eg.get_canonical_sets(df)

    assert subjects == set()
    assert shelves == set()


def test_query_history_load_save_roundtrip(root_path: InMemoryPath) -> None:
    history_path = root_path / "history.json"
    history_path.write_text('["old query"]')
    history = eg.QueryHistory(history_file=cast(Path, history_path))

    history.add("new query")
    assert history.get_recent(2) == ["old query", "new query"]
    assert "new query" in history_path.storage[history_path.path]


def _make_engine() -> tuple[eg.BooleanQueryEngine, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Book A", "Book B", "Book C", "Book D"],
            "authors": ["Author X", "Author Y", "Author X", "Author Z"],
            "subjects": [
                "Fiction; Drama",
                "Fiction; Adventure",
                "Poetry",
                "Science; History",
            ],
            "bookshelves": [
                "Classic; Adventure",
                "Adventure",
                "Poetry Collection",
                "Science",
            ],
            "download_count": [100, 200, 50, 300],
        }
    )
    return eg.BooleanQueryEngine(df), df


def test_tokenize_and_parse_field_query() -> None:
    engine, _ = _make_engine()
    query = 'subjects:"Fiction" AND download_count>100'
    tokens = engine._tokenize(query)  # pyright: ignore[reportPrivateUsage]
    assert 'subjects:"Fiction"' in tokens
    field, op, value = engine._parse_field_query(
        "download_count:10..20"
    )  # pyright: ignore[reportPrivateUsage]
    assert (field, op, value) == ("download_count", "..", "10..20")


def test_boolean_query_engine_contains_across_fields() -> None:
    engine, _ = _make_engine()

    result = engine.evaluate("Fiction")

    ids = cast(list[int], result["id"].tolist())
    assert set(ids) == {1, 2}


def test_boolean_query_engine_numeric_and_range() -> None:
    engine, _ = _make_engine()

    gt_result = engine.evaluate("download_count>150")
    gt_ids = cast(
        list[int], gt_result["id"].tolist()
    )  # pyright: ignore[reportUnknownMemberType]
    assert set(gt_ids) == {2, 4}

    range_result = engine.evaluate("download_count:50..150")
    range_ids = cast(
        list[int], range_result["id"].tolist()
    )  # pyright: ignore[reportUnknownMemberType]
    assert set(range_ids) == {1, 3}


def test_boolean_query_engine_exact_match_semicolon_split() -> None:
    engine, _ = _make_engine()

    result = engine.evaluate('subjects:"Fiction"')

    ids = cast(
        list[int], result["id"].tolist()
    )  # pyright: ignore[reportUnknownMemberType]
    assert set(ids) == {1, 2}


def test_boolean_query_engine_invalid_query_returns_empty() -> None:
    engine, _ = _make_engine()

    result = engine.evaluate("(")

    assert result.empty


def test_save_canonical_sets_writes_sorted(root_path: InMemoryPath) -> None:
    subjects = {"Zebra", "Apple"}
    bookshelves = {"Zoo", "Art"}
    eg.save_canonical_sets(subjects, bookshelves, cast(Path, root_path))
    subj_content = root_path.storage.get("root/subjects.txt", "")
    shelf_content = root_path.storage.get("root/bookshelves.txt", "")
    assert subj_content == "Apple\nZebra"
    assert shelf_content == "Art\nZoo"


def test_query_history_handles_missing_file(root_path: InMemoryPath) -> None:
    history_path = root_path / "missing.json"
    history = eg.QueryHistory(history_file=cast(Path, history_path))
    history.add("test query")
    assert history.get_recent() == ["test query"]


def test_pandas_helpers_cover_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    numbers = pd.Series([1, 2])
    string_series = eg._pandas_string_series(
        numbers
    )  # pyright: ignore[reportPrivateUsage]
    assert list(string_series) == ["1", "2"]

    split_values = pd.Series(["a;b", "c"], dtype=object)
    exploded = eg._pandas_split_explode(
        split_values, ";"
    )  # pyright: ignore[reportPrivateUsage]
    assert list(exploded) == ["a", "b", "c"]

    unique_source = pd.Series([" A ", "b", None, "a "])
    unique = eg._pandas_strip_unique(
        unique_source
    )  # pyright: ignore[reportPrivateUsage]
    assert set(unique.tolist()) == {"A", "b", "a"}

    contains_source = pd.Series(["foo", "bar", None])
    contains_mask = eg._pandas_contains(
        contains_source, "foo"
    )  # pyright: ignore[reportPrivateUsage]
    assert contains_mask.tolist() == [True, False, False]

    comparison_source = pd.Series([1, 2, 3])
    comparison = eg._pandas_comparison_mask(
        comparison_source, ">", 1
    )  # pyright: ignore[reportPrivateUsage]
    assert comparison.tolist() == [False, True, True]

    range_mask = eg._pandas_range_mask(
        comparison_source, 1, 2
    )  # pyright: ignore[reportPrivateUsage]
    assert range_mask.tolist() == [True, True, False]

    exact_source = pd.Series(["Foo", "bar"])
    exact_mask = eg._pandas_exact_match(
        exact_source, "foo"
    )  # pyright: ignore[reportPrivateUsage]
    assert exact_mask.tolist() == [True, False]

    filtered_mask = pd.Series([True, False, True])
    filtered_df = eg._pandas_filter_by_mask(
        pd.DataFrame({"a": [1, 2, 3]}), filtered_mask
    )  # pyright: ignore[reportPrivateUsage]
    assert filtered_df["a"].tolist() == [
        1,
        3,
    ]  # pyright: ignore[reportUnknownMemberType]

    sentinel_df = pd.DataFrame({"x": [1]})

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        return sentinel_df

    monkeypatch.setattr(eg.pd, "read_parquet", fake_read_parquet)
    assert (
        eg._pandas_read_parquet(Path("fake")) is sentinel_df
    )  # pyright: ignore[reportPrivateUsage]

    csv_calls: list[tuple[Path, bool]] = []
    parquet_calls: list[tuple[Path, bool]] = []

    def fake_to_csv(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        csv_calls.append((path, index))

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        parquet_calls.append((path, index))

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv, raising=False)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    df = pd.DataFrame({"a": [1, 2]})
    eg._pandas_to_csv(df, Path("out.csv"))  # pyright: ignore[reportPrivateUsage]
    eg._pandas_to_parquet(
        df, Path("out.parquet")
    )  # pyright: ignore[reportPrivateUsage]
    assert csv_calls == [(Path("out.csv"), False)]
    assert parquet_calls == [(Path("out.parquet"), False)]

    table = eg._pandas_to_string(
        df, ["a"], max_rows=1
    )  # pyright: ignore[reportPrivateUsage]
    assert "1" in table


def test_pandas_split_explode_skips_none_and_nan() -> None:
    series = pd.Series(["x;y", None, math.nan], dtype=object)

    exploded = eg._pandas_split_explode(
        series, ";"
    )  # pyright: ignore[reportPrivateUsage]

    assert exploded.tolist() == ["x", "y"]


def test_pandas_comparison_mask_additional_operators() -> None:
    numbers = pd.Series([1, 2, 3])

    lt_mask = eg._pandas_comparison_mask(
        numbers, "<", 3
    )  # pyright: ignore[reportPrivateUsage]
    ge_mask = eg._pandas_comparison_mask(
        numbers, ">=", 2
    )  # pyright: ignore[reportPrivateUsage]
    le_mask = eg._pandas_comparison_mask(
        numbers, "<=", 2
    )  # pyright: ignore[reportPrivateUsage]
    default_mask = eg._pandas_comparison_mask(
        numbers, "?", 0
    )  # pyright: ignore[reportPrivateUsage]

    assert lt_mask.tolist() == [True, True, False]
    assert ge_mask.tolist() == [False, True, True]
    assert le_mask.tolist() == [True, True, False]
    assert default_mask.tolist() == [True, True, True]


def test_pandas_filter_by_mask_handles_none_and_nan() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    mask = pd.Series([True, None, math.nan])

    filtered = eg._pandas_filter_by_mask(
        df, mask
    )  # pyright: ignore[reportPrivateUsage]

    assert filtered["a"].tolist() == [1]


def test_get_canonical_sets_ignores_nan_values() -> None:
    df = pd.DataFrame(
        {"subjects": ["Math;Sci", math.nan], "bookshelves": [math.nan, ""]}
    )

    subjects, shelves = eg.get_canonical_sets(df)

    assert subjects == {"Math", "Sci"}
    assert shelves == set()


def test_apply_field_filter_removes_quotes_without_field() -> None:
    engine, _ = _make_engine()

    mask = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        None, "contains", '"Fiction"'
    )

    assert mask.tolist().count(True) == 2


def test_apply_field_filter_unknown_field_returns_true_mask() -> None:
    engine, _ = _make_engine()

    mask = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "unknown_field", ":", "value"
    )

    assert mask.tolist() == [True, True, True, True]


def test_apply_field_filter_range_and_numeric_errors() -> None:
    engine, _ = _make_engine()

    non_numeric_range = (
        engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
            "subjects", "..", "1..2"
        )
    )
    invalid_range = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "download_count", "..", "low..high"
    )
    numeric_on_text = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "subjects", ">", "10"
    )
    invalid_numeric = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "download_count", ">", "abc"
    )

    assert non_numeric_range.tolist() == [True, True, True, True]
    assert invalid_range.tolist() == [True, True, True, True]
    assert numeric_on_text.tolist() == [True, True, True, True]
    assert invalid_numeric.tolist() == [True, True, True, True]


def test_apply_field_filter_exact_match_handles_nulls() -> None:
    df = pd.DataFrame({"subjects": [None, math.nan, "Fiction;Mystery"]})
    engine = eg.BooleanQueryEngine(df)

    mask = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "subjects", ":", '"Fiction"'
    )

    assert mask.tolist() == [False, False, True]


def test_apply_field_filter_exact_and_contains_on_text_field() -> None:
    engine, _ = _make_engine()

    exact_title = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "title", ":", '"Book A"'
    )
    partial_title = engine._apply_field_filter(  # pyright: ignore[reportPrivateUsage]
        "title", ":", "Book"
    )

    assert exact_title.tolist() == [True, False, False, False]
    assert partial_title.tolist() == [True, True, True, True]


def test_evaluate_empty_query_returns_original_dataframe() -> None:
    engine, df = _make_engine()

    result = engine.evaluate("   ")

    assert result is df


def test_evaluate_parse_error_returns_empty_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _make_engine()

    def fail_tokenize(query: str) -> list[str]:
        raise ValueError("bad query")

    monkeypatch.setattr(engine, "_tokenize", fail_tokenize)

    result = engine.evaluate("query")

    assert result.empty


def test_evaluate_missing_operands(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _make_engine()

    assert engine.evaluate("AND").empty
    assert engine.evaluate("OR").empty
    assert engine.evaluate("NOT").empty

    def bad_parse(_: str) -> tuple[str | None, str, str | None]:
        return None, ":", None

    monkeypatch.setattr(engine, "_parse_field_query", bad_parse)
    assert engine.evaluate("token").empty


def test_evaluate_empty_stack_and_extra_operands() -> None:
    engine, _ = _make_engine()

    assert engine.evaluate(")").empty
    assert engine.evaluate("one two").empty


def test_evaluate_operator_precedence_populates_rpn() -> None:
    engine, _ = _make_engine()

    result = engine.evaluate("Fiction AND Poetry OR Science")
    ids = cast(
        list[int], result["id"].tolist()
    )  # pyright: ignore[reportUnknownMemberType]

    assert set(ids) == {4}


def test_evaluate_complex_boolean_expression() -> None:
    engine, _ = _make_engine()

    result = engine.evaluate("(Fiction OR Poetry) AND NOT Science")
    ids = cast(
        list[int], result["id"].tolist()
    )  # pyright: ignore[reportUnknownMemberType]

    assert set(ids) == {1, 2, 3}


def test_query_history_load_handles_invalid_json(root_path: InMemoryPath) -> None:
    history_path = root_path / "history.json"
    history_path.write_text("{bad json}")

    history = eg.QueryHistory(history_file=cast(Path, history_path))

    assert history.get_recent() == []


def test_query_history_save_handles_io_error(
    monkeypatch: pytest.MonkeyPatch, root_path: InMemoryPath
) -> None:
    history_path = root_path / "history.json"
    history = eg.QueryHistory(history_file=cast(Path, history_path))

    def fail_open(*args: object, **kwargs: object) -> object:
        raise OSError("fail to write")

    monkeypatch.setattr(history.history_file, "open", fail_open, raising=False)

    history.add("query")
    assert history.get_recent() == ["query"]
