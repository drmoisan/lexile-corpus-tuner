"""Unit tests for lexile_corpus_tuner/pipeline_scripts/gutenberg_query_core.py."""

from __future__ import annotations

from lexile_corpus_tuner.pipeline_scripts.gutenberg_query_core import (
    QueryConstraintModel,
    QueryGroupModel,
    SavedQuery,
)


def test_constraint_contains() -> None:
    constraint = QueryConstraintModel("title", "contains", "Fiction")

    assert constraint.to_query_string() == "title:Fiction"


def test_constraint_list_value() -> None:
    constraint = QueryConstraintModel("subjects", "contains", ["Fiction", "Adventure"])

    assert (
        constraint.to_query_string() == '(subjects:"Fiction" OR subjects:"Adventure")'
    )


def test_group_nested_query_string() -> None:
    group = QueryGroupModel(
        logic="AND",
        constraints=[
            QueryConstraintModel("title", "contains", "Fiction"),
            QueryGroupModel(
                logic="OR",
                constraints=[
                    QueryConstraintModel("download_count", ">", "100"),
                    QueryConstraintModel("download_count", "<", "50"),
                ],
            ),
        ],
    )

    assert (
        group.to_query_string()
        == "(title:Fiction AND (download_count>100 OR download_count<50))"
    )


def test_saved_query_round_trip() -> None:
    group = QueryGroupModel(
        logic="AND",
        constraints=[
            QueryConstraintModel("title", "contains", "Fiction"),
            QueryConstraintModel("download_count", ">", "200"),
        ],
    )

    saved = SavedQuery.from_query_group(group)
    loaded_group = SavedQuery.from_json(saved.to_json()).to_query_group()

    assert loaded_group.to_query_string() == group.to_query_string()


