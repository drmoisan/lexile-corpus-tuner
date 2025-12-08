"""Unit tests for lexile_corpus_tuner/pipeline_scripts/gutenberg_query_core.py."""

from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    gutenberg_query_core as gq_core,
)

QueryConstraintModel = gq_core.QueryConstraintModel
QueryGroupModel = gq_core.QueryGroupModel
SavedQuery = gq_core.SavedQuery


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


def test_constraint_variants_render() -> None:
    constraint_eq = QueryConstraintModel("title", "=", "Novel")
    constraint_not = QueryConstraintModel("title", "!=", "Horror")
    constraint_range = QueryConstraintModel("download_count", "range", "10..20")
    constraint_gt = QueryConstraintModel("download_count", ">", "50")

    assert constraint_eq.to_query_string() == 'title="Novel"'
    assert constraint_not.to_query_string() == 'NOT title="Horror"'
    assert constraint_range.to_query_string() == "download_count:10..20"
    assert constraint_gt.to_query_string() == "download_count>50"


def test_group_empty_and_single_child() -> None:
    empty_group = QueryGroupModel(logic="AND", constraints=[])
    assert empty_group.to_query_string() == ""

    single_group = QueryGroupModel(
        logic="OR", constraints=[QueryConstraintModel("id", "=", "1")]
    )
    assert single_group.to_query_string() == 'id="1"'


def test_saved_query_wraps_constraint_into_group() -> None:
    saved_constraint = SavedQuery(
        version="1.0",
        created="now",
        modified="now",
        query={
            "type": "constraint",
            "field": "title",
            "operator": "contains",
            "value": "Example",
        },
    )

    group = saved_constraint.to_query_group()
    assert isinstance(group, QueryGroupModel)
    assert len(group.constraints) == 1
    assert group.to_query_string() == "title:Example"
