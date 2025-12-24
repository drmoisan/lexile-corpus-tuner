"""Core query models and serialization for the Gutenberg query builder UI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class QueryConstraintModel:
    """Model for a single query constraint (field:operator:value)."""

    field: str
    operator: str
    value: str | list[str]

    def to_query_string(self) -> str:
        """Convert the constraint to a BooleanQueryEngine-compatible string."""
        if isinstance(self.value, list):
            terms = [f'{self.field}:"{v}"' for v in self.value]
            return f"({' OR '.join(terms)})"

        if self.operator == "contains":
            # Quote value if it contains spaces or special characters
            if " " in self.value or ":" in self.value or '"' in self.value:
                # Escape any quotes in the value
                escaped_value = self.value.replace('"', '\\"')
                return f'{self.field}:"{escaped_value}"'
            return f"{self.field}:{self.value}"
        if self.operator == "=":
            return f'{self.field}="{self.value}"'
        if self.operator == "!=":
            return f'NOT {self.field}="{self.value}"'
        if self.operator == "range":
            return f"{self.field}:{self.value}"

        # Comparison operators: >, <, >=, <=
        return f"{self.field}{self.operator}{self.value}"


def _empty_constraints() -> list[QueryConstraintModel | QueryGroupModel]:
    """Typed default factory for query group constraints."""
    return []


@dataclass
class QueryGroupModel:
    """Model for a group of constraints with AND/OR logic."""

    logic: str  # 'AND' or 'OR'
    constraints: list[QueryConstraintModel | QueryGroupModel] = field(
        default_factory=_empty_constraints
    )

    def to_query_string(self) -> str:
        """Convert the group (and any nested children) to a query string."""
        if not self.constraints:
            return ""

        parts: list[str] = [c.to_query_string() for c in self.constraints]
        parts = [p for p in parts if p]

        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        joined = f" {self.logic} ".join(parts)
        return f"({joined})"


@dataclass
class SavedQuery:
    """Persistent query format for save/load operations."""

    version: str
    created: str
    modified: str
    query: dict[str, Any]  # Serialized QueryGroupModel

    @classmethod
    def from_query_group(cls, group: QueryGroupModel) -> SavedQuery:
        """Create a SavedQuery from a QueryGroupModel."""
        now = datetime.now().isoformat()
        return cls(
            version="1.0",
            created=now,
            modified=now,
            query=cls._serialize_group(group),
        )

    @staticmethod
    def _serialize_group(
        group: QueryGroupModel | QueryConstraintModel,
    ) -> dict[str, Any]:
        """Recursively serialize query structure."""
        if isinstance(group, QueryConstraintModel):
            return {
                "type": "constraint",
                "field": group.field,
                "operator": group.operator,
                "value": group.value,
            }

        constraints_serialized: list[dict[str, Any]] = [
            SavedQuery._serialize_group(c) for c in group.constraints
        ]
        return {
            "type": "group",
            "logic": group.logic,
            "constraints": constraints_serialized,
        }

    @classmethod
    def from_json(cls, json_str: str) -> SavedQuery:
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_query_group(self) -> QueryGroupModel:
        """Deserialize query structure to a QueryGroupModel."""
        result = self._deserialize_group(self.query)
        if not isinstance(result, QueryGroupModel):
            result = QueryGroupModel(logic="AND", constraints=[result])
        return result

    @staticmethod
    def _deserialize_group(
        data: dict[str, Any],
    ) -> QueryGroupModel | QueryConstraintModel:
        """Recursively deserialize query structure."""
        if data["type"] == "constraint":
            return QueryConstraintModel(
                field=data["field"],
                operator=data["operator"],
                value=data["value"],
            )
        return QueryGroupModel(
            logic=data["logic"],
            constraints=[SavedQuery._deserialize_group(c) for c in data["constraints"]],
        )


__all__ = ["QueryConstraintModel", "QueryGroupModel", "SavedQuery"]
