from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, cast

from sklearn.linear_model import ElasticNet


def save_model(
    model: ElasticNet, metrics: dict[str, float], feature_names: list[str], path: Path
) -> None:
    """Persist a trained ElasticNet model as a JSON spec."""
    model_any = cast(Any, model)
    spec: dict[str, Any] = {
        "version": date.today().isoformat(),
        "features": feature_names,
        "coefficients": [float(value) for value in model_any.coef_],
        "intercept": float(model_any.intercept_),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def load_model_spec(path: Path) -> dict[str, Any]:
    """Load a serialized model spec from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
