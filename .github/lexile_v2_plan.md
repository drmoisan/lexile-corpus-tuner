Here’s a “drop this into Copilot/Codex” spec that tells it exactly how to wire **lexile-determination-v2** into the repo it already built from the previous instructions.

I’ll assume the repo structure from before:

* Package: `lexile_corpus_tuner`
* Estimator interface: `LexileEstimator` in `src/lexile_corpus_tuner/estimators/base.py`
* Estimator factory: `create_estimator` in `src/lexile_corpus_tuner/estimators/__init__.py`
* Config: `LexileTunerConfig` in `src/lexile_corpus_tuner/config.py`
* Pipeline: `pipeline.py` is already implemented and uses `LexileEstimator`

You can paste this entire thing into Copilot Chat and say “implement these changes in this repo.”

---

## 0. High-level goal

Add a **real Lexile-based estimator** that wraps the external **lexile-determination-v2** Keras model.

* New class: `LexileDeterminationV2Estimator` implementing `LexileEstimator`.
* It should:

  * Load a trained Keras model and vectorizer (and optional label encoder).
  * Predict a **class label** for input text.
  * Convert that label to a **numeric Lexile value** compatible with the pipeline (float).
* Make it selectable via the existing configuration and CLI:

  * `config.estimator_name = "lexile_v2"`.

The code should work even if the user hasn’t downloaded the actual model yet by providing clear TODOs / error messages and by keeping tests independent of the real model files.

---

## 1. Add optional dependencies (pyproject)

**File:** `pyproject.toml`

1. In the `[project]` / dependencies section, add **optional** dependencies for the adapter:

   ```toml
   [project.optional-dependencies]
   lexile-v2 = [
     "tensorflow>=2.10.0",   # or keras>=2.10, adjust as needed
     "joblib>=1.1.0",
   ]
   ```

2. Do **not** make these mandatory for the whole package:

   * They should be optional, so users can install the core package without TensorFlow.
   * Document in README that to use `lexile_v2` estimator they must install extras:
     `pip install .[lexile-v2]`.

---

## 2. Extend the configuration object

**File:** `src/lexile_corpus_tuner/config.py`

1. In `LexileTunerConfig`, add fields for lexile-v2 paths and band mapping:

   ```python
   from dataclasses import dataclass, field
   from typing import Dict, Optional

   @dataclass
   class LexileTunerConfig:
       # existing fields...
       window_size: int = 500
       stride: int = 250
       max_window_lexile: float = 450.0
       target_avg_lexile: float = 350.0
       avg_tolerance: float = 20.0
       max_passes: int = 3
       smoothing_kernel_size: int = 3
       estimator_name: str = "dummy"  # now accepts "dummy" or "lexile_v2"
       rewrite_enabled: bool = False
       rewrite_model: Optional[str] = None

       # NEW: lexile-determination-v2 related config
       lexile_v2_model_path: Optional[str] = None
       lexile_v2_vectorizer_path: Optional[str] = None
       lexile_v2_label_encoder_path: Optional[str] = None

       # Optional mapping from band labels to numeric midpoints
       lexile_v2_band_to_midpoint: Dict[str, float] = field(default_factory=dict)
   ```

2. In any config loader (e.g., YAML loader), ensure new fields are read from the config file if present.

3. In README/examples config, add an example snippet showing these fields (we’ll specify that later).

---

## 3. Implement the LexileDeterminationV2Estimator adapter

**File:** `src/lexile_corpus_tuner/estimators/lexile_determination_v2_adapter.py` (new file)

Create a new estimator class that implements `LexileEstimator`:

```python
from __future__ import annotations

from typing import Dict, Optional
import re

import numpy as np

try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:  # pragma: no cover
    load_model = None  # will guard at runtime

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # will guard at runtime

from .base import LexileEstimator
```

Then define the class:

```python
class LexileDeterminationV2Estimator(LexileEstimator):
    """
    Adapter around a Keras classifier from the external 'lexile-determination-v2' project.

    Responsibilities:
    - Load a trained Keras model and accompanying vectorizer (and optional label encoder).
    - For each input text, apply the same vectorization as used during training.
    - Run the model to obtain a probability distribution over Lexile classes.
    - Pick the most likely class, obtain its label, and convert that label to a numeric Lexile score.

    This class does NOT know how to train the model; it only loads and uses
    artifacts that were trained in the external project.
    """

    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        label_encoder_path: Optional[str] = None,
        band_to_midpoint: Optional[Dict[str, float]] = None,
    ) -> None:
        if load_model is None or joblib is None:
            raise ImportError(
                "TensorFlow/Keras and joblib are required for LexileDeterminationV2Estimator. "
                "Install the 'lexile-v2' extra, e.g. `pip install .[lexile-v2]`."
            )

        if not model_path or not vectorizer_path:
            raise ValueError("model_path and vectorizer_path must be provided for LexileDeterminationV2Estimator.")

        # 1) Load trained Keras model
        # TODO: Align this call with how the lexile-determination-v2 project saves its model
        self.model = load_model(model_path)

        # 2) Load vectorizer (e.g. TF-IDF or similar)
        self.vectorizer = joblib.load(vectorizer_path)

        # 3) Load label encoder, if provided; otherwise expect a hardcoded mapping
        self.label_encoder = joblib.load(label_encoder_path) if label_encoder_path else None

        # Optional: band labels like "900-999" -> numeric midpoint
        self._band_to_midpoint = band_to_midpoint or {}

        # Optional: fallback mapping from class indices to labels when no label encoder is provided.
        # TODO: If lexile-determination-v2 exposes a class -> label mapping, bake it in here.
        self._index_to_band: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Public API used by the rest of the pipeline
    # ------------------------------------------------------------------ #

    def predict_scalar(self, text: str) -> float:
        """
        Predict a numeric Lexile-like difficulty value for the given text.

        Steps:
        1. Transform the raw text using the loaded vectorizer.
        2. Run the Keras model to obtain class probabilities.
        3. Pick the argmax class index.
        4. Map the class index to a human-readable label.
        5. Convert the label into a numeric Lexile (float).
        """
        # Ensure vectorizer and model are available
        X = self.vectorizer.transform([text])
        probs = self.model.predict(X, verbose=0)[0]
        class_idx = int(np.argmax(probs))

        label = self._index_to_label(class_idx)
        lexile_value = self._label_to_numeric_lexile(label)
        return float(lexile_value)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _index_to_label(self, idx: int) -> str:
        """
        Map a class index (int) to a label string such as:
        - '300-399'
        - '900-999'
        - '950L'
        """
        if self.label_encoder is not None:
            # scikit-learn LabelEncoder returns numpy array; cast to str
            return str(self.label_encoder.inverse_transform([idx])[0])

        if idx in self._index_to_band:
            return self._index_to_band[idx]

        raise RuntimeError(
            "No label_encoder provided and no _index_to_band mapping is defined. "
            "Populate one of these based on lexile-determination-v2's training labels."
        )

    def _label_to_numeric_lexile(self, label: str) -> float:
        """
        Convert a Lexile label into a numeric value.

        Supported label formats (examples):
        - '900-999'
        - '600 - 699L'
        - '950L'
        - '850'
        """
        label = label.strip()

        # 1) Explicit mapping (if provided by config)
        if label in self._band_to_midpoint:
            return self._band_to_midpoint[label]

        # 2) Range like '900-999' or '900 - 999L'
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", label)
        if m:
            low = int(m.group(1))
            high = int(m.group(2))
            return (low + high) / 2.0

        # 3) Single number like '950L' or '950'
        m = re.search(r"(\d+)", label)
        if m:
            return float(m.group(1))

        # 4) Fallback: fail loudly so configuration issues are obvious
        raise ValueError(f"Cannot parse Lexile label: {label!r}")
```

Notes for Codex:

* Leave **TODO comments** where external repo-specific details are required:

  * How exactly the model / vectorizer / labels are saved in `lexile-determination-v2`.
  * How to populate `_index_to_band` if `label_encoder` is not used.
* The goal is for the adapter to be **compilable** with placeholder mappings, and to give clear runtime errors if misconfigured.

---

## 4. Wire the estimator into the factory

**File:** `src/lexile_corpus_tuner/estimators/__init__.py`

1. Import the new adapter:

   ```python
   from .lexile_determination_v2_adapter import LexileDeterminationV2Estimator  # new
   ```

2. Modify `create_estimator` to support `"lexile_v2"`:

   ```python
   from .base import LexileEstimator
   from .dummy_estimator import DummyLexileEstimator
   from .lexile_determination_v2_adapter import LexileDeterminationV2Estimator


   def create_estimator(name: str, **kwargs) -> LexileEstimator:
       name = name.lower()

       if name == "dummy":
           return DummyLexileEstimator()

       if name in ("lexile_v2", "lexile-determination-v2"):
           return LexileDeterminationV2Estimator(**kwargs)

       raise ValueError(f"Unknown estimator name: {name!r}")
   ```

---

## 5. Connect config to estimator creation

Wherever the pipeline currently constructs an estimator from `LexileTunerConfig`, ensure it passes the lexile-v2-specific paths.

**File:** likely `src/lexile_corpus_tuner/pipeline.py` or a helper.

Add a helper function:

```python
from .config import LexileTunerConfig
from .estimators import create_estimator, LexileEstimator


def build_estimator_from_config(config: LexileTunerConfig) -> LexileEstimator:
    if config.estimator_name.lower() in ("lexile_v2", "lexile-determination-v2"):
        return create_estimator(
            config.estimator_name,
            model_path=config.lexile_v2_model_path or "",
            vectorizer_path=config.lexile_v2_vectorizer_path or "",
            label_encoder_path=config.lexile_v2_label_encoder_path,
            band_to_midpoint=config.lexile_v2_band_to_midpoint,
        )

    # Fall back to generic creation (e.g. dummy)
    return create_estimator(config.estimator_name)
```

Then in the main pipeline entry points where estimators were previously created directly, replace with:

```python
estimator = build_estimator_from_config(config)
```

---

## 6. Update the CLI to expose lexile_v2 configuration

**File:** `src/lexile_corpus_tuner/cli.py`

1. Ensure the CLI can read the config file (which already includes the new fields).
2. Optionally add CLI flags for model paths (they should override config file):

   * `--estimator-name` / `-e` with choices `dummy`, `lexile_v2`.
   * `--lexile-v2-model-path`
   * `--lexile-v2-vectorizer-path`
   * `--lexile-v2-label-encoder-path`

   Example (if using `typer`):

   ```python
   import typer
   from .config import LexileTunerConfig
   from .pipeline import process_corpus
   from .pipeline import build_estimator_from_config
   from .rewriting import NoOpRewriter, LLMRewriter

   app = typer.Typer()

   @app.command()
   def analyze(
       input_path: str,
       config_path: str = typer.Option(None, help="Path to YAML config file."),
       estimator_name: str = typer.Option("dummy", help="Estimator name, e.g. 'dummy' or 'lexile_v2'."),
       lexile_v2_model_path: str = typer.Option("", help="Path to lexile-determination-v2 Keras model."),
       lexile_v2_vectorizer_path: str = typer.Option("", help="Path to lexile-determination-v2 vectorizer."),
       lexile_v2_label_encoder_path: str = typer.Option("", help="Path to lexile-determination-v2 label encoder."),
   ):
       config = load_config_somehow(config_path)  # existing logic
       config.estimator_name = estimator_name

       if estimator_name.lower() in ("lexile_v2", "lexile-determination-v2"):
           if lexile_v2_model_path:
               config.lexile_v2_model_path = lexile_v2_model_path
           if lexile_v2_vectorizer_path:
               config.lexile_v2_vectorizer_path = lexile_v2_vectorizer_path
           if lexile_v2_label_encoder_path:
               config.lexile_v2_label_encoder_path = lexile_v2_label_encoder_path

       estimator = build_estimator_from_config(config)
       rewriter = NoOpRewriter()  # or build from config

       documents = load_documents_from_input_path(input_path)
       results = process_corpus(documents, config, estimator, rewriter)
       # existing output logic...
   ```

   (Codex should wire this into the existing CLI structure rather than duplicating logic.)

---

## 7. Add example config and usage

**File:** `examples/example_config.yaml` (already exists)

Append an example section for `lexile_v2`:

```yaml
# Example configuration for using the lexile_v2 estimator

estimator_name: "lexile_v2"

lexile_v2_model_path: "/absolute/path/to/lexile-determination-v2/model/model.h5"
lexile_v2_vectorizer_path: "/absolute/path/to/lexile-determination-v2/model/vectorizer.pkl"
lexile_v2_label_encoder_path: "/absolute/path/to/lexile-determination-v2/model/label_encoder.pkl"

# Optional explicit mapping if labels are bands like '900-999'
lexile_v2_band_to_midpoint:
  "300-399": 350
  "400-499": 450
  "500-599": 550
```

**README.md**:

Add a short section:

* “Using the lexile_v2 estimator”

  * Explain that:

    * It requires external model artifacts from `lexile-determination-v2`.
    * Users must set `estimator_name: "lexile_v2"` and point to the model/vectorizer/labels in the config.
    * They need to install the extra dependencies: `pip install .[lexile-v2]`.

---

## 8. Add a minimal example script

**File:** `examples/run_lexile_v2_adapter.py` (new)

Create a small script that:

1. Loads config from YAML.
2. Builds `LexileDeterminationV2Estimator`.
3. Prints Lexile scores for a few hard-coded texts.

Example:

```python
from pathlib import Path

from lexile_corpus_tuner.config import LexileTunerConfig
from lexile_corpus_tuner.estimators import create_estimator, LexileEstimator


def main() -> None:
    # For simplicity, inline config; in real usage, load from YAML
    config = LexileTunerConfig(
        estimator_name="lexile_v2",
        lexile_v2_model_path="/absolute/path/to/model.h5",
        lexile_v2_vectorizer_path="/absolute/path/to/vectorizer.pkl",
        lexile_v2_label_encoder_path="/absolute/path/to/label_encoder.pkl",
    )

    estimator: LexileEstimator = create_estimator(
        config.estimator_name,
        model_path=config.lexile_v2_model_path or "",
        vectorizer_path=config.lexile_v2_vectorizer_path or "",
        label_encoder_path=config.lexile_v2_label_encoder_path,
        band_to_midpoint=config.lexile_v2_band_to_midpoint,
    )

    texts = [
        "The cat sat on the mat. It was raining outside, but the cat was warm and happy.",
        "Quantum entanglement is a physical phenomenon that occurs when pairs or groups of particles are generated, interact, or share spatial proximity in ways such that the quantum state of each particle cannot be described independently.",
    ]

    for t in texts:
        score = estimator.predict_scalar(t)
        print("Text:", t)
        print("Estimated Lexile:", score)
        print("-" * 40)


if __name__ == "__main__":
    main()
```

This gives the user a sanity check that lexile_v2 estimator is wired correctly before running the full corpus pipeline.

---

## 9. Testing strategy

**File:** `tests/test_lexile_v2_adapter.py` (new)

We don’t want tests to depend on a real TensorFlow model file, so:

1. Use `pytest` and `monkeypatch` to stub out:

   * `load_model` to return a simple object whose `predict` method returns a fixed probability distribution.
   * `joblib.load` to return:

     * A fake vectorizer whose `.transform` method returns a small numpy array.
     * A fake label encoder whose `.inverse_transform` returns a fixed label like `"900-999"`.

2. Example pseudo-test:

   ```python
   import numpy as np
   import types
   import pytest

   from lexile_corpus_tuner.estimators.lexile_determination_v2_adapter import LexileDeterminationV2Estimator


   class DummyModel:
       def predict(self, X, verbose: int = 0):
           # Always return a single distribution over 3 classes: class 1 is highest
           return np.array([[0.1, 0.8, 0.1]])


   class DummyVectorizer:
       def transform(self, texts):
           # Return a dummy numeric matrix with correct shape
           return np.array([[1.0, 2.0, 3.0]])


   class DummyLabelEncoder:
       def inverse_transform(self, indices):
           # Always map index 1 -> "900-999"
           return np.array(["900-999"])


   def test_predict_scalar_with_stubbed_artifacts(monkeypatch):
       # Stub load_model and joblib.load BEFORE importing inside class
       import lexile_corpus_tuner.estimators.lexile_determination_v2_adapter as adapter

       monkeypatch.setattr(adapter, "load_model", lambda path: DummyModel())
       monkeypatch.setattr(
           adapter, "joblib", types.SimpleNamespace(load=lambda path: DummyVectorizer() if "vectorizer" in path else DummyLabelEncoder())
       )

       est = LexileDeterminationV2Estimator(
           model_path="dummy_model.h5",
           vectorizer_path="dummy_vectorizer.pkl",
           label_encoder_path="dummy_labels.pkl",
       )

       score = est.predict_scalar("Some input text.")
       # For label "900-999" we expect midpoint (900 + 999) / 2 == 949.5
       assert abs(score - 949.5) < 1e-6
   ```

3. Mark tests that require TensorFlow/joblib as skipped if imports fail:

   ```python
   pytest.importorskip("tensorflow")
   pytest.importorskip("joblib")
   ```

   (But since we monkeypatch, we can avoid actual imports by patching in the module where used.)

