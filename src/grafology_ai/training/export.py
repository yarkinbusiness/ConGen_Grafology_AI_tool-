"""Save/load a trained :class:`~grafology_ai.training.baseline.BaselineModel`.

PLUMBING PROOF ONLY -- see ``grafology_ai/training/__init__.py``'s package
docstring. This module extends the same content-hash versioning *pattern*
used by :mod:`grafology_ai.pipeline.export` (read that module's docstring
and ``_compute_content_hash`` first) to trained model artifacts, so the
"how do I get a reproducible, comparable version identifier for an
artifact" answer is consistent across the dataset-snapshot side and the
training side of this pipeline. It does not modify or import from that
module's private internals -- it is a parallel implementation following
the same spirit for a different artifact shape.

Versioning
----------

A saved model is written to ``output_dir / f"{version}.json"``, where
``version`` is ``f"v_{content_hash[:VERSION_HASH_LENGTH]}"`` and
``content_hash`` is a SHA-256 digest of the model's own trained
parameters and metadata (weights, feature standardization stats, label
vocabulary, indicator name, and the hyperparameters used to train it) --
JSON-serialized with sorted keys, exactly like
:mod:`grafology_ai.pipeline.export`'s scheme, so the identifier depends
only on the model's actual content, never on a counter or a timestamp.
Two models trained on identical data with an identical seed are, per
:func:`grafology_ai.training.baseline.train_baseline`'s determinism
guarantee, bit-for-bit identical -- so they hash (and version) identically.
Training on different data, different synthetic labels, or a different
seed changes the trained weights and therefore the version identifier.

No pickle: the on-disk format is plain JSON (numbers, strings, lists) for
security/portability, matching this module's sibling
:mod:`grafology_ai.pipeline.export`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grafology_ai.training.baseline import BaselineModel

#: Number of hex characters of the SHA-256 content hash used in the
#: version identifier / filename. Matches
#: ``grafology_ai.pipeline.export.VERSION_HASH_LENGTH`` for consistency,
#: not because the two modules share code.
VERSION_HASH_LENGTH = 16

_JSON_INDENT = 2


@dataclass(frozen=True)
class SavedModelResult:
    """The outcome of saving a :class:`BaselineModel` to disk.

    Attributes:
        version: The content-derived version identifier, e.g.
            ``"v_3f9a1c2b7e4d5a6b"``.
        path: The file the model was written to (``output_dir /
            f"{version}.json"``).
    """

    version: str
    path: Path


def _model_to_dict(model: BaselineModel) -> dict[str, Any]:
    """Plain-JSON representation of every field that defines a trained model.

    Deliberately excludes nothing: the version identifier is meant to
    change if *any* trained parameter or training-time metadata differs
    between two models, per this module's versioning guarantee.
    """
    return {
        "indicator": model.indicator,
        "feature_names": list(model.feature_names),
        "classes": list(model.classes),
        "weights": [list(row) for row in model.weights],
        "feature_mean": list(model.feature_mean),
        "feature_std": list(model.feature_std),
        "seed": model.seed,
        "n_iterations": model.n_iterations,
        "learning_rate": model.learning_rate,
    }


def _model_from_dict(data: dict[str, Any]) -> BaselineModel:
    return BaselineModel(
        indicator=data["indicator"],
        feature_names=tuple(data["feature_names"]),
        classes=tuple(data["classes"]),
        weights=tuple(tuple(float(v) for v in row) for row in data["weights"]),
        feature_mean=tuple(float(v) for v in data["feature_mean"]),
        feature_std=tuple(float(v) for v in data["feature_std"]),
        seed=int(data["seed"]),
        n_iterations=int(data["n_iterations"]),
        learning_rate=float(data["learning_rate"]),
    )


def _compute_model_content_hash(model: BaselineModel) -> str:
    """SHA-256 hex digest over everything that defines ``model``'s content.

    Sorted-key JSON serialization, matching
    ``grafology_ai.pipeline.export._compute_content_hash``'s approach --
    see the module docstring's versioning guarantee.
    """
    serialized = json.dumps(_model_to_dict(model), sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def save_model(model: BaselineModel, output_dir: Path) -> SavedModelResult:
    """Save a trained, plumbing-proof :class:`BaselineModel` as plain JSON.

    NOT a real deployable model artifact -- see the package docstring. See
    the module docstring for the versioning scheme.

    Args:
        output_dir: Directory to write the model file into. Created if it
            does not already exist.

    Returns:
        A :class:`SavedModelResult` with the version identifier and the
        file path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _compute_model_content_hash(model)
    version = f"v_{content_hash[:VERSION_HASH_LENGTH]}"
    path = output_dir / f"{version}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(_model_to_dict(model), f, indent=_JSON_INDENT, sort_keys=True)
        f.write("\n")

    return SavedModelResult(version=version, path=path)


def load_model(path: Path) -> BaselineModel:
    """Load a :class:`BaselineModel` previously written by :func:`save_model`.

    Plain JSON parsing only -- no pickle, no code execution -- so this is
    safe to call on a file from an untrusted source (subject to the usual
    caveat that its *content* is still just numbers taken at face value,
    not verified against any schema beyond basic structural parsing).

    Returns:
        A :class:`BaselineModel` reconstructed from the file, usable with
        :func:`grafology_ai.training.baseline.predict` exactly like the
        original in-memory model.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return _model_from_dict(data)
