"""A hand-rolled baseline classifier -- PLUMBING PROOF ONLY, not a real model.

See the package docstring (``grafology_ai/training/__init__.py``) for the
full context. In short: this module trains and runs a small, from-scratch
multinomial logistic regression, implemented in plain numpy (no
scikit-learn/torch -- this project adds no ML dependency, in keeping with
the hand-rolled numpy algorithms already in
:mod:`grafology_ai.analysis.features` and
:mod:`grafology_ai.validation.validators`). It exists solely to prove that
a train -> predict -> save -> load pipeline can run end to end against
:class:`~grafology_ai.analysis.features.Features`; it is NOT trained on
real graphological data and its predictions carry no diagnostic meaning
whatsoever.

Indicator choice
-----------------

Of the indicator vocabulary in ``docs/labeling_rubric.md``, this module
targets **"Pressure"** as the one concrete, testable case: the rubric
defines pressure as "the apparent force behind the strokes, inferred from
stroke weight" -- and :class:`~grafology_ai.analysis.features.Features`
already measures exactly that directly, as ``stroke_width_mean`` (a
pressure proxy: thicker measured strokes read as heavier pen pressure) and
``stroke_width_std`` (consistency of that pressure across the sample). That
makes pressure the indicator with the most direct, already-measured,
continuous feature-to-label path in the current codebase -- an easy,
concrete target for a plumbing proof, unlike indicators (e.g. "Overall
Organization") that would need new feature engineering first.

Model
-----

A multinomial (softmax) logistic regression over the 2-dimensional feature
vector ``[stroke_width_mean, stroke_width_std]``, trained by full-batch
gradient descent from a random initialization. Deliberately minimal:

- Features are standardized (zero mean, unit variance) using statistics
  computed from the training set only, and those statistics are stored on
  the model so inference reproduces the same scaling.
- Weights are randomly initialized from a seeded
  :class:`numpy.random.Generator` (never numpy's global random state), so
  :func:`train_baseline` is fully deterministic given the same input and
  seed.
- Training deliberately runs a small, *fixed* number of gradient-descent
  steps rather than iterating to convergence. This keeps the final weights
  sensitive to the random initialization (i.e. to ``seed``) -- appropriate
  for a plumbing proof whose versioning tests need "different seed produces
  a different trained model", and irrelevant to prediction quality since
  there is no real accuracy bar for fake data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grafology_ai.analysis.features import Features

#: The single rubric indicator this plumbing-proof baseline targets. See
#: the module docstring's "Indicator choice" section.
INDICATOR = "pressure"

#: Names of the Features fields used as the model's input vector, in
#: order. Both are direct pressure proxies per docs/labeling_rubric.md
#: (see module docstring).
FEATURE_NAMES: tuple[str, str] = ("stroke_width_mean", "stroke_width_std")

#: Fixed gradient-descent hyperparameters. Not tuned against any held-out
#: accuracy metric -- there is no real labeled data to tune against yet --
#: chosen only to make the plumbing (loss decreases, predictions vary with
#: input) visibly work on synthetic data. Revisit once real data exists.
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_N_ITERATIONS = 300

#: Standard deviation of the initial random weights. Small, so early
#: gradient-descent steps aren't dominated by an arbitrary large init.
_WEIGHT_INIT_SCALE = 0.01


@dataclass(frozen=True)
class BaselineModel:
    """A trained plumbing-proof baseline classifier for one rubric indicator.

    NOT a real graphological model -- see the module docstring. Every
    field here is a plain, JSON-serializable value on purpose (see
    :mod:`grafology_ai.training.export`), so the model can be saved without
    pickle.

    Attributes:
        indicator: The rubric indicator this model predicts (always
            ``"pressure"`` for this plumbing proof, see :data:`INDICATOR`).
        feature_names: Names of the :class:`Features` fields used as the
            model's input vector, in order (see :data:`FEATURE_NAMES`).
        classes: The label vocabulary seen during training, sorted for a
            deterministic, canonical ordering. :func:`predict` always
            returns one of these strings.
        weights: The trained weight matrix, as a tuple of
            ``len(classes)`` rows, each a tuple of ``len(feature_names) + 1``
            floats (a bias term followed by one weight per feature) --
            i.e. row ``i`` is the softmax logit weight vector for
            ``classes[i]``.
        feature_mean: Per-feature mean computed from the training set,
            used to standardize inputs at inference time.
        feature_std: Per-feature standard deviation computed from the
            training set (never zero -- see :func:`train_baseline`), used
            to standardize inputs at inference time.
        seed: The seed :func:`train_baseline` was called with.
        n_iterations: Number of full-batch gradient-descent steps used to
            train this model.
        learning_rate: Gradient-descent step size used to train this
            model.
    """

    indicator: str
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    seed: int
    n_iterations: int
    learning_rate: float


def _extract_feature_vector(features: Features) -> np.ndarray:
    """Pull the pressure-proxy feature vector out of a :class:`Features` record.

    Plumbing-proof detail, not a real feature-engineering choice beyond
    what's documented in the module docstring: just the two stroke-width
    fields, in :data:`FEATURE_NAMES` order.
    """
    return np.array([features.stroke_width_mean, features.stroke_width_std], dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax over a 2D array of logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_baseline(
    features_and_labels: list[tuple[Features, str]],
    seed: int = 0,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_iterations: int = DEFAULT_N_ITERATIONS,
) -> BaselineModel:
    """Train the plumbing-proof "pressure" baseline classifier.

    NOT a real graphological training procedure -- see the module
    docstring. Trains a from-scratch multinomial logistic regression (full
    -batch gradient descent, plain numpy) on
    ``[stroke_width_mean, stroke_width_std]`` -> label pairs.

    Deterministic: calling this twice with the same ``features_and_labels``
    (same content, any order -- see below) and the same ``seed`` produces a
    bit-for-bit identical :class:`BaselineModel`, since every random choice
    (only the initial weights) is drawn from a fresh
    ``numpy.random.default_rng(seed)`` and every other step is plain
    arithmetic over the input in the order given.

    Args:
        features_and_labels: Training data as a list of
            ``(Features, label)`` pairs. In a real pipeline these labels
            would be graphologist ground truth; in this plumbing proof
            they are expected to come from
            :mod:`grafology_ai.training.synthetic_labels` (fabricated, for
            testing only). Must be non-empty.
        seed: Seed for the weight initialization's random generator.
        learning_rate: Gradient-descent step size. Exposed for tests;
            defaults to :data:`DEFAULT_LEARNING_RATE`.
        n_iterations: Number of full-batch gradient-descent steps.
            Exposed for tests; defaults to :data:`DEFAULT_N_ITERATIONS`.

    Returns:
        A trained :class:`BaselineModel`.

    Raises:
        ValueError: If ``features_and_labels`` is empty.
    """
    if not features_and_labels:
        raise ValueError("train_baseline() requires at least one (Features, label) pair")

    x = np.stack([_extract_feature_vector(f) for f, _label in features_and_labels])
    raw_labels = [label for _f, label in features_and_labels]
    classes = tuple(sorted(set(raw_labels)))
    class_index = {label: i for i, label in enumerate(classes)}

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    # Guard against a zero-variance feature (e.g. every sample has an
    # identical stroke width in a tiny/degenerate training set): treat it
    # as std=1 so standardization is a no-op (just subtracts the constant
    # mean) instead of dividing by zero.
    feature_std_safe = np.where(feature_std == 0.0, 1.0, feature_std)

    x_norm = (x - feature_mean) / feature_std_safe
    n_samples = x_norm.shape[0]
    x_design = np.hstack([np.ones((n_samples, 1)), x_norm])  # bias column

    n_classes = len(classes)
    n_params = x_design.shape[1]

    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    for row, label in enumerate(raw_labels):
        y_onehot[row, class_index[label]] = 1.0

    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=_WEIGHT_INIT_SCALE, size=(n_classes, n_params))

    for _ in range(n_iterations):
        logits = x_design @ weights.T
        probs = _softmax(logits)
        gradient = (probs - y_onehot).T @ x_design / n_samples
        weights = weights - learning_rate * gradient

    return BaselineModel(
        indicator=INDICATOR,
        feature_names=FEATURE_NAMES,
        classes=classes,
        weights=tuple(tuple(float(v) for v in row) for row in weights),
        feature_mean=tuple(float(v) for v in feature_mean),
        feature_std=tuple(float(v) for v in feature_std_safe),
        seed=seed,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
    )


def predict(model: BaselineModel, features: Features) -> str:
    """Run inference with a plumbing-proof :class:`BaselineModel`.

    NOT a real graphological prediction -- see the module docstring.
    Standardizes ``features`` using the model's stored training-set
    mean/std, computes softmax class probabilities, and returns the
    highest-probability label from ``model.classes``.

    Args:
        model: A trained :class:`BaselineModel` (from :func:`train_baseline`
            or :func:`grafology_ai.training.export.load_model`).
        features: The sample's measured :class:`Features`.

    Returns:
        One of the strings in ``model.classes``.
    """
    x = _extract_feature_vector(features)
    mean = np.array(model.feature_mean, dtype=np.float64)
    std = np.array(model.feature_std, dtype=np.float64)
    x_norm = (x - mean) / std
    x_design = np.concatenate([[1.0], x_norm])

    weights = np.array(model.weights, dtype=np.float64)
    logits = weights @ x_design
    probs = _softmax(logits[np.newaxis, :])[0]
    best_index = int(np.argmax(probs))
    return model.classes[best_index]
