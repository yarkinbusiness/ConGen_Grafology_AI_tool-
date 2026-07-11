"""Tests for grafology_ai.training -- a PLUMBING PROOF, not a real model.

Every test in this file exercises the train -> predict -> save -> load ->
version *machinery* end to end against synthetic fixture images
(:func:`generate_fixture_dataset`) and fabricated, documented-arbitrary
synthetic labels (:mod:`grafology_ai.training.synthetic_labels`). None of
this proves anything about real graphological pressure assessment -- see
``src/grafology_ai/training/__init__.py``'s package docstring. The goal
here is to prove the wiring works (determinism, save/load fidelity,
content-hash versioning, and that predictions are driven by real input
signal), not to hit any accuracy bar on fake data.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from grafology_ai.analysis import Features, analyze
from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.training import (
    PRESSURE_LABELS,
    BaselineModel,
    generate_synthetic_pressure_labels,
    load_model,
    predict,
    save_model,
    train_baseline,
)

FIXTURE_COUNT = 24
FIXTURE_SEED = 0


def _fixture_features(tmp_path: Path, *, count: int = FIXTURE_COUNT, seed: int = FIXTURE_SEED) -> list[Features]:
    """Generate a fixture dataset and run analyze() on every image.

    Purely a test helper: builds the (Features list) input that
    train_baseline()/predict() consume, exactly as a real caller would
    (generate_fixture_dataset() -> analyze() per image), since no real
    labeled dataset exists to test against instead.
    """
    raw_dir = tmp_path / f"raw-{seed}-{count}"
    entries = generate_fixture_dataset(raw_dir, count=count, seed=seed)
    images_dir = raw_dir / "images"
    return [analyze(images_dir / f"{entry.sample_id}.png") for entry in entries]


def _make_training_data(tmp_path: Path, *, count: int = FIXTURE_COUNT, seed: int = FIXTURE_SEED):
    features_list = _fixture_features(tmp_path, count=count, seed=seed)
    labels = generate_synthetic_pressure_labels(features_list)
    return list(zip(features_list, labels)), features_list, labels


# --- end-to-end -----------------------------------------------------------


def test_end_to_end_pipeline_produces_known_vocabulary_predictions(tmp_path: Path) -> None:
    training_data, features_list, labels = _make_training_data(tmp_path)

    # A meaningful mix of synthetic labels should exist across 24 varied
    # fixture samples -- otherwise this test wouldn't actually exercise a
    # multi-class model.
    assert len(set(labels)) >= 2
    assert set(labels) <= set(PRESSURE_LABELS)

    # Hold out the last 6 samples' features as "unseen" inputs.
    train_pairs = training_data[:-6]
    held_out = features_list[-6:]

    model = train_baseline(train_pairs, seed=0)
    assert isinstance(model, BaselineModel)
    assert model.indicator == "pressure"
    assert set(model.classes) <= set(PRESSURE_LABELS)

    predictions = [predict(model, features) for features in held_out]
    assert len(predictions) == len(held_out)
    for prediction in predictions:
        assert prediction in model.classes
        assert prediction in PRESSURE_LABELS


def test_train_baseline_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        train_baseline([], seed=0)


# --- determinism ------------------------------------------------------------


def test_train_baseline_is_deterministic(tmp_path: Path) -> None:
    training_data, features_list, _labels = _make_training_data(tmp_path)

    model_a = train_baseline(training_data, seed=7)
    model_b = train_baseline(training_data, seed=7)

    # Bit-for-bit identical trained parameters.
    assert model_a.weights == model_b.weights
    assert model_a.feature_mean == model_b.feature_mean
    assert model_a.feature_std == model_b.feature_std
    assert model_a.classes == model_b.classes

    # And therefore label-for-label identical predictions on the same data.
    predictions_a = [predict(model_a, f) for f in features_list]
    predictions_b = [predict(model_b, f) for f in features_list]
    assert predictions_a == predictions_b


# --- save / load round trip -------------------------------------------------


def test_save_load_round_trip_predictions_match(tmp_path: Path) -> None:
    training_data, features_list, _labels = _make_training_data(tmp_path)
    model = train_baseline(training_data, seed=3)

    result = save_model(model, tmp_path / "models")
    assert result.path.exists()
    assert result.path.suffix == ".json"
    assert result.version.startswith("v_")

    loaded = load_model(result.path)
    assert dataclasses.astuple(loaded) == dataclasses.astuple(model)

    original_predictions = [predict(model, f) for f in features_list]
    loaded_predictions = [predict(loaded, f) for f in features_list]
    assert original_predictions == loaded_predictions


# --- versioning ---------------------------------------------------------


def test_versioning_same_data_same_seed_same_version(tmp_path: Path) -> None:
    training_data, _features_list, _labels = _make_training_data(tmp_path, seed=1)

    model_a = train_baseline(training_data, seed=5)
    model_b = train_baseline(training_data, seed=5)

    result_a = save_model(model_a, tmp_path / "models_a")
    result_b = save_model(model_b, tmp_path / "models_b")

    assert result_a.version == result_b.version
    assert result_a.path.read_bytes() == result_b.path.read_bytes()


def test_versioning_changes_with_different_seed(tmp_path: Path) -> None:
    training_data, _features_list, _labels = _make_training_data(tmp_path, seed=2)

    model_a = train_baseline(training_data, seed=0)
    model_b = train_baseline(training_data, seed=999)

    result_a = save_model(model_a, tmp_path / "models_a")
    result_b = save_model(model_b, tmp_path / "models_b")

    assert result_a.version != result_b.version


def test_versioning_changes_with_different_synthetic_labels(tmp_path: Path) -> None:
    # Two different fixture seeds produce different underlying Features,
    # and therefore different fabricated synthetic labels -- "different
    # data" per the versioning guarantee in
    # grafology_ai.training.export's docstring.
    training_data_a, _f_a, labels_a = _make_training_data(tmp_path, seed=10)
    training_data_b, _f_b, labels_b = _make_training_data(tmp_path, seed=11)
    assert labels_a != labels_b or training_data_a != training_data_b

    model_a = train_baseline(training_data_a, seed=0)
    model_b = train_baseline(training_data_b, seed=0)

    result_a = save_model(model_a, tmp_path / "models_a")
    result_b = save_model(model_b, tmp_path / "models_b")

    assert result_a.version != result_b.version


# --- sanity / plumbing check: the model uses real signal --------------------


def test_predictions_vary_with_meaningfully_different_input_features(tmp_path: Path) -> None:
    """The model must not just always predict the majority class regardless
    of input -- confirming real signal flows through the wiring. A full
    accuracy bar is not required (this is fake data), just that varying
    the input meaningfully varies the output."""
    training_data, _features_list, labels = _make_training_data(tmp_path, count=30, seed=4)
    model = train_baseline(training_data, seed=0)

    base = training_data[0][0]

    def _with_stroke_width(mean: float) -> Features:
        return dataclasses.replace(base, stroke_width_mean=mean, stroke_width_std=0.5)

    light_prediction = predict(model, _with_stroke_width(1.0))
    firm_prediction = predict(model, _with_stroke_width(15.0))

    assert light_prediction in model.classes
    assert firm_prediction in model.classes
    assert light_prediction != firm_prediction
