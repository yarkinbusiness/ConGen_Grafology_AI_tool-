"""Tests for grafology_ai.dataset.fixtures.generate_fixture_dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.dataset.schema import ManifestEntry
from grafology_ai.validation.validators import validate_sample


def test_generates_expected_number_of_images_and_manifest_entries(
    tmp_path: Path,
) -> None:
    entries = generate_fixture_dataset(tmp_path, count=10, seed=42)

    assert len(entries) == 10

    image_files = sorted((tmp_path / "images").glob("*.png"))
    assert len(image_files) == 10

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as f:
        raw_entries = json.load(f)
    assert len(raw_entries) == 10


def test_every_manifest_entry_round_trips_through_from_dict(tmp_path: Path) -> None:
    generate_fixture_dataset(tmp_path, count=10, seed=42)

    with (tmp_path / "manifest.json").open() as f:
        raw_entries = json.load(f)

    assert len(raw_entries) == 10
    for raw_entry in raw_entries:
        entry = ManifestEntry.from_dict(raw_entry)
        # A genuine round trip: re-serializing must reproduce the same dict.
        assert entry.to_dict() == raw_entry


def test_generated_images_exist_at_the_paths_implied_by_sample_id(
    tmp_path: Path,
) -> None:
    entries = generate_fixture_dataset(tmp_path, count=5, seed=1)

    for entry in entries:
        image_path = tmp_path / "images" / f"{entry.sample_id}.png"
        assert image_path.exists()


def test_generated_entries_have_no_fabricated_labels(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=5, seed=1)

    assert all(entry.labels == {} for entry in entries)


def test_deterministic_given_same_count_and_seed(tmp_path_factory) -> None:
    out1 = tmp_path_factory.mktemp("fixtures_run_1")
    out2 = tmp_path_factory.mktemp("fixtures_run_2")

    entries1 = generate_fixture_dataset(out1, count=10, seed=42)
    entries2 = generate_fixture_dataset(out2, count=10, seed=42)

    # Entry-by-entry equality (ManifestEntry is a frozen, comparable dataclass).
    assert entries1 == entries2

    # Byte-for-byte identical manifest JSON.
    manifest1 = (out1 / "manifest.json").read_bytes()
    manifest2 = (out2 / "manifest.json").read_bytes()
    assert manifest1 == manifest2

    # Byte-for-byte identical image content for at least one sample.
    sample_name = "synthetic-0000.png"
    image1 = (out1 / "images" / sample_name).read_bytes()
    image2 = (out2 / "images" / sample_name).read_bytes()
    assert image1 == image2


def test_quality_distribution_is_mixed_and_low_is_a_minority(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=20, seed=0)

    quality_counts = Counter(entry.quality for entry in entries)

    # A real mix of quality values appears, not a single repeated value.
    assert len(quality_counts) > 1

    # "low" is a minority of the dataset, roughly matching the intended
    # ~10% proportion (allow some slack rather than asserting exact counts).
    assert quality_counts["low"] < quality_counts["high"]
    assert quality_counts["low"] <= quality_counts["medium"] + quality_counts["high"]
    assert quality_counts["low"] / len(entries) <= 0.25


def test_low_quality_images_genuinely_fail_an_automated_check(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=20, seed=0)
    low_quality_entries = [e for e in entries if e.quality == "low"]
    assert low_quality_entries, "expected at least one low-quality fixture"

    found_a_failure = False
    for entry in low_quality_entries:
        image_path = tmp_path / "images" / f"{entry.sample_id}.png"
        results = validate_sample(image_path, quality_label=entry.quality)
        failing_checks = {"blur", "contrast", "resolution"}
        if any(
            r.check_name in failing_checks and r.verdict != "accept" for r in results
        ):
            found_a_failure = True
            break

    assert found_a_failure, (
        "expected at least one quality='low' fixture to fail blur, "
        "contrast, or resolution -- not just be labeled low"
    )


def test_high_quality_image_passes_every_automated_check(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=20, seed=0)
    high_quality_entries = [e for e in entries if e.quality == "high"]
    assert high_quality_entries, "expected at least one high-quality fixture"

    found_a_pass = False
    for entry in high_quality_entries:
        image_path = tmp_path / "images" / f"{entry.sample_id}.png"
        results = validate_sample(image_path, quality_label=entry.quality)
        if all(r.verdict == "accept" for r in results):
            found_a_pass = True
            break

    assert found_a_pass, (
        "expected at least one quality='high' fixture to pass every "
        "automated check"
    )


def test_sample_ids_are_deterministic_and_sequential(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=3, seed=7)
    assert [entry.sample_id for entry in entries] == [
        "synthetic-0000",
        "synthetic-0001",
        "synthetic-0002",
    ]


def test_acquisition_methods_and_language_are_valid(tmp_path: Path) -> None:
    entries = generate_fixture_dataset(tmp_path, count=20, seed=3)

    for entry in entries:
        assert entry.acquisition_method in ("photo", "scan")
        assert entry.language == "en"
