"""Tests for grafology_ai.pipeline (ingest, validate, split, export)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.dataset.schema import ManifestEntry
from grafology_ai.pipeline.export import export_snapshot
from grafology_ai.pipeline.ingest import IngestedSample, ingest_dataset
from grafology_ai.pipeline.run import run_pipeline
from grafology_ai.pipeline.split import split_dataset
from grafology_ai.pipeline.validate import partition_by_status, validate_dataset

FIXTURE_COUNT = 20
FIXTURE_SEED = 0


# --- ingest -----------------------------------------------------------


def test_ingest_reads_every_manifest_entry_with_resolved_image_path(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    entries = generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    samples = ingest_dataset(raw_dir)

    assert len(samples) == len(entries)
    assert all(isinstance(sample, IngestedSample) for sample in samples)
    assert [s.entry.sample_id for s in samples] == [e.sample_id for e in entries]
    for sample in samples:
        assert sample.image_path.exists()
        assert sample.image_path == raw_dir / "images" / f"{sample.entry.sample_id}.png"


def test_ingest_raises_clear_error_for_missing_image(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=5, seed=1)

    missing_image = raw_dir / "images" / "synthetic-0000.png"
    missing_image.unlink()

    with pytest.raises(FileNotFoundError, match="synthetic-0000"):
        ingest_dataset(raw_dir)


def test_ingest_raises_for_missing_manifest(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        ingest_dataset(empty_dir)


# --- validate -----------------------------------------------------------


def test_validate_rejected_samples_excluded_and_flagged_samples_included(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    samples = ingest_dataset(raw_dir)

    validated = validate_dataset(samples)
    included, rejected = partition_by_status(validated)

    # Every rejected sample genuinely has a "reject" verdict on some check.
    for sample in rejected:
        assert sample.status == "rejected"
        assert any(r.verdict == "reject" for r in sample.results)

    # Every included sample has no "reject" verdict at all.
    for sample in included:
        assert sample.status in ("accepted", "flagged")
        assert all(r.verdict != "reject" for r in sample.results)

    # A sample with only "flag" verdicts (no reject) is included, not
    # excluded -- this is the core "flag != reject" requirement. The
    # fixture generator's "low" quality tier is tuned to fail blur/contrast,
    # which validate_sample downgrades reject->flag for quality="low"
    # samples, so we expect at least one flagged-but-included sample here.
    flagged = [s for s in included if s.status == "flagged"]
    assert flagged, "expected at least one flagged (but included) sample"
    for sample in flagged:
        assert any(r.verdict == "flag" for r in sample.results)
        assert all(r.verdict != "reject" for r in sample.results)

    # included + rejected covers every validated sample exactly once.
    assert len(included) + len(rejected) == len(validated)
    included_ids = {s.entry.sample_id for s in included}
    rejected_ids = {s.entry.sample_id for s in rejected}
    assert included_ids.isdisjoint(rejected_ids)
    assert included_ids | rejected_ids == {s.entry.sample_id for s in validated}


# --- split ----------------------------------------------------------------


def _accepted_entries(tmp_path: Path, subdir: str) -> list[ManifestEntry]:
    raw_dir = tmp_path / subdir
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    validated = validate_dataset(ingest_dataset(raw_dir))
    included, _rejected = partition_by_status(validated)
    return [sample.entry for sample in included]


def test_split_is_deterministic_given_same_input_and_seed(tmp_path: Path) -> None:
    entries = _accepted_entries(tmp_path, "raw")

    split1 = split_dataset(entries, seed=42)
    split2 = split_dataset(entries, seed=42)

    assert [e.sample_id for e in split1.train] == [e.sample_id for e in split2.train]
    assert [e.sample_id for e in split1.val] == [e.sample_id for e in split2.val]
    assert [e.sample_id for e in split1.test] == [e.sample_id for e in split2.test]


def test_split_is_stable_under_reordered_input(tmp_path: Path) -> None:
    entries = _accepted_entries(tmp_path, "raw")
    reversed_entries = list(reversed(entries))

    split_forward = split_dataset(entries, seed=7)
    split_reversed = split_dataset(reversed_entries, seed=7)

    assert {e.sample_id for e in split_forward.train} == {
        e.sample_id for e in split_reversed.train
    }
    assert {e.sample_id for e in split_forward.val} == {
        e.sample_id for e in split_reversed.val
    }
    assert {e.sample_id for e in split_forward.test} == {
        e.sample_id for e in split_reversed.test
    }


def test_split_is_disjoint_and_covers_every_accepted_sample(tmp_path: Path) -> None:
    entries = _accepted_entries(tmp_path, "raw")

    split = split_dataset(entries, seed=3)

    train_ids = {e.sample_id for e in split.train}
    val_ids = {e.sample_id for e in split.val}
    test_ids = {e.sample_id for e in split.test}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    all_ids = {e.sample_id for e in entries}
    assert train_ids | val_ids | test_ids == all_ids
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_ids)


def test_split_rejects_ratios_that_do_not_sum_to_one(tmp_path: Path) -> None:
    entries = _accepted_entries(tmp_path, "raw")

    with pytest.raises(ValueError, match="ratio"):
        split_dataset(entries, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# --- export / end-to-end ---------------------------------------------------


def test_end_to_end_pipeline_produces_expected_snapshot_structure(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    export_dir = tmp_path / "export"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    result = run_pipeline(raw_dir, export_dir, seed=0)

    assert result.snapshot_dir == export_dir / result.version
    assert result.version.startswith("v_")
    assert result.snapshot_dir.is_dir()

    expected_files = {
        "train_manifest.json",
        "val_manifest.json",
        "test_manifest.json",
        "rejected_manifest.json",
        "validation_report.json",
    }
    actual_files = {p.name for p in result.snapshot_dir.iterdir() if p.is_file()}
    assert expected_files <= actual_files

    images_dir = result.snapshot_dir / "images"
    assert images_dir.is_dir()
    image_files = list(images_dir.glob("*.png"))
    assert len(image_files) > 0

    with (result.snapshot_dir / "train_manifest.json").open() as f:
        train_manifest = json.load(f)
    with (result.snapshot_dir / "val_manifest.json").open() as f:
        val_manifest = json.load(f)
    with (result.snapshot_dir / "test_manifest.json").open() as f:
        test_manifest = json.load(f)
    with (result.snapshot_dir / "rejected_manifest.json").open() as f:
        rejected_manifest = json.load(f)
    with (result.snapshot_dir / "validation_report.json").open() as f:
        validation_report = json.load(f)

    # Every train/val/test entry round-trips through ManifestEntry, and
    # has a copied image file in images/.
    all_split_ids = set()
    for manifest in (train_manifest, val_manifest, test_manifest):
        for raw_entry in manifest:
            entry = ManifestEntry.from_dict(raw_entry)
            all_split_ids.add(entry.sample_id)
            assert (images_dir / f"{entry.sample_id}.png").exists()

    # Rejected entries have rejection_reasons and are disjoint from the splits.
    rejected_ids = set()
    for raw_entry in rejected_manifest:
        assert raw_entry["rejection_reasons"]
        rejected_ids.add(raw_entry["sample_id"])
    assert all_split_ids.isdisjoint(rejected_ids)

    # validation_report summary counts are internally consistent.
    summary = validation_report["summary"]
    assert summary["total_samples"] == FIXTURE_COUNT
    assert summary["accepted"] + summary["flagged"] + summary["rejected"] == FIXTURE_COUNT
    assert summary["rejected"] == len(rejected_ids)
    assert len(validation_report["samples"]) == FIXTURE_COUNT


def test_export_flagged_samples_are_included_in_a_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    export_dir = tmp_path / "export"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    validated = validate_dataset(ingest_dataset(raw_dir))
    flagged_ids = {s.entry.sample_id for s in validated if s.status == "flagged"}
    assert flagged_ids, "expected at least one flagged sample from the fixture set"

    result = run_pipeline(raw_dir, export_dir, seed=0)

    all_split_ids = set()
    for name in ("train_manifest.json", "val_manifest.json", "test_manifest.json"):
        with (result.snapshot_dir / name).open() as f:
            all_split_ids.update(entry["sample_id"] for entry in json.load(f))

    assert flagged_ids <= all_split_ids


def test_versioning_is_deterministic_across_independently_generated_datasets(
    tmp_path: Path,
) -> None:
    raw_dir_1 = tmp_path / "raw1"
    raw_dir_2 = tmp_path / "raw2"
    export_dir_1 = tmp_path / "export1"
    export_dir_2 = tmp_path / "export2"

    generate_fixture_dataset(raw_dir_1, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    generate_fixture_dataset(raw_dir_2, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    result1 = run_pipeline(raw_dir_1, export_dir_1, seed=0)
    result2 = run_pipeline(raw_dir_2, export_dir_2, seed=0)

    assert result1.version == result2.version

    for name in (
        "train_manifest.json",
        "val_manifest.json",
        "test_manifest.json",
        "rejected_manifest.json",
        "validation_report.json",
    ):
        content1 = (result1.snapshot_dir / name).read_bytes()
        content2 = (result2.snapshot_dir / name).read_bytes()
        assert content1 == content2, f"{name} differed between identical runs"

    # Byte-identical image content too, for at least one shared sample.
    images1 = sorted((result1.snapshot_dir / "images").glob("*.png"))
    images2 = sorted((result2.snapshot_dir / "images").glob("*.png"))
    assert [p.name for p in images1] == [p.name for p in images2]
    assert images1[0].read_bytes() == images2[0].read_bytes()


def test_versioning_changes_with_different_fixture_seed(tmp_path: Path) -> None:
    raw_dir_a = tmp_path / "raw_a"
    raw_dir_b = tmp_path / "raw_b"
    export_dir_a = tmp_path / "export_a"
    export_dir_b = tmp_path / "export_b"

    generate_fixture_dataset(raw_dir_a, count=FIXTURE_COUNT, seed=0)
    generate_fixture_dataset(raw_dir_b, count=FIXTURE_COUNT, seed=999)

    result_a = run_pipeline(raw_dir_a, export_dir_a, seed=0)
    result_b = run_pipeline(raw_dir_b, export_dir_b, seed=0)

    assert result_a.version != result_b.version


def test_export_snapshot_directly_matches_run_pipeline(tmp_path: Path) -> None:
    """export_snapshot() called directly with the intermediate objects
    produces the same version as the run_pipeline() convenience wrapper."""
    raw_dir = tmp_path / "raw"
    export_dir = tmp_path / "export"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    ingested = ingest_dataset(raw_dir)
    validated = validate_dataset(ingested)
    included, _rejected = partition_by_status(validated)
    split = split_dataset([s.entry for s in included], seed=0)

    manual_result = export_snapshot(export_dir, validated, split)
    pipeline_result = run_pipeline(raw_dir, tmp_path / "export2", seed=0)

    assert manual_result.version == pipeline_result.version
