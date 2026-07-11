"""Tests for grafology_ai.intake (run_intake) and grafology_ai.intake_cli (main).

Cross-checks `run_intake`'s counts against calling `ingest_dataset`/
`validate_dataset` directly (the same functions it reuses internally), and
exercises the manifest-level-error path with hand-crafted broken datasets,
per the module's documented "no crash, classify the root cause" contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.intake import run_intake
from grafology_ai.intake_cli import main
from grafology_ai.pipeline.ingest import ingest_dataset
from grafology_ai.pipeline.validate import validate_dataset

FIXTURE_COUNT = 20
FIXTURE_SEED = 0

# seed=6 / count=20 is known (see comment on the "clean dataset" test below)
# to produce zero rejected samples for this fixture generator.
CLEAN_FIXTURE_SEED = 6


# --- run_intake: counts match ingest_dataset/validate_dataset directly --------


def test_run_intake_counts_are_internally_consistent_and_match_reality(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    report = run_intake(raw_dir)

    # Internally consistent.
    assert report.accepted + report.flagged + report.rejected == report.total_samples
    assert not report.manifest_errors

    # Matches calling the underlying pipeline stages directly.
    validated = validate_dataset(ingest_dataset(raw_dir))
    expected_accepted = sum(1 for s in validated if s.status == "accepted")
    expected_flagged = sum(1 for s in validated if s.status == "flagged")
    expected_rejected = sum(1 for s in validated if s.status == "rejected")

    assert report.total_samples == len(validated)
    assert report.accepted == expected_accepted
    assert report.flagged == expected_flagged
    assert report.rejected == expected_rejected

    # Known distribution for count=20, seed=0: some samples fail blur.
    assert report.rejected > 0
    expected_rejected_ids = {s.entry.sample_id for s in validated if s.status == "rejected"}
    reported_rejected_ids = {s.sample_id for s in report.rejected_samples}
    assert reported_rejected_ids == expected_rejected_ids

    # Every rejected sample's reported failed_checks matches its real
    # rejection_results (check_name, reason, measured_value).
    validated_by_id = {s.entry.sample_id: s for s in validated}
    for rejected_sample in report.rejected_samples:
        real = validated_by_id[rejected_sample.sample_id]
        expected_checks = [
            {
                "check_name": r.check_name,
                "reason": r.reason,
                "measured_value": r.measured_value,
            }
            for r in real.rejection_results
        ]
        assert rejected_sample.failed_checks == expected_checks
        assert expected_checks, "a rejected sample must have at least one failed check"

    # Fix-it checklist mentions the real, failing check (blur), and only
    # real categories -- not every possible category.
    checklist_text = " ".join(report.fix_it_checklist)
    assert "blur" in checklist_text.lower()
    assert str(report.rejected) in checklist_text


# --- run_intake: broken manifest handled gracefully, root cause distinguishable ---


def _copy_fixture_images(raw_dir: Path, broken_dir: Path, sample_ids: list[str]) -> None:
    generate_fixture_dataset(raw_dir, count=5, seed=1)
    images_dir = broken_dir / "images"
    images_dir.mkdir(parents=True)
    for sample_id in sample_ids:
        src = raw_dir / "images" / f"{sample_id}.png"
        (images_dir / f"{sample_id}.png").write_bytes(src.read_bytes())


def test_run_intake_reports_missing_manifest_without_crashing(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    report = run_intake(empty_dir)

    assert report.total_samples == 0
    assert report.accepted == report.flagged == report.rejected == 0
    assert not report.is_clean
    assert len(report.manifest_errors) == 1
    assert report.manifest_errors[0].error_type == "missing_manifest"
    assert report.fix_it_checklist  # not empty -- the error is surfaced


def test_run_intake_reports_invalid_json_without_crashing(tmp_path: Path) -> None:
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")

    report = run_intake(broken_dir)

    assert not report.is_clean
    assert len(report.manifest_errors) == 1
    assert report.manifest_errors[0].error_type == "invalid_json"


def test_run_intake_distinguishes_missing_image_from_schema_failure(
    tmp_path: Path,
) -> None:
    """A hand-crafted manifest with two distinct broken entries: one
    referencing a missing image file, one with an invalid `quality` value
    that fails `ManifestEntry` schema validation. `run_intake` must not
    crash, and the root cause of each problem must be distinguishable in
    the report (not conflated into one generic error)."""
    raw_dir = tmp_path / "raw_source"
    broken_dir = tmp_path / "broken"
    _copy_fixture_images(raw_dir, broken_dir, ["synthetic-0000", "synthetic-0001"])

    manifest = [
        {
            "sample_id": "synthetic-0000",
            "acquisition_method": "photo",
            "quality": "high",
            "language": "en",
            "labels": {},
        },
        {
            # References an image that was never copied into broken_dir/images/.
            "sample_id": "does-not-exist-0099",
            "acquisition_method": "scan",
            "quality": "high",
            "language": "en",
            "labels": {},
        },
        {
            "sample_id": "synthetic-0001",
            "acquisition_method": "photo",
            # Invalid: not one of "high"/"medium"/"low".
            "quality": "excellent",
            "language": "en",
            "labels": {},
        },
    ]
    (broken_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_intake(broken_dir)

    assert not report.is_clean
    assert report.total_samples == 0
    assert report.accepted == report.flagged == report.rejected == 0

    error_types = {e.error_type for e in report.manifest_errors}
    assert "missing_image_file" in error_types
    assert "schema_validation_failed" in error_types

    missing_image_errors = [e for e in report.manifest_errors if e.error_type == "missing_image_file"]
    assert any(e.sample_id == "does-not-exist-0099" for e in missing_image_errors)

    schema_errors = [e for e in report.manifest_errors if e.error_type == "schema_validation_failed"]
    assert any(e.sample_id == "synthetic-0001" for e in schema_errors)

    # The valid entry (synthetic-0000) produced no error at all.
    assert all(e.sample_id != "synthetic-0000" for e in report.manifest_errors)

    # Both root causes show up, distinctly, in the human-readable checklist too.
    checklist_text = " ".join(report.fix_it_checklist)
    assert "missing_image_file" in checklist_text
    assert "schema_validation_failed" in checklist_text


def test_run_intake_reports_invalid_manifest_structure(tmp_path: Path) -> None:
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "images").mkdir()
    # A JSON object, not an array.
    (broken_dir / "manifest.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    report = run_intake(broken_dir)

    assert not report.is_clean
    assert len(report.manifest_errors) == 1
    assert report.manifest_errors[0].error_type == "invalid_manifest_structure"


# --- run_intake: zero-rejected dataset gives an empty (not boilerplate) checklist --


def test_run_intake_clean_dataset_has_empty_checklist(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_clean"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=CLEAN_FIXTURE_SEED)

    # Confirm the premise directly: this fixture seed genuinely has zero
    # rejected samples (may have flagged samples, which is fine -- flagged
    # is not a rejection).
    validated = validate_dataset(ingest_dataset(raw_dir))
    assert all(s.status != "rejected" for s in validated)

    report = run_intake(raw_dir)

    assert report.rejected == 0
    assert not report.manifest_errors
    assert report.is_clean
    assert report.fix_it_checklist == []


# --- CLI: main() ----------------------------------------------------------


def test_cli_main_exit_code_zero_and_checklist_printed_for_clean_dataset(
    tmp_path: Path, capsys
) -> None:
    raw_dir = tmp_path / "raw_clean"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=CLEAN_FIXTURE_SEED)

    exit_code = main([str(raw_dir)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Fix-it checklist:" in captured.out
    assert "No issues found." in captured.out
    assert "Result: dataset looks usable" in captured.out


def test_cli_main_exit_code_nonzero_for_dataset_with_rejections(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)

    exit_code = main([str(raw_dir)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Fix-it checklist:" in captured.out
    assert "blur" in captured.out.lower()
    assert "Result: dataset is NOT ready" in captured.out


def test_cli_main_exit_code_nonzero_for_broken_manifest(tmp_path: Path, capsys) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    exit_code = main([str(empty_dir)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "missing_manifest" in captured.out
    assert "Result: dataset is NOT ready" in captured.out


def test_cli_main_writes_json_output_file(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=FIXTURE_SEED)
    output_path = tmp_path / "report.json"

    exit_code = main([str(raw_dir), "--output", str(output_path)])

    assert exit_code != 0  # this dataset has rejections
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["total_samples"] == 20
    assert data["rejected"] == 3
    assert data["is_clean"] is False
    assert len(data["rejected_samples"]) == 3
    assert data["fix_it_checklist"]

    captured = capsys.readouterr()
    assert f"Detailed report written to {output_path}" in captured.out


# --- out-of-process: exercises the actual `python -m grafology_ai.intake_cli` ---


def test_cli_help_via_subprocess() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.intake_cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "grafology-intake" in completed.stdout
    assert "dataset_dir" in completed.stdout


def test_cli_subprocess_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_clean"
    generate_fixture_dataset(raw_dir, count=FIXTURE_COUNT, seed=CLEAN_FIXTURE_SEED)

    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.intake_cli", str(raw_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "Fix-it checklist:" in completed.stdout
    assert "No issues found." in completed.stdout
