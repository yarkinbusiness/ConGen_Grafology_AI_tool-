"""Dataset export stage: write a versioned, reproducible snapshot to disk.

This is the "export/versioning" half of the client's technical proposal's
"Pipeline setup (ingest/split/export/versioning)" line item: take a
:class:`~grafology_ai.pipeline.split.DatasetSplit` plus the full set of
:class:`~grafology_ai.pipeline.validate.ValidatedSample` records it was
computed from, and write a self-contained "dataset snapshot" directory
that a later training/reporting stage can point at.

Versioning
----------

Each snapshot is written under ``export_dir / f"v_{content_hash}"``, where
``content_hash`` is a SHA-256 digest (truncated to
:data:`VERSION_HASH_LENGTH` hex characters for a readable directory name)
of everything that determines the snapshot's contents: every sample's
manifest fields, validation results and dataset-level status, split
assignment, and raw image bytes. The hash is computed over samples sorted
by ``sample_id`` and JSON-serialized with sorted keys, so it does not
depend on input list ordering -- only on the underlying data. This makes
the version identifier content-derived rather than a counter or a
timestamp: running the full pipeline twice against byte-identical input
(e.g. two fixture datasets generated with the same seed/count into
different directories) produces the same version identifier and a
byte-identical snapshot, while changing the input (a different fixture
seed, a different image, a different manifest field) changes the
identifier. That is the versioning guarantee this module exists to provide
-- re-running against unchanged data is a safe no-op you can detect by
comparing version strings, and re-running against changed data always
produces a new, distinguishable version.

If ``export_dir / f"v_{content_hash}"`` already exists (e.g. a previous,
identical run already wrote it), it is replaced with a freshly written
copy rather than left as-is or appended to -- since the content hash
guarantees the new write would be byte-identical anyway, this just keeps
the on-disk snapshot's file set exactly in sync with what this run would
produce, with no risk of stale leftover files from a differently-coded
past run under the same hash.

Images: copy-in, not reference
-------------------------------

Every accepted/flagged sample's image file is copied into the snapshot
under ``images/`` (rejected samples' images are not copied -- they are
outside the dataset and copying them would only bloat every snapshot with
excluded data). Copying was chosen over storing a reference back to the
source directory because a snapshot is meant to be a durable, standalone
artifact: the source dataset directory (e.g. a tmp dir from a fixture run,
or a client intake drop) is not guaranteed to still exist, or to still
contain the same bytes, by the time something consumes the snapshot later.
For the dataset sizes this pipeline currently deals with (synthetic
fixtures, and a not-yet-existing real client dataset expected to be of
comparable scale) the extra disk usage from copying is a small, acceptable
cost for that durability; if the dataset grows large enough for copying to
become wasteful, this is the function to revisit.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grafology_ai.dataset.schema import ManifestEntry
from grafology_ai.pipeline.split import DatasetSplit
from grafology_ai.pipeline.validate import ValidatedSample
from grafology_ai.validation.validators import ValidationResult

#: Number of hex characters of the SHA-256 content hash used in the
#: version directory name (``v_<these hex chars>``). 16 hex chars is 64
#: bits of the digest -- short enough to keep directory names readable,
#: while collisions between genuinely different dataset contents remain
#: astronomically unlikely for this pipeline's scale.
VERSION_HASH_LENGTH = 16

_JSON_INDENT = 2


@dataclass(frozen=True)
class ExportResult:
    """The outcome of writing a dataset snapshot.

    Attributes:
        version: The content-derived version identifier, e.g.
            ``"v_3f9a1c2b7e4d5a6b"``.
        snapshot_dir: The directory the snapshot was written to
            (``export_dir / version``).
    """

    version: str
    snapshot_dir: Path


def _validation_result_to_dict(result: ValidationResult) -> dict[str, Any]:
    return {
        "check_name": result.check_name,
        "verdict": result.verdict,
        "reason": result.reason,
        "measured_value": result.measured_value,
    }


def _write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented, deterministically-ordered JSON.

    ``sort_keys=True`` plus every JSON-bound dict in this module using a
    fixed key set means the same logical content always serializes to the
    same bytes, which is what the versioning-determinism guarantee relies
    on.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=_JSON_INDENT, sort_keys=True)
        f.write("\n")


def _compute_content_hash(
    validated_samples: Sequence[ValidatedSample], split: DatasetSplit
) -> str:
    """Compute a SHA-256 hex digest over everything a snapshot's content depends on.

    Iterates ``validated_samples`` sorted by ``sample_id`` (never in
    caller-supplied order) and folds in, per sample: the manifest entry,
    the dataset-level status, every validation result, and a hash of the
    raw image bytes. Also folds in the split assignment (train/val/test)
    for every entry, again sorted by ``sample_id``. Because every input to
    the hash is either sorted or itself content (never a path string or
    anything else that could differ between two directories holding
    identical data), two independently-generated but content-identical
    datasets hash to the same value.
    """
    hasher = hashlib.sha256()

    for split_name, split_entries in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        hasher.update(f"split:{split_name}".encode("utf-8"))
        for entry in sorted(split_entries, key=lambda e: e.sample_id):
            hasher.update(
                json.dumps(entry.to_dict(), sort_keys=True).encode("utf-8")
            )

    for sample in sorted(validated_samples, key=lambda s: s.entry.sample_id):
        hasher.update(b"sample:")
        hasher.update(json.dumps(sample.entry.to_dict(), sort_keys=True).encode("utf-8"))
        hasher.update(sample.status.encode("utf-8"))
        for result in sample.results:
            hasher.update(
                json.dumps(_validation_result_to_dict(result), sort_keys=True).encode(
                    "utf-8"
                )
            )
        hasher.update(hashlib.sha256(Path(sample.image_path).read_bytes()).digest())

    return hasher.hexdigest()


def _entries_to_manifest_json(entries: Sequence[ManifestEntry]) -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in sorted(entries, key=lambda e: e.sample_id)]


def _rejected_to_manifest_json(
    rejected: Sequence[ValidatedSample],
) -> list[dict[str, Any]]:
    entries = []
    for sample in sorted(rejected, key=lambda s: s.entry.sample_id):
        entry_dict = sample.entry.to_dict()
        entry_dict["rejection_reasons"] = [
            _validation_result_to_dict(result) for result in sample.rejection_results
        ]
        entries.append(entry_dict)
    return entries


def _build_validation_report(
    validated_samples: Sequence[ValidatedSample],
) -> dict[str, Any]:
    """Build the summary object written to ``validation_report.json``."""
    status_counts = {"accepted": 0, "flagged": 0, "rejected": 0}
    check_verdict_counts: dict[str, dict[str, int]] = {}
    samples_report = []

    for sample in sorted(validated_samples, key=lambda s: s.entry.sample_id):
        status_counts[sample.status] += 1
        for result in sample.results:
            per_check = check_verdict_counts.setdefault(
                result.check_name, {"accept": 0, "reject": 0, "flag": 0}
            )
            per_check[result.verdict] += 1

        samples_report.append(
            {
                "sample_id": sample.entry.sample_id,
                "status": sample.status,
                "results": [_validation_result_to_dict(r) for r in sample.results],
            }
        )

    return {
        "summary": {
            "total_samples": len(validated_samples),
            "accepted": status_counts["accepted"],
            "flagged": status_counts["flagged"],
            "rejected": status_counts["rejected"],
            "check_verdict_counts": check_verdict_counts,
        },
        "samples": samples_report,
    }


def export_snapshot(
    export_dir: Path,
    validated_samples: Sequence[ValidatedSample],
    split: DatasetSplit,
) -> ExportResult:
    """Write a versioned dataset snapshot to ``export_dir``.

    See the module docstring for the versioning scheme and the choice to
    copy image files into the snapshot. Writes, under
    ``export_dir / f"v_{content_hash}"``:

    - ``train_manifest.json``, ``val_manifest.json``, ``test_manifest.json``:
      each a JSON array of ``entry.to_dict()`` for that split, sorted by
      ``sample_id``.
    - ``rejected_manifest.json``: a JSON array of rejected entries'
      ``entry.to_dict()``, each with an added ``"rejection_reasons"`` key
      listing the ``"reject"``-verdict checks that excluded the sample.
    - ``validation_report.json``: accept/flag/reject counts overall and
      per check, plus every sample's full per-check results.
    - ``images/{sample_id}{original suffix}``: a copy of the image file
      for every sample in the train/val/test splits.

    Args:
        export_dir: Parent directory snapshots are written under. Created
            if it does not already exist.
        validated_samples: Every validated sample from
            :func:`grafology_ai.pipeline.validate.validate_dataset`
            (accepted, flagged, *and* rejected) -- used to write
            ``rejected_manifest.json`` and ``validation_report.json``.
        split: The train/val/test split computed from the accepted
            (non-rejected) subset of ``validated_samples`` via
            :func:`grafology_ai.pipeline.split.split_dataset`.

    Returns:
        An :class:`ExportResult` with the version identifier and the
        snapshot directory path.
    """
    export_dir = Path(export_dir)
    content_hash = _compute_content_hash(validated_samples, split)
    version = f"v_{content_hash[:VERSION_HASH_LENGTH]}"
    snapshot_dir = export_dir / version

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    images_dir = snapshot_dir / "images"
    images_dir.mkdir(parents=True)

    _write_json(snapshot_dir / "train_manifest.json", _entries_to_manifest_json(split.train))
    _write_json(snapshot_dir / "val_manifest.json", _entries_to_manifest_json(split.val))
    _write_json(snapshot_dir / "test_manifest.json", _entries_to_manifest_json(split.test))

    rejected = [s for s in validated_samples if s.status == "rejected"]
    _write_json(snapshot_dir / "rejected_manifest.json", _rejected_to_manifest_json(rejected))

    _write_json(
        snapshot_dir / "validation_report.json",
        _build_validation_report(validated_samples),
    )

    by_id = {sample.entry.sample_id: sample for sample in validated_samples}
    included_entries = sorted(
        [*split.train, *split.val, *split.test], key=lambda e: e.sample_id
    )
    for entry in included_entries:
        source = Path(by_id[entry.sample_id].image_path)
        shutil.copyfile(source, images_dir / f"{entry.sample_id}{source.suffix}")

    return ExportResult(version=version, snapshot_dir=snapshot_dir)
