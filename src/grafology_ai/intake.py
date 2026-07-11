"""Day-zero dataset intake: "is this dataset usable?" before training work starts.

No real, labeled client handwriting dataset exists yet. When it eventually
arrives, whoever receives it needs a fast, trustworthy way to answer "is
this dataset usable?" before any training work starts -- not to label it,
and not to collect more of it (both are explicitly the client's
responsibility, per the technical proposal's "What Is NOT Included"
section), just to check it.

:func:`run_intake` is that check. It is a thin orchestration layer over the
existing dataset pipeline stages -- :func:`grafology_ai.pipeline.ingest.ingest_dataset`
and :func:`grafology_ai.pipeline.validate.validate_dataset` -- that turns
their output into a human-actionable :class:`IntakeReport`: total/accepted/
flagged/rejected counts, exactly which check(s) failed for each rejected
sample, and a "fix-it checklist" of aggregated, human-readable findings
(e.g. "3 samples are below the minimum resolution -- recapture or rescan
these at higher quality"). It deliberately reimplements none of the
image-quality-check logic itself (that stays in
:mod:`grafology_ai.validation.validators`, via ``validate_dataset``).

Manifest-level problems (a missing ``manifest.json``, invalid JSON, a
manifest entry that fails schema validation, a sample referencing a
missing image file) are common for a first-ever drop of a real dataset,
and are exactly the kind of thing this tool exists to catch early. Rather
than let ``ingest_dataset``'s first raised exception crash the whole intake
run, :func:`run_intake` catches it and, on the error path only, does a
lightweight structural re-scan of the manifest (schema validation via the
real :class:`~grafology_ai.dataset.schema.ManifestEntry`, and image-file
existence via the same ``{sample_id}.*`` glob ``ingest_dataset`` itself
uses) so the report can list *every* structural problem found, each
labeled with a distinguishable root-cause type, in one pass -- instead of
the fail-fast, fix-one-error-at-a-time experience calling ``ingest_dataset``
directly gives you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grafology_ai.dataset.schema import ManifestEntry
from grafology_ai.pipeline.ingest import ingest_dataset
from grafology_ai.pipeline.validate import ValidatedSample, validate_dataset

#: Human-readable fix-it messages for each automated check, keyed by
#: `ValidationResult.check_name` (see `grafology_ai.validation.validators`).
#: `{count}`/`{plural}` are filled in by `_aggregate_checklist`.
_CHECK_REJECT_TEMPLATES: dict[str, str] = {
    "resolution": (
        "{count} {plural} below the minimum resolution -- recapture or "
        "rescan these at higher quality."
    ),
    "blur": (
        "{count} {plural} too blurry to use -- recapture with a steadier "
        "hand or better camera focus."
    ),
    "contrast": (
        "{count} {plural} too low-contrast (faint ink or a washed-out "
        "scan) -- rescan with better lighting or contrast."
    ),
    "format": (
        "{count} {plural} in an unsupported image format -- convert to "
        "JPEG or PNG."
    ),
}


def _sample_plural(count: int) -> str:
    return "sample is" if count == 1 else "samples are"


@dataclass(frozen=True)
class ManifestError:
    """A single manifest/ingest-level problem found before validation could run.

    Attributes:
        error_type: A short, stable machine-readable category, e.g.
            ``"missing_manifest"``, ``"invalid_json"``,
            ``"invalid_manifest_structure"``, ``"schema_validation_failed"``,
            ``"missing_image_file"``, ``"ambiguous_image_file"``,
            ``"missing_images_directory"``, ``"unreadable_image"``, or
            ``"unknown_error"`` -- always distinguishable by root cause.
        message: Human-readable explanation.
        sample_id: The manifest entry this error is about, if known (some
            error types, like a missing ``manifest.json``, are not tied to
            any single entry).
    """

    error_type: str
    message: str
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True)
class RejectedSampleDetail:
    """Which check(s) rejected one sample, and why.

    Attributes:
        sample_id: The rejected sample's identifier.
        failed_checks: One entry per check with a ``"reject"`` verdict for
            this sample (``check_name``, ``reason``, ``measured_value`` --
            mirrors `grafology_ai.validation.validators.ValidationResult`).
    """

    sample_id: str
    failed_checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "failed_checks": self.failed_checks}


@dataclass(frozen=True)
class IntakeReport:
    """Summary of one dataset-intake run: is this dataset usable, and what to fix.

    Attributes:
        dataset_dir: The dataset directory this report is about.
        total_samples: Total samples listed in the manifest (0 if the
            manifest itself could not be read -- see ``manifest_errors``).
        accepted: Count of samples with status ``"accepted"``.
        flagged: Count of samples with status ``"flagged"``.
        rejected: Count of samples with status ``"rejected"``.
        rejected_samples: Per-rejected-sample detail of which check(s)
            failed and why.
        manifest_errors: Manifest/ingest-level problems that stopped
            validation from running at all (empty in the normal case).
        fix_it_checklist: Human-readable, actionable strings describing
            what's wrong across the dataset in aggregate. Empty (not full
            of "0 issues" boilerplate) when there is nothing to report.
    """

    dataset_dir: Path
    total_samples: int
    accepted: int
    flagged: int
    rejected: int
    rejected_samples: list[RejectedSampleDetail] = field(default_factory=list)
    manifest_errors: list[ManifestError] = field(default_factory=list)
    fix_it_checklist: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if the dataset has zero manifest-level errors and zero rejected samples.

        This is the pass/fail signal `grafology-intake`'s exit code is
        based on -- a dataset can still have flagged samples (allowed
        through for manual review) and be considered "clean" for intake
        purposes.
        """
        return not self.manifest_errors and self.rejected == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report to a plain JSON-compatible dict."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "total_samples": self.total_samples,
            "accepted": self.accepted,
            "flagged": self.flagged,
            "rejected": self.rejected,
            "is_clean": self.is_clean,
            "rejected_samples": [s.to_dict() for s in self.rejected_samples],
            "manifest_errors": [e.to_dict() for e in self.manifest_errors],
            "fix_it_checklist": list(self.fix_it_checklist),
        }


def _diagnose_broken_manifest(
    dataset_dir: Path, original_exc: Exception
) -> list[ManifestError]:
    """Re-scan a dataset directory that `ingest_dataset` failed on.

    `ingest_dataset` fails fast (raises on the first structural problem it
    finds), which is the right behavior for a pipeline that needs to
    ingest a *usable* dataset, but the wrong behavior for an intake report
    that should tell the user everything wrong in one pass. This function
    re-does the same structural checks `ingest_dataset` does --
    `manifest.json` presence/JSON-validity/array-shape, each entry's
    `ManifestEntry.from_dict` schema validation, and each entry's image
    file resolution via the same `{sample_id}.*` glob -- but collects every
    problem found instead of stopping at the first one.

    Only called on `ingest_dataset`'s error path, and never used to
    actually process a dataset (no `ValidatedSample`s come out of this) --
    it exists purely to produce a richer diagnostic than the single
    exception `ingest_dataset` raised.
    """
    manifest_path = dataset_dir / "manifest.json"
    images_dir = dataset_dir / "images"

    if not manifest_path.is_file():
        return [
            ManifestError(
                error_type="missing_manifest",
                message=f"No manifest.json found at {manifest_path}.",
            )
        ]

    try:
        with manifest_path.open(encoding="utf-8") as f:
            raw_entries = json.load(f)
    except json.JSONDecodeError as exc:
        return [
            ManifestError(
                error_type="invalid_json",
                message=f"{manifest_path} is not valid JSON: {exc}",
            )
        ]

    if not isinstance(raw_entries, list):
        return [
            ManifestError(
                error_type="invalid_manifest_structure",
                message=(
                    f"{manifest_path} must contain a JSON array of manifest "
                    f"entries, got {type(raw_entries).__name__}."
                ),
            )
        ]

    if not images_dir.is_dir():
        return [
            ManifestError(
                error_type="missing_images_directory",
                message=f"No images/ directory found at {images_dir}.",
            )
        ]

    errors: list[ManifestError] = []
    for index, raw_entry in enumerate(raw_entries):
        declared_id = raw_entry.get("sample_id") if isinstance(raw_entry, dict) else None
        label = declared_id if declared_id else f"entry #{index}"

        try:
            entry = ManifestEntry.from_dict(raw_entry)
        except ValueError as exc:
            errors.append(
                ManifestError(
                    error_type="schema_validation_failed",
                    message=f"manifest entry {label!r} failed schema validation: {exc}",
                    sample_id=declared_id,
                )
            )
            continue

        candidates = sorted(images_dir.glob(f"{entry.sample_id}.*"))
        if not candidates:
            errors.append(
                ManifestError(
                    error_type="missing_image_file",
                    message=(
                        f"manifest entry {entry.sample_id!r} references an image "
                        f"under {images_dir}, but no file named "
                        f"{entry.sample_id}.* was found there."
                    ),
                    sample_id=entry.sample_id,
                )
            )
        elif len(candidates) > 1:
            names = [candidate.name for candidate in candidates]
            errors.append(
                ManifestError(
                    error_type="ambiguous_image_file",
                    message=(
                        f"sample_id {entry.sample_id!r} matches multiple image "
                        f"files under {images_dir}: {names}."
                    ),
                    sample_id=entry.sample_id,
                )
            )

    if errors:
        return errors

    # The re-scan above found nothing wrong (e.g. the original failure was
    # a duplicate sample_id across two otherwise-valid entries, or some
    # other condition this function does not specifically re-check) --
    # fall back to surfacing ingest_dataset's own exception verbatim rather
    # than silently reporting no issues.
    return [ManifestError(error_type="unknown_error", message=str(original_exc))]


def _aggregate_checklist(validated: list[ValidatedSample]) -> list[str]:
    """Build the "fix-it checklist": aggregate, human-readable findings.

    Only covers *blocking* problems (rejected samples' failed checks, plus
    manifest metadata gaps like missing language) -- not flagged samples,
    which are already included in the dataset and only need manual review,
    not a fix. Flagged counts are still visible in `IntakeReport.flagged`
    and per-sample results; they are deliberately excluded from this
    checklist so "zero rejected samples" reliably means "empty checklist",
    matching what an all-clear intake run should look like.

    Only mentions categories with at least one real occurrence -- no "0
    issues in category X" boilerplate.
    """
    reject_counts: dict[str, int] = {}

    for sample in validated:
        reject_checks = {result.check_name for result in sample.rejection_results}
        for check_name in reject_checks:
            reject_counts[check_name] = reject_counts.get(check_name, 0) + 1

    checklist: list[str] = []

    for check_name in sorted(reject_counts):
        count = reject_counts[check_name]
        template = _CHECK_REJECT_TEMPLATES.get(
            check_name,
            "{count} {plural} failing the '" + check_name + "' check.",
        )
        checklist.append(template.format(count=count, plural=_sample_plural(count)))

    missing_language = sum(
        1 for sample in validated if not (sample.entry.language or "").strip()
    )
    if missing_language:
        checklist.append(
            f"{missing_language} {_sample_plural(missing_language)} missing "
            "language metadata in the manifest -- fill in the 'language' "
            "field if known."
        )

    return checklist


def run_intake(dataset_dir: Path) -> IntakeReport:
    """Check a raw dataset directory for day-zero usability.

    Reuses the existing pipeline stages end to end: `ingest_dataset` reads
    `dataset_dir/manifest.json` + `dataset_dir/images/`, then
    `validate_dataset` runs the same automated image-quality checks
    (`grafology_ai.validation.validators.validate_sample`) every other
    pipeline entry point uses, and classifies each sample as accepted,
    flagged, or rejected exactly as `grafology_ai.pipeline.validate` does.
    No validation logic is reimplemented here.

    Manifest/ingest-level problems (missing `manifest.json`, invalid JSON,
    a manifest entry that fails schema validation, a missing image file)
    do not raise -- they are caught and reported as `IntakeReport.manifest_errors`,
    with `total_samples`/`accepted`/`flagged`/`rejected` all `0` in that
    case. An image file that exists but cannot be opened as an image
    (corrupt file) is handled the same way, as an `"unreadable_image"`
    manifest error, rather than crashing the run.

    Args:
        dataset_dir: Directory containing `manifest.json` and `images/`
            (the same layout `generate_fixture_dataset` produces).

    Returns:
        An `IntakeReport` summarizing the dataset's usability. Never
        raises for a malformed dataset -- see above.
    """
    dataset_dir = Path(dataset_dir)

    try:
        ingested = ingest_dataset(dataset_dir)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any ingest
        # failure becomes a reported manifest error, never a crash.
        manifest_errors = _diagnose_broken_manifest(dataset_dir, exc)
        return IntakeReport(
            dataset_dir=dataset_dir,
            total_samples=0,
            accepted=0,
            flagged=0,
            rejected=0,
            manifest_errors=manifest_errors,
            fix_it_checklist=[
                f"[{error.error_type}] {error.message}" for error in manifest_errors
            ],
        )

    try:
        validated = validate_dataset(ingested)
    except Exception as exc:  # noqa: BLE001 - see comment above; e.g. a
        # sample's image file exists but is not a readable image.
        manifest_error = ManifestError(error_type="unreadable_image", message=str(exc))
        return IntakeReport(
            dataset_dir=dataset_dir,
            total_samples=len(ingested),
            accepted=0,
            flagged=0,
            rejected=0,
            manifest_errors=[manifest_error],
            fix_it_checklist=[f"[{manifest_error.error_type}] {manifest_error.message}"],
        )

    accepted = sum(1 for sample in validated if sample.status == "accepted")
    flagged = sum(1 for sample in validated if sample.status == "flagged")
    rejected_samples_full = [sample for sample in validated if sample.status == "rejected"]

    rejected_samples = [
        RejectedSampleDetail(
            sample_id=sample.entry.sample_id,
            failed_checks=[
                {
                    "check_name": result.check_name,
                    "reason": result.reason,
                    "measured_value": result.measured_value,
                }
                for result in sample.rejection_results
            ],
        )
        for sample in rejected_samples_full
    ]

    return IntakeReport(
        dataset_dir=dataset_dir,
        total_samples=len(validated),
        accepted=accepted,
        flagged=flagged,
        rejected=len(rejected_samples_full),
        rejected_samples=rejected_samples,
        manifest_errors=[],
        fix_it_checklist=_aggregate_checklist(validated),
    )
