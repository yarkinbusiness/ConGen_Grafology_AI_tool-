"""Dataset validate stage: run image-quality checks over ingested samples.

This wraps :func:`grafology_ai.validation.validators.validate_sample`
(which already implements the client's automated per-check accept/reject/
flag rules -- see that module's docstring) at the dataset level: run it over
every :class:`~grafology_ai.pipeline.ingest.IngestedSample`, and turn each
sample's list of :class:`~grafology_ai.validation.validators.ValidationResult`
into a single dataset-level disposition.

Per the client's technical proposal, a "reject" verdict on any individual
check excludes the sample from the pipeline entirely (it does not get
split into train/val/test). A "flag" verdict does *not* exclude the
sample -- it is recorded for manual review but still flows through to the
splits, since "flag" already means "downgraded from reject" (see
``validate_sample``'s ``quality_label="low"`` downgrade rule) or otherwise
still-usable-but-notable. A sample with only "accept" verdicts is a clean
pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from grafology_ai.dataset.schema import ManifestEntry
from grafology_ai.pipeline.ingest import IngestedSample
from grafology_ai.validation.validators import ValidationResult, validate_sample

#: Dataset-level disposition for one sample, derived from its per-check
#: :class:`ValidationResult` verdicts:
#:
#: - ``"rejected"``: at least one check verdict is ``"reject"``. Excluded
#:   from the train/val/test splits.
#: - ``"flagged"``: no check rejected the sample, but at least one check
#:   verdict is ``"flag"``. Included in the splits, but recorded as
#:   needing manual review.
#: - ``"accepted"``: every check verdict is ``"accept"``. Included in the
#:   splits, no review flag.
SampleStatus = Literal["accepted", "flagged", "rejected"]


@dataclass(frozen=True)
class ValidatedSample:
    """The validation outcome for one ingested sample.

    Attributes:
        entry: The sample's manifest entry.
        image_path: Resolved path to the sample's image file.
        results: Every :class:`ValidationResult` produced by
            :func:`validate_sample` for this image, one per check.
        status: The dataset-level disposition derived from ``results`` --
            see :data:`SampleStatus`.
    """

    entry: ManifestEntry
    image_path: Path
    results: list[ValidationResult]
    status: SampleStatus

    @property
    def is_included(self) -> bool:
        """Whether this sample belongs in the train/val/test splits.

        True for ``"accepted"`` and ``"flagged"`` samples; false for
        ``"rejected"`` samples.
        """
        return self.status != "rejected"

    @property
    def rejection_results(self) -> list[ValidationResult]:
        """The subset of ``results`` with verdict ``"reject"``."""
        return [result for result in self.results if result.verdict == "reject"]

    @property
    def flag_results(self) -> list[ValidationResult]:
        """The subset of ``results`` with verdict ``"flag"``."""
        return [result for result in self.results if result.verdict == "flag"]


def _status_from_results(results: Sequence[ValidationResult]) -> SampleStatus:
    verdicts = {result.verdict for result in results}
    if "reject" in verdicts:
        return "rejected"
    if "flag" in verdicts:
        return "flagged"
    return "accepted"


def validate_dataset(samples: Sequence[IngestedSample]) -> list[ValidatedSample]:
    """Run :func:`validate_sample` over every ingested sample.

    Each sample's manifest ``quality`` is passed through as
    ``validate_sample``'s ``quality_label``, so the manifest-declared
    ``quality="low"`` downgrade rule (reject -> flag for blur/contrast)
    applies exactly as it does when calling ``validate_sample`` directly.

    Args:
        samples: Ingested samples to validate, as returned by
            :func:`grafology_ai.pipeline.ingest.ingest_dataset`.

    Returns:
        One :class:`ValidatedSample` per input sample, in the same order.
    """
    validated: list[ValidatedSample] = []
    for sample in samples:
        results = validate_sample(sample.image_path, quality_label=sample.entry.quality)
        validated.append(
            ValidatedSample(
                entry=sample.entry,
                image_path=sample.image_path,
                results=results,
                status=_status_from_results(results),
            )
        )
    return validated


def partition_by_status(
    validated_samples: Sequence[ValidatedSample],
) -> tuple[list[ValidatedSample], list[ValidatedSample]]:
    """Split validated samples into (included, rejected).

    "Included" is every sample with status ``"accepted"`` or
    ``"flagged"`` (i.e. :attr:`ValidatedSample.is_included`); "rejected"
    is every sample with status ``"rejected"``. Every input sample appears
    in exactly one of the two returned lists, in their original relative
    order.
    """
    included = [sample for sample in validated_samples if sample.is_included]
    rejected = [sample for sample in validated_samples if not sample.is_included]
    return included, rejected
