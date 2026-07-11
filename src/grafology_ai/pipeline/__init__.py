"""Dataset pipeline: ingest, validate, split, and versioned export.

This subpackage implements the client's technical proposal's "Pipeline
setup (ingest/split/export/versioning)" line item, tying together the
dataset manifest schema (:mod:`grafology_ai.dataset.schema`), the synthetic
fixture generator (:mod:`grafology_ai.dataset.fixtures`), and the automated
image-quality validators (:mod:`grafology_ai.validation.validators`) into
one deterministic, reproducible flow:

1. :mod:`grafology_ai.pipeline.ingest` -- read a raw dataset directory
   (``manifest.json`` + ``images/``) into in-memory samples with resolved
   image paths.
2. :mod:`grafology_ai.pipeline.validate` -- run automated quality checks
   over every ingested sample and classify each as accepted, flagged (for
   manual review, but still included), or rejected.
3. :mod:`grafology_ai.pipeline.split` -- deterministically split the
   accepted samples into train/val/test sets by ``sample_id``.
4. :mod:`grafology_ai.pipeline.export` -- write a versioned snapshot
   directory whose version identifier is derived from the snapshot's
   content, so re-running against unchanged data reproduces the same
   version and byte-identical output.

:func:`grafology_ai.pipeline.run.run_pipeline` chains all four stages for
the common end-to-end case. See ``docs/pipeline_runbook.md`` for the
operational how-to.
"""

from grafology_ai.pipeline.export import ExportResult, export_snapshot
from grafology_ai.pipeline.ingest import IngestedSample, ingest_dataset
from grafology_ai.pipeline.run import run_pipeline
from grafology_ai.pipeline.split import (
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DatasetSplit,
    split_dataset,
)
from grafology_ai.pipeline.validate import (
    SampleStatus,
    ValidatedSample,
    partition_by_status,
    validate_dataset,
)

__all__ = [
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DatasetSplit",
    "ExportResult",
    "IngestedSample",
    "SampleStatus",
    "ValidatedSample",
    "export_snapshot",
    "ingest_dataset",
    "partition_by_status",
    "run_pipeline",
    "split_dataset",
    "validate_dataset",
]
