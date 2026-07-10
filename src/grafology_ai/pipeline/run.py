"""End-to-end pipeline: ingest -> validate -> split -> export.

A thin convenience wrapper chaining the four pipeline stages
(:mod:`grafology_ai.pipeline.ingest`, :mod:`grafology_ai.pipeline.validate`,
:mod:`grafology_ai.pipeline.split`, :mod:`grafology_ai.pipeline.export`) so
that the common "run the whole pipeline" case -- what
``docs/pipeline_runbook.md`` documents and what most tests exercise -- is a
single call. Each stage remains independently usable for callers that need
to inspect or alter intermediate results (e.g. custom split ratios, or
inspecting rejected samples before deciding whether to export).
"""

from __future__ import annotations

from pathlib import Path

from grafology_ai.pipeline.export import ExportResult, export_snapshot
from grafology_ai.pipeline.ingest import ingest_dataset
from grafology_ai.pipeline.split import (
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    split_dataset,
)
from grafology_ai.pipeline.validate import partition_by_status, validate_dataset


def run_pipeline(
    input_dir: Path,
    export_dir: Path,
    seed: int = 0,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> ExportResult:
    """Run ingest -> validate -> split -> export against ``input_dir``.

    Args:
        input_dir: Raw dataset directory (``manifest.json`` + ``images/``),
            e.g. one written by
            :func:`grafology_ai.dataset.fixtures.generate_fixture_dataset`.
        export_dir: Parent directory the versioned snapshot is written
            under (see :func:`~grafology_ai.pipeline.export.export_snapshot`).
        seed: Seed for the deterministic train/val/test split.
        train_ratio: Target training-split fraction.
        val_ratio: Target validation-split fraction.
        test_ratio: Target test-split fraction.

    Returns:
        The :class:`~grafology_ai.pipeline.export.ExportResult` describing
        the written snapshot.
    """
    ingested = ingest_dataset(Path(input_dir))
    validated = validate_dataset(ingested)
    included, _rejected = partition_by_status(validated)

    split = split_dataset(
        [sample.entry for sample in included],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    return export_snapshot(Path(export_dir), validated, split)
