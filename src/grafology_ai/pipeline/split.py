"""Dataset split stage: a deterministic, reproducible train/val/test split.

This is the "split" half of the client's technical proposal's "Pipeline
setup (ingest/split/export/versioning)" line item.

The split is computed over ``sample_id`` rather than list order: the set
of accepted sample_ids is sorted (for a canonical starting order that does
not depend on how the caller happened to assemble the input list), then
shuffled with a :class:`random.Random` seeded from ``seed`` and sliced
according to the requested ratios. Because the starting order is always
the sort order, two calls with the same accepted-sample *set* and the same
``seed`` produce the same split membership even if the two input lists
were built in different orders (e.g. ingested from manifests with entries
in a different order, or filtered differently upstream) -- which is what
"reproducible" needs to mean here, not just "stable if you don't touch
anything."
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from grafology_ai.dataset.schema import ManifestEntry

#: Default split proportions: 70% train, 15% val, 15% test.
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

_RATIO_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DatasetSplit:
    """A train/val/test partition of a set of manifest entries.

    Each list is sorted by ``sample_id`` (an arbitrary but deterministic
    order, chosen so re-running the split -- or exporting it -- produces
    byte-identical output). The three lists are disjoint and together
    cover every entry passed to :func:`split_dataset` exactly once.

    Attributes:
        train: Entries assigned to the training split.
        val: Entries assigned to the validation split.
        test: Entries assigned to the test split.
    """

    train: list[ManifestEntry] = field(default_factory=list)
    val: list[ManifestEntry] = field(default_factory=list)
    test: list[ManifestEntry] = field(default_factory=list)


def split_dataset(
    entries: Sequence[ManifestEntry],
    seed: int = 0,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> DatasetSplit:
    """Deterministically split ``entries`` into train/val/test sets.

    Intended to be called with the *accepted* (non-rejected) samples from
    :func:`grafology_ai.pipeline.validate.validate_dataset` /
    :func:`~grafology_ai.pipeline.validate.partition_by_status` --
    rejected samples should never reach this function.

    Determinism and reproducibility:

    - The split depends only on the *set* of ``entry.sample_id`` values
      and on ``seed`` (and the ratios) -- not on the order of ``entries``.
      ``sample_id``\\ s are sorted first, then shuffled with
      ``random.Random(seed)``, so the same set of ids with the same seed
      always yields the same shuffle, regardless of input ordering.
    - Calling this twice with the same input and seed produces identical
      train/val/test membership (same sample_ids in each split).
    - The three returned lists are disjoint and their union is exactly
      ``entries`` (by ``sample_id``) -- every accepted sample lands in
      exactly one split.

    Args:
        entries: Accepted manifest entries to split. Must have unique
            ``sample_id`` values.
        seed: Seed for the deterministic shuffle.
        train_ratio: Target fraction assigned to the training split.
        val_ratio: Target fraction assigned to the validation split.
        test_ratio: Target fraction assigned to the test split; also
            absorbs any rounding remainder from ``train_ratio``/
            ``val_ratio``, so the three splits always cover every entry
            exactly once.

    Returns:
        A :class:`DatasetSplit` with entries sorted by ``sample_id``
        within each split.

    Raises:
        ValueError: If ``train_ratio + val_ratio + test_ratio`` is not 1
            (within floating-point tolerance), if any ratio is negative,
            or if ``entries`` contains a duplicate ``sample_id``.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > _RATIO_SUM_TOLERANCE:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got "
            f"{ratio_sum!r} ({train_ratio!r} + {val_ratio!r} + {test_ratio!r})"
        )
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f"ratios must be non-negative, got train={train_ratio!r}, "
            f"val={val_ratio!r}, test={test_ratio!r}"
        )

    by_id: dict[str, ManifestEntry] = {}
    for entry in entries:
        if entry.sample_id in by_id:
            raise ValueError(f"duplicate sample_id {entry.sample_id!r} in entries")
        by_id[entry.sample_id] = entry

    shuffled_ids = sorted(by_id)
    random.Random(seed).shuffle(shuffled_ids)

    total = len(shuffled_ids)
    n_train = round(total * train_ratio)
    n_train = min(n_train, total)
    n_val = round(total * val_ratio)
    n_val = min(n_val, total - n_train)
    # test takes whatever remains, absorbing any rounding slack so the
    # three splits always sum to `total` exactly.

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    return DatasetSplit(
        train=[by_id[sample_id] for sample_id in sorted(train_ids)],
        val=[by_id[sample_id] for sample_id in sorted(val_ids)],
        test=[by_id[sample_id] for sample_id in sorted(test_ids)],
    )
