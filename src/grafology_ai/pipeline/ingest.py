"""Dataset ingest: read a raw dataset directory into in-memory samples.

This is the "ingest" half of the client's technical proposal's "Pipeline
setup (ingest/split/export/versioning)" line item. It reads the
``manifest.json`` + ``images/`` directory layout produced by
:func:`grafology_ai.dataset.fixtures.generate_fixture_dataset` (and, in
production, expected of the client's raw intake once it exists), and
resolves each manifest entry to the image file it describes.

Ingest deliberately does no image-quality judgment of its own -- that is
:mod:`grafology_ai.pipeline.validate`'s job, one stage later. This module's
only job is: does every manifest entry have a corresponding, readable image
file on disk? If not, fail loudly and early, before any downstream stage
has a chance to silently skip a sample or crash with a confusing low-level
error deep in image-processing code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from grafology_ai.dataset.schema import ManifestEntry


@dataclass(frozen=True)
class IngestedSample:
    """One manifest entry paired with the resolved path to its image file.

    Attributes:
        entry: The sample's manifest entry.
        image_path: Path to the sample's image file on disk, resolved from
            ``entry.sample_id`` against the dataset's ``images/`` directory.
            Guaranteed to exist at the time ingest ran.
    """

    entry: ManifestEntry
    image_path: Path


def _resolve_image_path(images_dir: Path, sample_id: str) -> Path:
    """Find the image file for ``sample_id`` under ``images_dir``.

    Manifest entries do not record a file extension, so the image is
    located by globbing ``images_dir`` for ``{sample_id}.*`` rather than
    assuming a fixed extension (e.g. PNG) -- this keeps ingest working for
    any supported image format, not just the one the fixture generator
    happens to emit.

    Raises:
        FileNotFoundError: If no file matches ``sample_id`` in
            ``images_dir``.
        ValueError: If more than one file matches ``sample_id`` (an
            unresolvable ambiguity -- which file is "the" image?).
    """
    candidates = sorted(images_dir.glob(f"{sample_id}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"manifest entry {sample_id!r} references an image under "
            f"{images_dir}, but no file named {sample_id}.* was found there"
        )
    if len(candidates) > 1:
        names = [candidate.name for candidate in candidates]
        raise ValueError(
            f"ambiguous image for sample_id {sample_id!r} under "
            f"{images_dir}: multiple candidate files found: {names}"
        )
    return candidates[0]


def ingest_dataset(input_dir: Path) -> list[IngestedSample]:
    """Read a raw dataset directory and resolve every entry's image path.

    Expects the same layout :func:`generate_fixture_dataset` produces (and
    the client's eventual raw intake is expected to match): a
    ``manifest.json`` file at ``input_dir / "manifest.json"`` containing a
    JSON array of :meth:`ManifestEntry.to_dict` records, and the
    corresponding image files under ``input_dir / "images"``.

    Args:
        input_dir: Root of the raw dataset directory.

    Returns:
        One :class:`IngestedSample` per manifest entry, in manifest order.

    Raises:
        FileNotFoundError: If ``manifest.json`` does not exist, or if a
            manifest entry references an image file that does not exist.
        ValueError: If ``manifest.json`` is not a JSON array, if an entry
            fails :meth:`ManifestEntry.from_dict` validation, or if an
            entry's ``sample_id`` matches more than one file under
            ``images/``.
    """
    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    images_dir = input_dir / "images"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no manifest.json found at {manifest_path} -- expected the "
            "same layout generate_fixture_dataset() produces"
        )

    with manifest_path.open(encoding="utf-8") as f:
        raw_entries = json.load(f)

    if not isinstance(raw_entries, list):
        raise ValueError(
            f"{manifest_path} must contain a JSON array of manifest "
            f"entries, got {type(raw_entries).__name__}"
        )

    samples: list[IngestedSample] = []
    for raw_entry in raw_entries:
        entry = ManifestEntry.from_dict(raw_entry)
        image_path = _resolve_image_path(images_dir, entry.sample_id)
        samples.append(IngestedSample(entry=entry, image_path=image_path))

    return samples
