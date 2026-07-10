"""Dataset manifest schema for graphological handwriting samples.

A dataset manifest is a JSON document listing every handwriting sample in
the dataset along with the metadata the client's technical proposal
requires for each sample:

- an anonymous sample identifier;
- how the sample was acquired (photo or scan);
- a subjective quality rating (high, medium, low);
- the language the sample was written in (when known);
- ground-truth ("gold") labels supplied by professional graphologists,
  keyed by indicator name (e.g. "slant", "pressure") with a label/value
  string, which may be empty for samples that are not yet labeled.

This module defines the in-memory representation of a single manifest
entry (:class:`ManifestEntry`) using the stdlib ``dataclasses`` module, plus
plain-dict (de)serialization so entries can be read from and written to
JSON manifest files. It intentionally adds no new runtime dependency.

The companion JSON Schema describing the same structure, for use by
external tooling, lives at ``manifest.schema.json`` in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

AcquisitionMethod = Literal["photo", "scan"]
Quality = Literal["high", "medium", "low"]

VALID_ACQUISITION_METHODS: tuple[str, ...] = get_args(AcquisitionMethod)
VALID_QUALITIES: tuple[str, ...] = get_args(Quality)


@dataclass(frozen=True)
class ManifestEntry:
    """A single sample entry in a dataset manifest.

    Attributes:
        sample_id: Anonymous identifier for the sample. Must not contain
            any personally identifying information.
        acquisition_method: How the sample was captured — ``"photo"`` for
            a photograph of a handwriting sample, or ``"scan"`` for a
            flatbed/document scan.
        quality: Subjective image-quality rating: ``"high"``, ``"medium"``,
            or ``"low"``.
        language: The language the handwriting sample was written in
            (e.g. ``"it"``, ``"en"``), or ``None`` if unknown/not recorded.
        labels: Ground-truth labels from professional graphologists,
            mapping indicator name (e.g. ``"slant"``, ``"pressure"``) to a
            label/value string. Empty when the sample is not yet labeled.

    Raises:
        ValueError: If ``acquisition_method`` or ``quality`` is not one of
            the accepted values, or if ``sample_id`` is empty.
    """

    sample_id: str
    acquisition_method: AcquisitionMethod
    quality: Quality
    language: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")

        if self.acquisition_method not in VALID_ACQUISITION_METHODS:
            raise ValueError(
                f"invalid acquisition_method {self.acquisition_method!r}: "
                f"must be one of {VALID_ACQUISITION_METHODS!r}"
            )

        if self.quality not in VALID_QUALITIES:
            raise ValueError(
                f"invalid quality {self.quality!r}: "
                f"must be one of {VALID_QUALITIES!r}"
            )

        if not isinstance(self.labels, dict):
            raise ValueError("labels must be a dict of indicator name -> label")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this entry to a plain JSON-compatible dict."""
        return {
            "sample_id": self.sample_id,
            "acquisition_method": self.acquisition_method,
            "quality": self.quality,
            "language": self.language,
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        """Construct a :class:`ManifestEntry` from a plain dict.

        Raises:
            ValueError: If required fields are missing, or if
                ``acquisition_method``/``quality`` are invalid.
        """
        if not isinstance(data, dict):
            raise ValueError(f"manifest entry must be a dict, got {type(data)!r}")

        required = ("sample_id", "acquisition_method", "quality")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"manifest entry missing required field(s): {missing}")

        return cls(
            sample_id=data["sample_id"],
            acquisition_method=data["acquisition_method"],
            quality=data["quality"],
            language=data.get("language"),
            labels=dict(data.get("labels", {})),
        )
