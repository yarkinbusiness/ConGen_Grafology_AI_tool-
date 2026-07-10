"""Tests for grafology_ai.dataset.schema.ManifestEntry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grafology_ai.dataset.schema import ManifestEntry

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MANIFEST_PATH = FIXTURES_DIR / "sample_manifest.json"
JSON_SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "grafology_ai"
    / "dataset"
    / "manifest.schema.json"
)


def _load_json_schema() -> dict:
    with JSON_SCHEMA_PATH.open() as f:
        return json.load(f)


def _validate_against_json_schema(entry: dict, schema: dict) -> None:
    """Minimal structural validation against manifest.schema.json.

    Checks required fields, allowed enum values, additionalProperties, and
    basic types -- without adding a jsonschema dependency for this
    lightweight, fixed schema.
    """
    required = schema.get("required", [])
    for key in required:
        assert key in entry, f"missing required field {key!r}"

    properties = schema.get("properties", {})
    if not schema.get("additionalProperties", True):
        unknown = set(entry) - set(properties)
        assert not unknown, f"unexpected field(s) not in schema: {unknown}"

    enum = properties.get("acquisition_method", {}).get("enum")
    if enum is not None:
        assert entry["acquisition_method"] in enum

    enum = properties.get("quality", {}).get("enum")
    if enum is not None:
        assert entry["quality"] in enum

    assert isinstance(entry["sample_id"], str) and entry["sample_id"]
    assert entry.get("language") is None or isinstance(entry["language"], str)
    assert isinstance(entry.get("labels", {}), dict)


def test_valid_entry_round_trips_through_dict() -> None:
    data = {
        "sample_id": "sample-abc",
        "acquisition_method": "scan",
        "quality": "high",
        "language": "it",
        "labels": {"slant": "right", "pressure": "medium"},
    }
    entry = ManifestEntry.from_dict(data)

    assert entry.sample_id == "sample-abc"
    assert entry.acquisition_method == "scan"
    assert entry.quality == "high"
    assert entry.language == "it"
    assert entry.labels == {"slant": "right", "pressure": "medium"}

    round_tripped = ManifestEntry.from_dict(entry.to_dict())
    assert round_tripped == entry


def test_valid_entry_with_defaults_round_trips() -> None:
    data = {
        "sample_id": "sample-empty-labels",
        "acquisition_method": "photo",
        "quality": "low",
    }
    entry = ManifestEntry.from_dict(data)

    assert entry.language is None
    assert entry.labels == {}

    round_tripped = ManifestEntry.from_dict(entry.to_dict())
    assert round_tripped == entry


def test_invalid_acquisition_method_raises_value_error() -> None:
    data = {
        "sample_id": "sample-bad-method",
        "acquisition_method": "fax",
        "quality": "high",
    }
    with pytest.raises(ValueError, match="acquisition_method"):
        ManifestEntry.from_dict(data)


def test_invalid_quality_raises_value_error() -> None:
    data = {
        "sample_id": "sample-bad-quality",
        "acquisition_method": "scan",
        "quality": "excellent",
    }
    with pytest.raises(ValueError, match="quality"):
        ManifestEntry.from_dict(data)


def test_missing_required_field_raises_value_error() -> None:
    data = {"acquisition_method": "scan", "quality": "high"}
    with pytest.raises(ValueError, match="sample_id"):
        ManifestEntry.from_dict(data)


def test_json_schema_file_is_valid_and_matches_dataclass_fields() -> None:
    schema = _load_json_schema()

    dataclass_fields = {
        "sample_id",
        "acquisition_method",
        "quality",
        "language",
        "labels",
    }
    schema_fields = set(schema["properties"])
    assert schema_fields == dataclass_fields

    assert set(schema["required"]) == {
        "sample_id",
        "acquisition_method",
        "quality",
    }


def test_sample_manifest_fixture_validates_against_schema_and_dataclass() -> None:
    with SAMPLE_MANIFEST_PATH.open() as f:
        raw_entries = json.load(f)

    assert len(raw_entries) >= 2

    schema = _load_json_schema()
    entries = []
    for raw_entry in raw_entries:
        _validate_against_json_schema(raw_entry, schema)
        entries.append(ManifestEntry.from_dict(raw_entry))

    # At least one fully labeled entry and at least one unlabeled entry,
    # as required by the fixture's design.
    assert any(entry.labels for entry in entries)
    assert any(not entry.labels for entry in entries)
