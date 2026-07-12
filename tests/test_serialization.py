"""Tests for the ``to_dict()`` serialization capability (Phase D1).

Every frozen dataclass that makes up an :class:`~grafology_ai.run_analysis.
AnalysisResult` -- :class:`~grafology_ai.analysis.Features`,
:class:`~grafology_ai.interpretation.Finding` /
:class:`~grafology_ai.interpretation.StructuredFindings`,
:class:`~grafology_ai.validation.ValidationResult`, and
:class:`~grafology_ai.run_analysis.AnalysisResult` itself -- has a
``to_dict()`` method that returns a plain, JSON-safe ``dict``. This module
verifies that capability end-to-end: real pipeline output, at both
``depth`` settings, actually serializes to JSON, every dataclass field is
present in the output (checked programmatically against
``dataclasses.fields`` so the check cannot silently go stale), the
conversion is deterministic, and no non-JSON-safe type (a ``Path``, a
dataclass instance, a numpy scalar, etc.) leaks through anywhere in the
nested structure.

Fixture images reuse the same deterministic-drawing approach as
``tests/test_run_analysis.py``'s ``_draw_stroke_grid``/
``_write_clean_sample`` helpers.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from grafology_ai.analysis import Features
from grafology_ai.interpretation import Finding, StructuredFindings
from grafology_ai.run_analysis import AnalysisResult, run_analysis
from grafology_ai.validation import ValidationResult

BACKGROUND = 255
INK = 0

# JSON-safe leaf/container types per the acceptance bar: only these may
# appear anywhere in a serialized dict.
_JSON_SAFE_SCALAR_TYPES = (str, int, float, bool, type(None))


def _draw_stroke_grid(
    size: tuple[int, int],
    *,
    slant_deg: float = 12.0,
    stroke_width: int = 6,
    stroke_height: int = 18,
    stroke_spacing: int = 16,
    line_spacing: int = 42,
    margin: int = 30,
) -> Image.Image:
    """A deterministic grid of short strokes arranged in lines.

    Mirrors ``tests/test_run_analysis.py``'s ``_draw_stroke_grid``.
    """
    image = Image.new("L", size, color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    dx = stroke_height * math.tan(math.radians(slant_deg))

    y = margin + stroke_height
    while y <= size[1] - margin:
        x = margin
        while x <= size[0] - margin:
            draw.line([(x, y), (x + dx, y - stroke_height)], fill=INK, width=stroke_width)
            x += stroke_spacing
        y += line_spacing
    return image.convert("RGB")


def _write_clean_sample(tmp_path: Path, name: str = "sample.png") -> Path:
    """A stroke-grid image, saved to disk, that passes every automated check.

    Saved and reopened by path for the same reason
    ``tests/test_run_analysis.py`` does: ``check_format`` needs a real
    file/``Image.open()`` result to detect a supported format from.
    """
    path = tmp_path / name
    _draw_stroke_grid((800, 1000)).save(path, format="PNG")
    return path


def _assert_json_safe(value: Any, *, path: str = "$") -> None:
    """Recursively assert every leaf/container in `value` is JSON-safe.

    JSON-safe here means: exactly `dict`, `list`, `str`, `int`, `float`,
    `bool`, or `None` -- no `Path`, no dataclass instance, no numpy
    scalar, no other custom object anywhere in the nested structure.
    """
    if isinstance(value, dict):
        assert type(value) is dict, f"{path}: expected a plain dict, got {type(value)!r}"
        for key, sub_value in value.items():
            assert isinstance(key, str), f"{path}: dict key {key!r} is not a str"
            _assert_json_safe(sub_value, path=f"{path}.{key}")
    elif isinstance(value, list):
        assert type(value) is list, f"{path}: expected a plain list, got {type(value)!r}"
        for index, item in enumerate(value):
            _assert_json_safe(item, path=f"{path}[{index}]")
    else:
        assert isinstance(value, _JSON_SAFE_SCALAR_TYPES), (
            f"{path}: value {value!r} has non-JSON-safe type {type(value)!r}"
        )
        # bool is a subclass of int, so being in `_JSON_SAFE_SCALAR_TYPES`
        # via `int`/`float` is fine, but explicitly reject dataclass
        # instances / Path objects that happen to also be falsy-truthy --
        # dataclasses.is_dataclass would raise on primitives so guard first.
        assert not dataclasses.is_dataclass(value), f"{path}: dataclass instance leaked through"


def _dataclass_field_names(cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def _make_result(tmp_path: Path, depth: str) -> AnalysisResult:
    sample_path = _write_clean_sample(tmp_path)
    return run_analysis(sample_path, depth=depth, sample_id="ser-001")  # type: ignore[arg-type]


# --- 1: real pipeline output, both depths, json.dumps succeeds --------------


def test_concise_result_dict_is_json_dumpable(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "concise")
    result_dict = result.to_dict()
    dumped = json.dumps(result_dict)
    assert isinstance(dumped, str) and dumped


def test_indepth_result_dict_is_json_dumpable(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    result_dict = result.to_dict()
    dumped = json.dumps(result_dict)
    assert isinstance(dumped, str) and dumped


# --- 2/3: programmatic field-completeness checks -----------------------------


def test_features_to_dict_has_every_dataclass_field(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    features_dict = result.features.to_dict()

    expected = _dataclass_field_names(Features)
    assert expected <= set(features_dict.keys())
    assert set(features_dict.keys()) == expected

    # confidence is a nested dict[str, float], not flattened/dropped.
    assert isinstance(features_dict["confidence"], dict)
    assert features_dict["confidence"] == result.features.confidence
    assert features_dict["confidence"] is not result.features.confidence


def test_structured_findings_to_dict_has_every_dataclass_field(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    findings_dict = result.findings.to_dict()

    expected = _dataclass_field_names(StructuredFindings)
    assert set(findings_dict.keys()) == expected

    assert isinstance(findings_dict["findings"], list)
    assert findings_dict["findings"], "expected at least one Finding at depth=indepth"

    finding_expected = _dataclass_field_names(Finding)
    for finding_dict in findings_dict["findings"]:
        assert isinstance(finding_dict, dict)
        assert set(finding_dict.keys()) == finding_expected
        for field_name in finding_expected:
            assert field_name in finding_dict
    # Spot-check the well-known Finding field names named in the task.
    assert finding_expected == {"indicator", "observation", "interpretation", "confidence"}


def test_validation_result_to_dict_has_every_dataclass_field(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    assert result.validation_results, "expected at least one ValidationResult"

    expected = _dataclass_field_names(ValidationResult)
    for validation_result in result.validation_results:
        vr_dict = validation_result.to_dict()
        assert set(vr_dict.keys()) == expected


def test_analysis_result_to_dict_has_every_dataclass_field_including_nested(
    tmp_path: Path,
) -> None:
    result = _make_result(tmp_path, "indepth")
    result_dict = result.to_dict()

    expected = _dataclass_field_names(AnalysisResult)
    assert set(result_dict.keys()) == expected

    # Nested structures, not nested dataclass instances.
    assert isinstance(result_dict["features"], dict)
    assert set(result_dict["features"].keys()) == _dataclass_field_names(Features)

    assert isinstance(result_dict["findings"], dict)
    assert set(result_dict["findings"].keys()) == _dataclass_field_names(StructuredFindings)

    assert isinstance(result_dict["validation_results"], list)
    vr_expected = _dataclass_field_names(ValidationResult)
    for vr_dict in result_dict["validation_results"]:
        assert isinstance(vr_dict, dict)
        assert set(vr_dict.keys()) == vr_expected

    # validation_results is a list of plain dicts, each with all 4 fields --
    # not ValidationResult instances.
    assert all(not dataclasses.is_dataclass(vr) for vr in result_dict["validation_results"])
    assert not dataclasses.is_dataclass(result_dict["features"])
    assert not dataclasses.is_dataclass(result_dict["findings"])


# --- 4: determinism -----------------------------------------------------------


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")

    first = result.to_dict()
    second = result.to_dict()
    assert first == second

    assert result.features.to_dict() == result.features.to_dict()
    assert result.findings.to_dict() == result.findings.to_dict()
    for vr in result.validation_results:
        assert vr.to_dict() == vr.to_dict()


# --- 5: round-trip safety + no non-JSON-safe type leaks anywhere ------------


def test_round_trip_through_json_preserves_structure(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    result_dict = result.to_dict()

    round_tripped = json.loads(json.dumps(result_dict))
    assert round_tripped == result_dict


def test_no_non_json_safe_types_leak_through_analysis_result(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    _assert_json_safe(result.to_dict())


def test_no_non_json_safe_types_leak_through_concise_analysis_result(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "concise")
    _assert_json_safe(result.to_dict())


def test_no_non_json_safe_types_leak_through_component_dicts(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "indepth")
    _assert_json_safe(result.features.to_dict())
    _assert_json_safe(result.findings.to_dict())
    for vr in result.validation_results:
        _assert_json_safe(vr.to_dict())
