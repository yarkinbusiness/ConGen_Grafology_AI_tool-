"""Tests for grafology_ai.validation.validators.

Fixture images are generated programmatically with Pillow rather than
loaded from external files, so the test suite has no external data
dependency:

- ``_make_sharp_image`` draws high-contrast, pen-stroke-like lines on a
  light background at an adequate pixel size -- the "good" case for
  resolution, blur, and contrast.
- ``_make_isolated_blur_bad_image`` is a mildly Gaussian-blurred copy of
  the sharp image, tuned so *only* the blur check fails (contrast stays
  well above its threshold), for testing the blur-specific
  quality_label="low" downgrade in isolation.
- ``_make_isolated_low_contrast_image`` is a near-uniform gray field with
  small per-pixel noise, tuned so *only* the contrast check fails (the
  noise still produces enough high-frequency variation for the blur check
  to pass), for testing the contrast-specific downgrade in isolation.
- ``_make_heavily_blurred_image`` / ``_make_low_contrast_image`` are
  simpler, non-isolated bad fixtures used for the per-check good/bad tests.
- ``_make_tiny_image`` is a 20x20 image, too small to pass the resolution
  check regardless of any other property.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from grafology_ai.validation import (
    ValidationResult,
    check_blur,
    check_contrast,
    check_format,
    check_resolution,
    validate_sample,
)

GOOD_SIZE = (800, 600)


def _make_sharp_image(size: tuple[int, int] = GOOD_SIZE) -> Image.Image:
    """A high-contrast image with crisp, handwriting-like strokes."""
    image = Image.new("L", size, color=245)
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7)
    for line_y in range(50, size[1] - 30, 30):
        x = 15
        while x < size[0] - 15:
            x2 = x + int(rng.integers(15, 35))
            y_off = int(rng.integers(-6, 6))
            draw.line(
                [(x, line_y + y_off), (x2, line_y + int(rng.integers(-6, 6)))],
                fill=5,
                width=6,
            )
            x = x2 + int(rng.integers(3, 10))
    return image.convert("RGB")


def _make_heavily_blurred_image() -> Image.Image:
    """A clearly-blurred bad fixture (also drags contrast down)."""
    return _make_sharp_image().filter(ImageFilter.GaussianBlur(radius=8))


def _make_isolated_blur_bad_image() -> Image.Image:
    """Blurred just enough to fail *only* the blur check, not contrast."""
    return _make_sharp_image().filter(ImageFilter.GaussianBlur(radius=2))


def _make_low_contrast_image(size: tuple[int, int] = GOOD_SIZE) -> Image.Image:
    """A near-uniform gray field: the simple, non-isolated bad fixture."""
    rng = np.random.default_rng(1)
    array = np.full((size[1], size[0]), 128, dtype=np.int16)
    noise = rng.integers(-3, 4, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _make_isolated_low_contrast_image(size: tuple[int, int] = GOOD_SIZE) -> Image.Image:
    """Low-contrast but with enough noise texture to keep blur passing."""
    rng = np.random.default_rng(3)
    array = np.full((size[1], size[0]), 130, dtype=np.int16)
    noise = rng.integers(-8, 9, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _make_tiny_image() -> Image.Image:
    return Image.new("RGB", (20, 20), color=(255, 255, 255))


def _save(tmp_path: Path, image: Image.Image, name: str, fmt: str) -> Path:
    path = tmp_path / name
    image.save(path, format=fmt)
    return path


# --- check_resolution -------------------------------------------------


def test_check_resolution_accepts_adequately_sized_image() -> None:
    result = check_resolution(_make_sharp_image())
    assert result.check_name == "resolution"
    assert result.verdict == "accept"
    assert result.measured_value == 600.0


def test_check_resolution_rejects_tiny_image() -> None:
    result = check_resolution(_make_tiny_image())
    assert result.check_name == "resolution"
    assert result.verdict == "reject"
    assert result.measured_value == 20.0


# --- check_blur ---------------------------------------------------------


def test_check_blur_accepts_sharp_image() -> None:
    result = check_blur(_make_sharp_image())
    assert result.check_name == "blur"
    assert result.verdict == "accept"
    assert result.measured_value is not None and result.measured_value > 150.0


def test_check_blur_rejects_blurred_image() -> None:
    result = check_blur(_make_heavily_blurred_image())
    assert result.check_name == "blur"
    assert result.verdict == "reject"
    assert result.measured_value is not None and result.measured_value < 150.0


# --- check_contrast -------------------------------------------------------


def test_check_contrast_accepts_high_contrast_image() -> None:
    result = check_contrast(_make_sharp_image())
    assert result.check_name == "contrast"
    assert result.verdict == "accept"
    assert result.measured_value is not None and result.measured_value >= 25.0


def test_check_contrast_rejects_low_contrast_image() -> None:
    result = check_contrast(_make_low_contrast_image())
    assert result.check_name == "contrast"
    assert result.verdict == "reject"
    assert result.measured_value is not None and result.measured_value < 25.0


# --- check_format -----------------------------------------------------


@pytest.mark.parametrize("fmt,ext", [("PNG", "png"), ("JPEG", "jpg")])
def test_check_format_accepts_supported_formats(
    tmp_path: Path, fmt: str, ext: str
) -> None:
    path = _save(tmp_path, _make_sharp_image(), f"good.{ext}", fmt)
    result = check_format(path)
    assert result.check_name == "format"
    assert result.verdict == "accept"


def test_check_format_rejects_unsupported_format(tmp_path: Path) -> None:
    path = _save(tmp_path, _make_sharp_image(), "bad.gif", "GIF")
    result = check_format(path)
    assert result.check_name == "format"
    assert result.verdict == "reject"
    assert "GIF" in result.reason


def test_check_format_rejects_in_memory_image_with_no_file() -> None:
    result = check_format(_make_sharp_image())
    assert result.check_name == "format"
    assert result.verdict == "reject"


# --- validate_sample ------------------------------------------------------


def test_validate_sample_returns_one_result_per_check_with_correct_names(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path, _make_sharp_image(), "good.png", "PNG")
    results = validate_sample(path)

    assert len(results) == 4
    assert all(isinstance(result, ValidationResult) for result in results)
    assert {result.check_name for result in results} == {
        "format",
        "resolution",
        "blur",
        "contrast",
    }


def test_validate_sample_good_image_accepts_every_check(tmp_path: Path) -> None:
    path = _save(tmp_path, _make_sharp_image(), "good.png", "PNG")
    results = validate_sample(path)

    assert all(result.verdict == "accept" for result in results)


def test_validate_sample_downgrades_blur_reject_to_flag_when_quality_low(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path, _make_isolated_blur_bad_image(), "blurry.png", "PNG")

    default_results = {r.check_name: r for r in validate_sample(path)}
    assert default_results["blur"].verdict == "reject"

    low_quality_results = {
        r.check_name: r for r in validate_sample(path, quality_label="low")
    }
    assert low_quality_results["blur"].verdict == "flag"


def test_validate_sample_downgrades_contrast_reject_to_flag_when_quality_low(
    tmp_path: Path,
) -> None:
    path = _save(
        tmp_path, _make_isolated_low_contrast_image(), "low_contrast.png", "PNG"
    )

    default_results = {r.check_name: r for r in validate_sample(path)}
    assert default_results["contrast"].verdict == "reject"

    low_quality_results = {
        r.check_name: r for r in validate_sample(path, quality_label="low")
    }
    assert low_quality_results["contrast"].verdict == "flag"


def test_validate_sample_resolution_failure_still_rejects_when_quality_low(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path, _make_tiny_image(), "tiny.png", "PNG")

    results = {
        r.check_name: r for r in validate_sample(path, quality_label="low")
    }
    assert results["resolution"].verdict == "reject"


def test_validate_sample_format_failure_still_rejects_when_quality_low(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path, _make_sharp_image(), "bad_format.bmp", "BMP")

    results = {
        r.check_name: r for r in validate_sample(path, quality_label="low")
    }
    assert results["format"].verdict == "reject"
