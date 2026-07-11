"""Tests for grafology_ai.input.pdf.

Every PDF fixture here is built in-test with Pillow itself
(``Image.save(..., format="PDF")``, multi-page via ``save_all=True,
append_images=[...]``) -- no extra PDF-generation dependency, no checked
in binary fixture files, matching the deterministic-PIL-fixture
convention used throughout this test suite (see e.g.
``tests/test_analysis.py``, ``grafology_ai.dataset.fixtures``).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from grafology_ai.analysis import analyze
from grafology_ai.input import (
    DEFAULT_PDF_RASTER_DPI,
    PdfInputError,
    PdfRasterization,
    is_pdf,
    rasterize_pdf,
)
from grafology_ai.interpretation.interpret import (
    SLANT_MODERATE_MAX_DEGREES,
    SLANT_NEUTRAL_MAX_DEGREES,
)


def _solid_page(size: tuple[int, int], gray_value: int) -> Image.Image:
    """A single-color page image, for pages that must be trivially distinguishable."""
    return Image.new("L", size, color=gray_value).convert("RGB")


def _pdf_bytes_from_images(images: list[Image.Image]) -> bytes:
    """Save `images` as a (possibly multi-page) in-memory PDF and return its bytes."""
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def _draw_stroke_grid(
    size: tuple[int, int],
    *,
    slant_deg: float = 0.0,
    stroke_width: int = 4,
    stroke_height: int = 16,
    stroke_spacing: int = 14,
    line_spacing: int = 40,
    margin: int = 30,
) -> Image.Image:
    """A deterministic grid of short, slanted strokes -- same drawing convention
    as ``tests/test_analysis.py``'s ``_draw_stroke_grid`` (``slant_deg > 0``
    leans right, matching ``Features.slant_angle_degrees``'s sign convention).
    """
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)
    dx = stroke_height * math.tan(math.radians(slant_deg))

    y = margin + stroke_height
    while y <= size[1] - margin:
        x = margin
        while x <= size[0] - margin:
            draw.line([(x, y), (x + dx, y - stroke_height)], fill=0, width=stroke_width)
            x += stroke_spacing
        y += line_spacing
    return image.convert("RGB")


def _slant_bucket(angle_degrees: float) -> str:
    """Classify `angle_degrees` into the same neutral/moderate/pronounced
    buckets `grafology_ai.interpretation.interpret._candidate_slant` uses,
    via its published threshold constants.
    """
    abs_angle = abs(angle_degrees)
    if abs_angle <= SLANT_NEUTRAL_MAX_DEGREES:
        return "neutral"
    if abs_angle <= SLANT_MODERATE_MAX_DEGREES:
        return "moderate"
    return "pronounced"


# --- DPI scaling -------------------------------------------------------------


def test_rasterize_pdf_pixel_dimensions_scale_with_dpi() -> None:
    page = _solid_page((400, 300), 255)
    pdf_bytes = _pdf_bytes_from_images([page])

    low = rasterize_pdf(pdf_bytes, dpi=150, page_index=0)
    high = rasterize_pdf(pdf_bytes, dpi=300, page_index=0)

    low_width, low_height = low.image.size
    high_width, high_height = high.image.size

    assert abs(high_width - 2 * low_width) <= 1
    assert abs(high_height - 2 * low_height) <= 1
    assert low.dpi_used == 150
    assert high.dpi_used == 300


# --- multi-page: page_count and page_index selection --------------------------


def test_multi_page_pdf_reports_correct_page_count_and_selects_correct_page() -> None:
    gray_values = [40, 130, 220]
    pages = [_solid_page((300, 200), value) for value in gray_values]
    pdf_bytes = _pdf_bytes_from_images(pages)

    for index, expected_value in enumerate(gray_values):
        result = rasterize_pdf(pdf_bytes, dpi=150, page_index=index)
        assert result.page_count == 3

        arr = np.asarray(result.image.convert("L"), dtype=np.float64)
        assert arr.mean() == pytest.approx(expected_value, abs=1.0)


# --- is_pdf sniffing -----------------------------------------------------------


def test_is_pdf_false_for_arbitrary_non_pdf_bytes() -> None:
    assert is_pdf(b"just some random bytes, not a pdf at all \x00\x01\x02") is False


def test_is_pdf_false_for_png_bytes() -> None:
    image = _solid_page((50, 50), 100)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    assert is_pdf(buf.getvalue()) is False


def test_is_pdf_true_for_real_pdf_bytes() -> None:
    pdf_bytes = _pdf_bytes_from_images([_solid_page((50, 50), 100)])
    assert is_pdf(pdf_bytes) is True


# --- corrupt input handling ------------------------------------------------------


def test_rasterize_pdf_raises_pdf_input_error_for_truncated_corrupt_bytes() -> None:
    # Starts with the real %PDF- magic header (so is_pdf would say True) but
    # is not a parseable PDF document beyond that.
    corrupt = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nthis is not really a pdf body at all"

    with pytest.raises(PdfInputError):
        rasterize_pdf(corrupt)


def test_rasterize_pdf_corrupt_bytes_error_is_not_a_raw_pdfium_exception() -> None:
    corrupt = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nthis is not really a pdf body at all"

    try:
        rasterize_pdf(corrupt)
    except PdfInputError as exc:
        assert type(exc) is PdfInputError
    else:
        pytest.fail("expected PdfInputError to be raised")


# --- determinism -----------------------------------------------------------------


def test_rasterization_is_deterministic_across_repeated_calls() -> None:
    image = _draw_stroke_grid((400, 300), slant_deg=15.0)
    pdf_bytes = _pdf_bytes_from_images([image])

    first = rasterize_pdf(pdf_bytes, dpi=200, page_index=0)
    second = rasterize_pdf(pdf_bytes, dpi=200, page_index=0)

    first_arr = np.asarray(first.image)
    second_arr = np.asarray(second.image)
    assert np.array_equal(first_arr, second_arr)


# --- parity with analysis: rasterization shouldn't flip the slant bucket --------


def test_rasterized_pdf_page_yields_same_slant_bucket_as_original_image() -> None:
    image = _draw_stroke_grid((560, 400), slant_deg=12.0)

    original_features = analyze(image)

    pdf_bytes = _pdf_bytes_from_images([image])
    rasterization = rasterize_pdf(pdf_bytes, dpi=DEFAULT_PDF_RASTER_DPI, page_index=0)
    rasterized_features = analyze(rasterization.image)

    assert _slant_bucket(original_features.slant_angle_degrees) == _slant_bucket(
        rasterized_features.slant_angle_degrees
    )


# --- out-of-range page_index ------------------------------------------------------


def test_out_of_range_page_index_raises_pdf_input_error_not_index_error() -> None:
    pdf_bytes = _pdf_bytes_from_images([_solid_page((200, 150), 128)])

    with pytest.raises(PdfInputError):
        rasterize_pdf(pdf_bytes, page_index=5)


def test_negative_page_index_raises_pdf_input_error() -> None:
    pdf_bytes = _pdf_bytes_from_images([_solid_page((200, 150), 128)])

    with pytest.raises(PdfInputError):
        rasterize_pdf(pdf_bytes, page_index=-1)


# --- accepts path input too, matching PdfInput's stated flexibility -------------


def test_rasterize_pdf_accepts_path_input(tmp_path) -> None:
    image = _solid_page((200, 150), 128)
    pdf_bytes = _pdf_bytes_from_images([image])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(pdf_bytes)

    from_path = rasterize_pdf(pdf_path)
    from_bytes = rasterize_pdf(pdf_bytes)

    assert from_path.image.size == from_bytes.image.size
    assert is_pdf(pdf_path) is True


# --- PdfRasterization shape --------------------------------------------------------


def test_pdf_rasterization_is_frozen_dataclass_with_expected_fields() -> None:
    pdf_bytes = _pdf_bytes_from_images([_solid_page((100, 80), 200)])
    result = rasterize_pdf(pdf_bytes)

    assert isinstance(result, PdfRasterization)
    assert isinstance(result.image, Image.Image)
    assert result.page_count == 1
    assert result.dpi_used == DEFAULT_PDF_RASTER_DPI

    with pytest.raises(Exception):
        result.page_count = 99  # type: ignore[misc]
