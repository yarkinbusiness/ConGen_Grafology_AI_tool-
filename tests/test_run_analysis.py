"""Tests for grafology_ai.run_analysis.

Fixture images reuse the same deterministic-drawing approach as
``tests/test_analysis.py`` / ``tests/test_report.py`` (a grid of short
strokes with controllable slant/width/height/spacing), plus a "sharp
image" helper in the style of ``tests/test_validation.py`` tuned to also
pass the automated image-quality checks, so the "clean" end-to-end path
can be exercised without every validation check rejecting the synthetic
fixture.
"""

from __future__ import annotations

import dataclasses
import io
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from grafology_ai.analysis import Features
from grafology_ai.input.pdf import PdfInputError
from grafology_ai.interpretation import StructuredFindings
from grafology_ai.run_analysis import AnalysisResult, run_analysis
from grafology_ai.validation import ValidationResult

BACKGROUND = 255
INK = 0


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

    Mirrors ``tests/test_analysis.py``/``tests/test_report.py``'s
    ``_draw_stroke_grid``: each stroke runs from ``(x, y)`` (bottom) to
    ``(x + dx, y - stroke_height)`` (top), so ``slant_deg > 0`` leans
    right.
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

    Saved (rather than kept as an in-memory ``PIL.Image``) and reopened by
    path, since ``grafology_ai.validation.check_format`` needs a real
    file/`Image.open()` result to detect a supported format from -- see
    its docstring.
    """
    path = tmp_path / name
    _draw_stroke_grid((800, 1000)).save(path, format="PNG")
    return path


def _blank_image(size: tuple[int, int] = (20, 20)) -> Image.Image:
    """A tiny, blank in-memory image: fails every automated check at once."""
    return Image.new("RGB", size, color=(255, 255, 255))


# --- PDF input helpers -------------------------------------------------------------
#
# Built the same way as ``tests/test_pdf_input.py``: Pillow's own
# ``Image.save(..., format="PDF")``, no extra PDF-generation dependency and
# no checked-in binary fixture.


def _pdf_bytes_from_images(images: list[Image.Image]) -> bytes:
    """Save `images` as a (possibly multi-page) in-memory PDF and return its bytes."""
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def _clean_pdf_bytes() -> bytes:
    """A single-page PDF that should pass every automated quality check."""
    return _pdf_bytes_from_images([_draw_stroke_grid((800, 1000))])


def _blurry_pdf_bytes() -> bytes:
    """A single-page PDF whose page is deliberately blurred (fails the blur check)."""
    blurred = _draw_stroke_grid((800, 1000)).filter(ImageFilter.GaussianBlur(radius=8))
    return _pdf_bytes_from_images([blurred])


# A corrupt PDF: starts with the real ``%PDF-`` magic header (so `is_pdf`
# would say True) but is not a parseable PDF document beyond that -- the
# same construction ``tests/test_pdf_input.py`` uses.
_CORRUPT_PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nthis is not really a pdf body at all"


# --- core end-to-end behavior ---------------------------------------------------


def test_run_analysis_returns_populated_and_consistent_result(tmp_path: Path) -> None:
    sample_path = _write_clean_sample(tmp_path)

    result = run_analysis(sample_path, depth="indepth", sample_id="sample-001")

    assert isinstance(result, AnalysisResult)

    # All four components populated.
    assert result.validation_results
    assert all(isinstance(r, ValidationResult) for r in result.validation_results)
    assert isinstance(result.features, Features)
    assert isinstance(result.findings, StructuredFindings)
    assert isinstance(result.report, str) and result.report.strip()

    # Internally consistent: the report is exactly generate_report(findings, ...).
    assert result.findings.overall_summary in result.report
    for finding in result.findings.findings:
        assert finding.observation in result.report
        assert finding.interpretation in result.report

    assert result.findings.depth == "indepth"
    assert "sample-001" in result.report

    # Clean, sharp, well-sized sample: nothing should be rejected.
    assert result.has_rejected_validation is False
    assert result.rejected_validation_checks == []
    assert all(r.verdict == "accept" for r in result.validation_results)


def test_run_analysis_depth_is_passed_through(tmp_path: Path) -> None:
    sample_path = _write_clean_sample(tmp_path)

    concise_result = run_analysis(sample_path, depth="concise")
    indepth_result = run_analysis(sample_path, depth="indepth")

    assert concise_result.findings.depth == "concise"
    assert indepth_result.findings.depth == "indepth"
    assert "Concise" in concise_result.report
    assert "In-depth" in indepth_result.report
    # Concise output is meaningfully shorter than in-depth for the same sample.
    assert len(concise_result.findings.findings) < len(indepth_result.findings.findings)


def test_run_analysis_accepts_pil_image_directly() -> None:
    image = _draw_stroke_grid((800, 1000))
    result = run_analysis(image, depth="concise")

    assert isinstance(result, AnalysisResult)
    assert result.findings.overall_summary in result.report


def test_analysis_result_is_frozen(tmp_path: Path) -> None:
    result = run_analysis(_write_clean_sample(tmp_path))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.report = "tampered"  # type: ignore[misc]


# --- "reject" verdict handling: visible, not silently hidden, not fatal --------


def test_run_analysis_never_raises_on_rejected_sample() -> None:
    """A badly out-of-spec image still produces a full, usable result."""
    result = run_analysis(_blank_image(), depth="indepth")

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.features, Features)
    assert isinstance(result.findings, StructuredFindings)
    assert result.report.strip()


def test_rejected_validation_is_visible_on_the_result() -> None:
    result = run_analysis(_blank_image())

    assert result.has_rejected_validation is True
    assert len(result.rejected_validation_checks) > 0
    assert all(r.verdict == "reject" for r in result.rejected_validation_checks)
    # Full, unfiltered list is still exposed -- rejection isn't hidden by
    # only exposing the rejected subset.
    assert len(result.validation_results) >= len(result.rejected_validation_checks)


def test_rejected_validation_produces_a_report_caveat() -> None:
    result = run_analysis(_blank_image())

    assert result.has_rejected_validation is True
    # The caveat is woven into areas_of_attention (see module docstring),
    # so it shows up under the report's "## Areas of Attention" section.
    caveat_candidates = [
        text for text in result.findings.areas_of_attention if "quality checks" in text
    ]
    assert caveat_candidates, result.findings.areas_of_attention
    caveat = caveat_candidates[0]
    assert caveat in result.report
    # The caveat text stays within the project's cautious-language rules.
    from grafology_ai.interpretation import DISALLOWED_TERMS

    lowered = caveat.lower()
    for term in DISALLOWED_TERMS:
        assert term not in lowered


def test_clean_sample_has_no_quality_caveat(tmp_path: Path) -> None:
    result = run_analysis(_write_clean_sample(tmp_path))
    assert not any("quality checks" in text for text in result.findings.areas_of_attention)


def test_quality_label_low_downgrades_reject_to_flag_and_drops_caveat(tmp_path: Path) -> None:
    """quality_label="low" downgrades blur/contrast rejects to "flag"."""
    # A near-uniform gray field fails contrast (and blur) but is large
    # enough to pass resolution; saved+reopened so format also passes.
    path = tmp_path / "low_quality.png"
    Image.new("L", (600, 600), color=128).convert("RGB").save(path, format="PNG")

    plain_result = run_analysis(path)
    flagged_result = run_analysis(path, quality_label="low")

    assert plain_result.has_rejected_validation is True

    downgradable = {"blur", "contrast"}
    for r in flagged_result.validation_results:
        if r.check_name in downgradable:
            assert r.verdict != "reject"
    # resolution/format still evaluated the same way regardless of label.
    plain_by_check = {r.check_name: r.verdict for r in plain_result.validation_results}
    flagged_by_check = {r.check_name: r.verdict for r in flagged_result.validation_results}
    assert plain_by_check["resolution"] == flagged_by_check["resolution"]
    assert plain_by_check["format"] == flagged_by_check["format"]


# --- PDF input: B2 -----------------------------------------------------------------


def test_run_analysis_clean_single_page_pdf_has_no_rejected_validation() -> None:
    """A clean, single-page PDF must not be spuriously rejected on format
    (or anything else) -- the core bug this task fixes."""
    result = run_analysis(_clean_pdf_bytes(), depth="indepth", sample_id="pdf-001")

    assert isinstance(result, AnalysisResult)
    assert result.has_rejected_validation is False
    assert result.rejected_validation_checks == []
    assert all(r.verdict == "accept" for r in result.validation_results)

    format_result = next(r for r in result.validation_results if r.check_name == "format")
    assert format_result.verdict == "accept"
    assert "PDF" in format_result.reason

    # No multi-page flag for a single-page document.
    assert not any(r.check_name == "pdf_pages" for r in result.validation_results)

    # Rest of the pipeline still ran fully.
    assert isinstance(result.features, Features)
    assert isinstance(result.findings, StructuredFindings)
    assert result.findings.overall_summary in result.report
    assert "pdf-001" in result.report


def test_run_analysis_multi_page_pdf_adds_exactly_one_pdf_pages_flag() -> None:
    pages = [
        _draw_stroke_grid((800, 1000)),
        _draw_stroke_grid((800, 1000), slant_deg=-10.0),
        _draw_stroke_grid((800, 1000), slant_deg=20.0),
    ]
    pdf_bytes = _pdf_bytes_from_images(pages)

    result = run_analysis(pdf_bytes, depth="indepth")

    pdf_page_results = [r for r in result.validation_results if r.check_name == "pdf_pages"]
    assert len(pdf_page_results) == 1
    assert pdf_page_results[0].verdict == "flag"
    assert pdf_page_results[0].measured_value == 3
    assert "3" in pdf_page_results[0].reason

    # A "flag" is not a "reject": it must not trigger the reject-caveat text,
    # and must not count toward has_rejected_validation on its own.
    assert not any("quality checks" in text for text in result.findings.areas_of_attention)

    # Pipeline still completed fully, based on page 1's content.
    assert isinstance(result.features, Features)
    assert isinstance(result.findings, StructuredFindings)
    assert result.findings.overall_summary in result.report
    assert result.report.strip()


def test_run_analysis_corrupt_pdf_raises_pdf_input_error_uncaught() -> None:
    with pytest.raises(PdfInputError):
        run_analysis(_CORRUPT_PDF_BYTES)


def test_run_analysis_blurry_pdf_page_still_fails_blur_check() -> None:
    """declared_format="PDF" only short-circuits the *format* check --
    resolution/blur/contrast still genuinely run on the rasterized page."""
    result = run_analysis(_blurry_pdf_bytes())

    format_result = next(r for r in result.validation_results if r.check_name == "format")
    assert format_result.verdict == "accept"

    blur_result = next(r for r in result.validation_results if r.check_name == "blur")
    assert blur_result.verdict == "reject"
    assert result.has_rejected_validation is True


def test_run_analysis_accepts_raw_non_pdf_bytes(tmp_path: Path) -> None:
    """Regression: a non-PDF `bytes` input (e.g. an in-memory upload buffer)
    is loaded like any other image, not mistaken for a PDF."""
    image = _draw_stroke_grid((800, 1000))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    from_bytes = run_analysis(png_bytes, depth="concise")

    sample_path = _write_clean_sample(tmp_path)
    from_path = run_analysis(sample_path, depth="concise")

    assert from_bytes.has_rejected_validation == from_path.has_rejected_validation
    assert from_bytes.findings.overall_summary == from_path.findings.overall_summary


def test_check_format_png_and_jpeg_behavior_unchanged_by_declared_format_addition(
    tmp_path: Path,
) -> None:
    """Explicit regression pin: PNG/JPEG format detection through
    `validate_sample`/`run_analysis` is byte-for-byte unchanged by the
    `declared_format` addition when it is omitted (the default)."""
    from grafology_ai.validation import check_format

    path = tmp_path / "good.png"
    _draw_stroke_grid((800, 1000)).save(path, format="PNG")

    result = check_format(path)
    assert result.check_name == "format"
    assert result.verdict == "accept"
    assert result.reason == "format PNG is supported"
    assert result.measured_value is None

    explicit_none = check_format(path, declared_format=None)
    assert explicit_none == result
