"""Tests for grafology_ai.api (the minimal FastAPI ``POST /analyze`` app).

Uses FastAPI's ``TestClient`` (``fastapi.testclient``, backed by
``httpx``) rather than a running server process -- see
``grafology_ai.api``'s module docstring: running a live server is out of
scope, only the ``app`` object matters here.

Fixture image reuses the same deterministic-drawing approach as
``tests/test_run_analysis.py`` / ``tests/test_cli.py``.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from grafology_ai.api import app
from grafology_ai.cli import main as cli_main
from grafology_ai.run_analysis import run_analysis

BACKGROUND = 255
INK = 0

client = TestClient(app)


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


def _sample_png_bytes(size: tuple[int, int] = (800, 1000)) -> bytes:
    buffer = io.BytesIO()
    _draw_stroke_grid(size).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_sample(tmp_path: Path, name: str = "sample.png") -> Path:
    path = tmp_path / name
    path.write_bytes(_sample_png_bytes())
    return path


# --- PDF input helpers (Phase B3) -----------------------------------------------
#
# Built the same way as ``tests/test_pdf_input.py`` / ``tests/test_run_analysis.py``:
# Pillow's own ``Image.save(..., format="PDF")``, no extra PDF-generation
# dependency and no checked-in binary fixture.


def _pdf_bytes_from_images(images: list[Image.Image]) -> bytes:
    """Save `images` as a (possibly multi-page) in-memory PDF and return its bytes."""
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def _clean_pdf_bytes() -> bytes:
    """A single-page PDF that should pass every automated quality check."""
    return _pdf_bytes_from_images([_draw_stroke_grid((800, 1000))])


def _multi_page_pdf_bytes() -> bytes:
    pages = [
        _draw_stroke_grid((800, 1000)),
        _draw_stroke_grid((800, 1000), slant_deg=-10.0),
        _draw_stroke_grid((800, 1000), slant_deg=20.0),
    ]
    return _pdf_bytes_from_images(pages)


# A corrupt PDF: starts with the real ``%PDF-`` magic header (so `is_pdf`
# would say True) but is not a parseable PDF document beyond that -- the
# same construction ``tests/test_pdf_input.py`` uses.
_CORRUPT_PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nthis is not really a pdf body at all"


# --- happy path -------------------------------------------------------------------


def test_analyze_endpoint_returns_200_with_expected_shape() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"depth": "indepth", "sample_id": "api-001"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["depth"] == "indepth"
    assert body["sample_id"] == "api-001"
    assert isinstance(body["report_markdown"], str)
    assert "# Handwriting Analysis Support Report" in body["report_markdown"]
    assert "api-001" in body["report_markdown"]
    assert isinstance(body["overall_summary"], str) and body["overall_summary"]
    assert body["overall_summary"] in body["report_markdown"]
    assert isinstance(body["has_rejected_validation"], bool)
    assert isinstance(body["validation_results"], list) and body["validation_results"]
    for entry in body["validation_results"]:
        assert set(entry) == {"check_name", "verdict", "reason", "measured_value"}
        assert entry["verdict"] in ("accept", "reject", "flag")


def test_analyze_endpoint_default_depth_is_indepth() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["depth"] == "indepth"


def test_analyze_endpoint_concise_depth_reflected() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"depth": "concise"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["depth"] == "concise"
    assert "Concise" in body["report_markdown"]


def test_analyze_endpoint_quality_label_passed_through() -> None:
    # Near-uniform gray field: fails contrast/blur but is a valid PNG.
    buffer = io.BytesIO()
    Image.new("L", (600, 600), color=128).convert("RGB").save(buffer, format="PNG")

    plain_response = client.post(
        "/analyze", files={"image": ("gray.png", buffer.getvalue(), "image/png")}
    )
    flagged_response = client.post(
        "/analyze",
        files={"image": ("gray.png", buffer.getvalue(), "image/png")},
        data={"quality_label": "low"},
    )

    assert plain_response.json()["has_rejected_validation"] is True

    flagged_body = flagged_response.json()
    downgradable = {"blur", "contrast"}
    for entry in flagged_body["validation_results"]:
        if entry["check_name"] in downgradable:
            assert entry["verdict"] != "reject"


# --- bad input: clear 4xx, not a 500/raw traceback -------------------------------


def test_analyze_endpoint_rejects_unopenable_file() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("not_an_image.png", b"this is not image data", "image/png")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


# --- PDF input (Phase B3) --------------------------------------------------------


def test_analyze_endpoint_accepts_clean_pdf_upload() -> None:
    image_response = client.post(
        "/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
    )
    pdf_response = client.post(
        "/analyze",
        files={"image": ("sample.pdf", _clean_pdf_bytes(), "application/pdf")},
        data={"sample_id": "pdf-api-001"},
    )

    assert pdf_response.status_code == 200
    pdf_body = pdf_response.json()
    assert set(pdf_body) == set(image_response.json())
    assert pdf_body["sample_id"] == "pdf-api-001"
    assert "# Handwriting Analysis Support Report" in pdf_body["report_markdown"]
    assert isinstance(pdf_body["validation_results"], list) and pdf_body["validation_results"]
    for entry in pdf_body["validation_results"]:
        assert set(entry) == {"check_name", "verdict", "reason", "measured_value"}


def test_analyze_endpoint_rejects_corrupt_pdf_upload() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("corrupt.pdf", _CORRUPT_PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], str) and body["detail"]


def test_analyze_endpoint_multi_page_pdf_upload_flags_pdf_pages() -> None:
    response = client.post(
        "/analyze",
        files={"image": ("multi.pdf", _multi_page_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    pdf_page_entries = [
        entry for entry in body["validation_results"] if entry["check_name"] == "pdf_pages"
    ]
    assert len(pdf_page_entries) == 1
    assert pdf_page_entries[0]["verdict"] == "flag"
    assert pdf_page_entries[0]["measured_value"] == 3


# --- CLI/API consistency: same input -> same underlying analysis ----------------


def test_cli_and_api_produce_consistent_results_for_same_input(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)

    cli_exit_code = cli_main([str(sample_path), "--depth", "indepth"])
    assert cli_exit_code == 0
    cli_report = capsys.readouterr().out

    api_response = client.post(
        "/analyze",
        files={"image": (sample_path.name, sample_path.read_bytes(), "image/png")},
        data={"depth": "indepth"},
    )
    api_body = api_response.json()

    direct = run_analysis(sample_path, depth="indepth")

    # Both wrappers reflect the exact same overall_summary that
    # run_analysis() itself produces -- proving CLI and API are both thin
    # wrappers over the same function, not divergent implementations.
    assert direct.findings.overall_summary in cli_report
    assert direct.findings.overall_summary in api_body["report_markdown"]
    assert direct.findings.overall_summary == api_body["overall_summary"]
    assert api_body["depth"] == direct.findings.depth == "indepth"
