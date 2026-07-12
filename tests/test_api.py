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
from grafology_ai.interpretation import INDICATOR_LABELS, INDICATOR_ORDER
from grafology_ai.report.generator import confidence_label
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


# --- Phase D2: typed /v1/analyze response, backward-compatible /analyze alias ----


_EXPECTED_ANALYZE_RESPONSE_KEYS = {
    "sample_id",
    "depth",
    "report_markdown",
    "overall_summary",
    "has_rejected_validation",
    "validation_results",
    "features",
    "findings",
    "strengths",
    "areas_of_attention",
}


def _assert_body_matches_direct_result(body: dict, direct, sample_id: str | None) -> None:
    """Assert an /v1/analyze (or /analyze) response `body` matches a direct
    `run_analysis()` call's `AnalysisResult` (`direct`), field by field."""
    assert set(body) == _EXPECTED_ANALYZE_RESPONSE_KEYS

    assert body["sample_id"] == sample_id
    assert body["depth"] == direct.findings.depth
    assert body["report_markdown"] == direct.report
    assert body["overall_summary"] == direct.findings.overall_summary
    assert body["has_rejected_validation"] == direct.has_rejected_validation

    assert body["validation_results"] == [vr.to_dict() for vr in direct.validation_results]

    assert body["features"] == direct.features.to_dict()

    assert len(body["findings"]) == len(direct.findings.findings)
    for entry, finding in zip(body["findings"], direct.findings.findings):
        assert entry["indicator"] == finding.indicator
        assert entry["label"] == INDICATOR_LABELS[finding.indicator]
        assert entry["observation"] == finding.observation
        assert entry["interpretation"] == finding.interpretation
        assert entry["confidence"] == finding.confidence
        assert entry["confidence_label"] == confidence_label(finding.confidence)

    assert body["strengths"] == list(direct.findings.strengths)
    assert body["areas_of_attention"] == list(direct.findings.areas_of_attention)


def test_v1_analyze_returns_200_with_full_shape_matching_direct_run_analysis() -> None:
    image_bytes = _sample_png_bytes()

    response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", image_bytes, "image/png")},
        data={"depth": "indepth", "sample_id": "v1-001"},
    )
    assert response.status_code == 200
    body = response.json()

    direct = run_analysis(image_bytes, depth="indepth", sample_id="v1-001")
    _assert_body_matches_direct_result(body, direct, sample_id="v1-001")


def test_analyze_alias_returns_full_expanded_shape_superset_of_legacy_keys() -> None:
    """`/analyze` (the legacy path) now returns the same full D2 shape as
    `/v1/analyze`, which is a strict superset of the original pre-D2 keys
    every pre-existing test above already exercises unmodified."""
    image_bytes = _sample_png_bytes()

    response = client.post(
        "/analyze",
        files={"image": ("sample.png", image_bytes, "image/png")},
        data={"depth": "indepth", "sample_id": "legacy-001"},
    )
    assert response.status_code == 200
    body = response.json()

    legacy_keys = {
        "sample_id",
        "depth",
        "report_markdown",
        "overall_summary",
        "has_rejected_validation",
        "validation_results",
    }
    assert legacy_keys.issubset(set(body))

    direct = run_analysis(image_bytes, depth="indepth", sample_id="legacy-001")
    _assert_body_matches_direct_result(body, direct, sample_id="legacy-001")


def test_openapi_schema_has_named_component_schemas_for_response_models() -> None:
    schema = client.get("/openapi.json").json()
    component_names = set(schema["components"]["schemas"])

    assert "AnalyzeResponse" in component_names
    assert "FeaturesModel" in component_names
    assert "FindingModel" in component_names
    assert "ValidationResultModel" in component_names

    # And the schema is actually used (not just declared) by /v1/analyze's
    # 200 response -- not merely a generic `object`/`dict[str, Any]`.
    analyze_response_schema = schema["paths"]["/v1/analyze"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert "AnalyzeResponse" in analyze_response_schema.get("$ref", "")


def test_v1_analyze_invalid_depth_returns_structured_422() -> None:
    response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"depth": "bogus"},
    )
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    assert body["detail"]


def test_v1_analyze_invalid_quality_label_returns_structured_422() -> None:
    response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"quality_label": "bogus"},
    )
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    assert body["detail"]


def test_v1_analyze_valid_quality_label_values_still_accepted() -> None:
    for label in ("high", "medium", "low"):
        response = client.post(
            "/v1/analyze",
            files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
            data={"quality_label": label},
        )
        assert response.status_code == 200, label


def test_v1_analyze_confidence_label_agrees_with_generator_thresholds() -> None:
    """Exercises at least two different confidence buckets and confirms each
    finding's `confidence_label` agrees with `report/generator.py`'s actual
    thresholds applied to that same finding's numeric `confidence`."""
    # A clean, high-signal sample plus a near-blank one give a spread of
    # confidence values (and therefore confidence_label buckets) across
    # their combined findings.
    grid_response = client.post(
        "/v1/analyze",
        files={"image": ("grid.png", _sample_png_bytes(), "image/png")},
        data={"depth": "indepth"},
    )
    blank_buffer = io.BytesIO()
    Image.new("L", (600, 600), color=255).convert("RGB").save(blank_buffer, format="PNG")
    blank_response = client.post(
        "/v1/analyze",
        files={"image": ("blank.png", blank_buffer.getvalue(), "image/png")},
        data={"depth": "indepth"},
    )

    assert grid_response.status_code == 200
    assert blank_response.status_code == 200

    all_findings = grid_response.json()["findings"] + blank_response.json()["findings"]
    labels_seen = set()
    for entry in all_findings:
        expected_label = confidence_label(entry["confidence"])
        assert entry["confidence_label"] == expected_label
        labels_seen.add(entry["confidence_label"])

    assert len(labels_seen) >= 2


def test_v1_analyze_concise_and_indepth_depths_both_work() -> None:
    concise_response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"depth": "concise"},
    )
    indepth_response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
        data={"depth": "indepth"},
    )

    assert concise_response.status_code == 200
    assert indepth_response.status_code == 200

    concise_body = concise_response.json()
    indepth_body = indepth_response.json()

    assert concise_body["depth"] == "concise"
    assert indepth_body["depth"] == "indepth"

    assert len(indepth_body["findings"]) == len(INDICATOR_ORDER)
    assert len(concise_body["findings"]) < len(indepth_body["findings"])
    assert len(concise_body["findings"]) == 4


def test_v1_analyze_pdf_upload_returns_same_full_response_shape() -> None:
    image_response = client.post(
        "/v1/analyze",
        files={"image": ("sample.png", _sample_png_bytes(), "image/png")},
    )
    pdf_response = client.post(
        "/v1/analyze",
        files={"image": ("sample.pdf", _clean_pdf_bytes(), "application/pdf")},
        data={"sample_id": "pdf-v1-001"},
    )

    assert pdf_response.status_code == 200
    pdf_body = pdf_response.json()
    assert set(pdf_body) == set(image_response.json()) == _EXPECTED_ANALYZE_RESPONSE_KEYS
    assert pdf_body["sample_id"] == "pdf-v1-001"
    assert isinstance(pdf_body["features"], dict)
    assert isinstance(pdf_body["findings"], list) and pdf_body["findings"]


def test_v1_analyze_rejects_unopenable_file_with_400_and_detail() -> None:
    response = client.post(
        "/v1/analyze",
        files={"image": ("not_an_image.png", b"this is not image data", "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], str) and body["detail"]
