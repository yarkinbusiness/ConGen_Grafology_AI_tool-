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
