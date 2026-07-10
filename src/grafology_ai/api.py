"""Minimal HTTP API: proves the pipeline is "web-ready" without a web app.

Per the client's technical proposal ("No complete web app in this
phase... Test tool in a quick form: CLI script or notebook") and the
overall plan's requirement to "prove web-readiness", this module is
deliberately small: one endpoint, ``POST /analyze``, that accepts an
uploaded image plus the same optional parameters
:func:`grafology_ai.run_analysis.run_analysis` takes, calls it, and
returns a JSON-serializable response. It is a thin wrapper exactly like
:mod:`grafology_ai.cli` -- both call the same
:func:`~grafology_ai.run_analysis.run_analysis` function, so they can
never diverge on how a sample is actually analyzed (see that module's
docstring).

Running a live server is explicitly out of scope here (only the ``app``
object matters for "proving web-readiness"); tests exercise this module
via FastAPI's ``TestClient`` (``fastapi.testclient``), never a running
process. Accordingly, this module only depends on ``fastapi`` itself, not
an ASGI server such as ``uvicorn``.
"""

from __future__ import annotations

import io
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from grafology_ai.run_analysis import run_analysis

app = FastAPI(
    title="Grafology AI Tool -- Analysis API",
    description=(
        "Minimal HTTP API wrapping grafology_ai.run_analysis.run_analysis. "
        "A support tool for professional graphologists, not a diagnostic tool."
    ),
    version="0.1.0",
)


@app.post("/analyze")
async def analyze_endpoint(
    image: UploadFile = File(..., description="Handwriting-sample image file (JPEG or PNG)."),
    depth: Literal["concise", "indepth"] = Form("indepth"),
    quality_label: str | None = Form(None),
    sample_id: str | None = Form(None),
) -> dict[str, Any]:
    """Run :func:`run_analysis` on an uploaded image and return a JSON summary.

    Request: ``multipart/form-data`` with an ``image`` file field plus
    optional ``depth`` (``"concise"``/``"indepth"``, default
    ``"indepth"``), ``quality_label`` (``"high"``/``"medium"``/``"low"``),
    and ``sample_id`` form fields -- the same parameters
    :func:`~grafology_ai.run_analysis.run_analysis` takes.

    Response body (200 on success)::

        {
            "sample_id": str | null,
            "depth": "concise" | "indepth",
            "report_markdown": str,
            "overall_summary": str,
            "has_rejected_validation": bool,
            "validation_results": [
                {"check_name": str, "verdict": "accept"|"reject"|"flag",
                 "reason": str, "measured_value": float | null},
                ...
            ],
        }

    A file that cannot be opened/identified as an image (corrupt data,
    unsupported format) yields ``400 Bad Request`` with a descriptive
    ``detail`` message rather than a raw traceback / 500.
    """
    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify '{image.filename}' as a supported image file: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not open uploaded file '{image.filename}': {exc}",
        ) from exc

    result = run_analysis(
        pil_image,
        depth=depth,
        quality_label=quality_label,
        sample_id=sample_id,
    )

    return {
        "sample_id": sample_id,
        "depth": result.findings.depth,
        "report_markdown": result.report,
        "overall_summary": result.findings.overall_summary,
        "has_rejected_validation": result.has_rejected_validation,
        "validation_results": [
            {
                "check_name": vr.check_name,
                "verdict": vr.verdict,
                "reason": vr.reason,
                "measured_value": vr.measured_value,
            }
            for vr in result.validation_results
        ],
    }
