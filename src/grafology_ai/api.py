"""Minimal HTTP API: proves the pipeline is "web-ready" without a web app.

Per the client's technical proposal ("No complete web app in this
phase... Test tool in a quick form: CLI script or notebook") and the
overall plan's requirement to "prove web-readiness", this module is
deliberately small: one primary endpoint, ``POST /v1/analyze`` (plus a
``POST /analyze`` backward-compatible alias -- see below), that accepts
an uploaded image plus the same optional parameters
:func:`grafology_ai.run_analysis.run_analysis` takes, calls it, and
returns a JSON-serializable response. It is a thin wrapper exactly like
:mod:`grafology_ai.cli` -- both call the same
:func:`~grafology_ai.run_analysis.run_analysis` function, so they can
never diverge on how a sample is actually analyzed (see that module's
docstring).

Typed, versioned API contract (Phase D2)
-----------------------------------------

The response body is a :class:`~grafology_ai.api_models.AnalyzeResponse`
Pydantic model (see :mod:`grafology_ai.api_models`) rather than a hand-built
``dict``, so FastAPI's generated OpenAPI schema documents the response
shape with real, named component schemas. ``POST /v1/analyze`` is the new,
primary route; ``POST /analyze`` is kept as a thin backward-compatible
alias -- both routes are served by the exact same handler function
(registered twice via a stacked ``@app.post`` decorator), so their
responses can never diverge. The response is a strict superset of the
original ``POST /analyze`` shape: every key that endpoint returned before
D2 (``sample_id``, ``depth``, ``report_markdown``, ``overall_summary``,
``has_rejected_validation``, ``validation_results``) is still present
with the same meaning, plus the newly exposed ``features``, ``findings``,
``strengths``, and ``areas_of_attention``.

Running a live server is explicitly out of scope here (only the ``app``
object matters for "proving web-readiness"); tests exercise this module
via FastAPI's ``TestClient`` (``fastapi.testclient``), never a running
process. Accordingly, this module only depends on ``fastapi`` (and the
``pydantic`` it already pulls in transitively) rather than an ASGI server
such as ``uvicorn``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from grafology_ai.api_models import (
    AnalyzeResponse,
    FeaturesModel,
    FindingModel,
    ValidationResultModel,
)
from grafology_ai.input.pdf import PdfInputError
from grafology_ai.interpretation import INDICATOR_LABELS
from grafology_ai.report.generator import confidence_label
from grafology_ai.run_analysis import AnalysisResult, run_analysis

app = FastAPI(
    title="Grafology AI Tool -- Analysis API",
    description=(
        "Minimal HTTP API wrapping grafology_ai.run_analysis.run_analysis. "
        "A support tool for professional graphologists, not a diagnostic tool."
    ),
    version="0.1.0",
)


def _build_response(
    result: AnalysisResult, *, sample_id: str | None
) -> AnalyzeResponse:
    """Assemble the typed :class:`AnalyzeResponse` for one `run_analysis` result.

    Every value here is copied straight from `result` (or a `to_dict()` of
    one of its nested dataclasses) -- see `grafology_ai.run_analysis.AnalysisResult`
    and its nested `Features`/`StructuredFindings`/`ValidationResult`
    `to_dict()` methods for the single source of truth. `has_rejected_validation`
    is computed from the real `AnalysisResult.has_rejected_validation`
    property (not re-derived from a dict), per D1's note that this
    property was deliberately excluded from `AnalysisResult.to_dict()`.
    `label`/`confidence_label` on each finding are the only two values not
    already present on `Finding` itself: `label` is looked up from
    `grafology_ai.interpretation.INDICATOR_LABELS`, and `confidence_label`
    reuses `grafology_ai.report.generator.confidence_label` verbatim rather
    than reimplementing its thresholds.
    """
    return AnalyzeResponse(
        sample_id=sample_id,
        depth=result.findings.depth,
        report_markdown=result.report,
        overall_summary=result.findings.overall_summary,
        has_rejected_validation=result.has_rejected_validation,
        validation_results=[
            ValidationResultModel(**vr.to_dict()) for vr in result.validation_results
        ],
        features=FeaturesModel(**result.features.to_dict()),
        findings=[
            FindingModel(
                indicator=finding.indicator,
                label=INDICATOR_LABELS[finding.indicator],
                observation=finding.observation,
                interpretation=finding.interpretation,
                confidence=finding.confidence,
                confidence_label=confidence_label(finding.confidence),
            )
            for finding in result.findings.findings
        ],
        strengths=list(result.findings.strengths),
        areas_of_attention=list(result.findings.areas_of_attention),
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    image: UploadFile = File(
        ..., description="Handwriting-sample file (JPEG, PNG, or PDF)."
    ),
    depth: Literal["concise", "indepth"] = Form("indepth"),
    quality_label: Literal["high", "medium", "low"] | None = Form(None),
    sample_id: str | None = Form(None),
) -> AnalyzeResponse:
    """Run :func:`run_analysis` on an uploaded file and return a typed JSON summary.

    Mounted at both ``POST /v1/analyze`` (the primary, versioned route)
    and ``POST /analyze`` (a backward-compatible alias -- both paths are
    served by this exact function, so their responses can never diverge).

    Request: ``multipart/form-data`` with an ``image`` file field (an
    image or a PDF -- see :func:`grafology_ai.input.pdf.is_pdf`, which
    :func:`~grafology_ai.run_analysis.run_analysis` uses internally to
    detect and rasterize PDF uploads; for a PDF, only the first page is
    analyzed) plus optional ``depth`` (``"concise"``/``"indepth"``,
    default ``"indepth"``), ``quality_label``
    (``"high"``/``"medium"``/``"low"``), and ``sample_id`` form fields --
    the same parameters :func:`~grafology_ai.run_analysis.run_analysis`
    takes. An invalid ``depth`` or ``quality_label`` value (anything
    outside those literal sets) yields FastAPI's standard structured
    ``422`` validation-error response, not a hand-rolled error.

    Response body (200 on success): see
    :class:`grafology_ai.api_models.AnalyzeResponse` for the full typed
    shape (``sample_id``, ``depth``, ``report_markdown``,
    ``overall_summary``, ``has_rejected_validation``,
    ``validation_results``, ``features``, ``findings``, ``strengths``,
    ``areas_of_attention``).

    A file that cannot be opened/identified as an image (corrupt data,
    unsupported format) or a PDF that cannot be rasterized (corrupt,
    encrypted, zero-page) yields ``400 Bad Request`` with a descriptive
    ``detail`` message rather than a raw traceback / 500.
    """
    raw_bytes = await image.read()
    try:
        result = run_analysis(
            raw_bytes,
            depth=depth,
            quality_label=quality_label,
            sample_id=sample_id,
        )
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify '{image.filename}' as a supported image file: {exc}",
        ) from exc
    except PdfInputError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read '{image.filename}' as a valid PDF file: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not open uploaded file '{image.filename}': {exc}",
        ) from exc

    return _build_response(result, sample_id=sample_id)
