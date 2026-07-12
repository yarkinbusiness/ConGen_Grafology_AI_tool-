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

PDF report delivery (Phase D3)
-------------------------------

An ``output_format`` form field (``"json"`` default | ``"pdf"``) lets a
caller ask for the same analyzed result rendered as a PDF report (via
:func:`grafology_ai.report.pdf.render_report_pdf`) instead of the JSON
:class:`~grafology_ai.api_models.AnalyzeResponse` body -- e.g. a
``multipart/form-data`` ``application/pdf`` response with a
``Content-Disposition: attachment`` header, suitable for a browser to
download directly. This only changes how a *successfully analyzed*
result is returned; upload/read errors (``400``) and parameter validation
(``422``, including an invalid ``output_format`` itself) are unaffected.
Because the endpoint can now return either a Pydantic model or a raw
:class:`fastapi.responses.Response`, its return type annotation is a
``Union`` of both -- FastAPI passes a returned ``Response`` straight
through untouched (bypassing ``response_model`` serialization) while
still validating/serializing a returned :class:`AnalyzeResponse` against
the declared ``response_model``, so the ``"json"`` path's OpenAPI schema
and runtime behavior are completely unchanged from D2.

Running a live server is explicitly out of scope here (only the ``app``
object matters for "proving web-readiness"); tests exercise this module
via FastAPI's ``TestClient`` (``fastapi.testclient``), never a running
process. Accordingly, this module only depends on ``fastapi`` (and the
``pydantic`` it already pulls in transitively) rather than an ASGI server
such as ``uvicorn``.
"""

from __future__ import annotations

import re
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
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
from grafology_ai.report.pdf import render_report_pdf
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


#: Characters kept as-is in a `Content-Disposition` filename derived from a
#: caller-supplied `sample_id`; everything else collapses to `"_"`. `sample_id`
#: is free text (see `api_models`/`run_analysis`), so it may contain quotes,
#: path separators, control characters, or other bytes that are unsafe or
#: ambiguous inside an HTTP header value -- this is a simple allow-list, not
#: an attempt at exhaustive RFC 6266 correctness, which is more than this
#: internal support tool's filename needs (see `_pdf_filename`'s docstring).
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _pdf_filename(sample_id: str | None) -> str:
    """Build a safe-ish `report-{name}.pdf` filename for the PDF `output_format`.

    `name` is `sample_id` with any character outside a small allow-list
    (letters, digits, `.`, `_`, `-`) collapsed to `"_"`, so a `sample_id`
    containing a `"`, a path separator, or other header-unsafe characters
    can never break out of the `Content-Disposition` header value or be
    interpreted as a path by a naive client. Falls back to the literal
    `"sample"` when `sample_id` is absent/blank or sanitizes down to
    nothing. This is deliberately simple allow-list sanitization, not a
    full RFC 6266 `filename*`/percent-encoding implementation -- adequate
    for a support tool's downloaded-report filename, not a hardened
    multi-tenant file-serving path.
    """
    name = _SAFE_FILENAME_CHARS.sub("_", sample_id).strip("_") if sample_id else ""
    return f"report-{name or 'sample'}.pdf"


@app.post("/v1/analyze", response_model=AnalyzeResponse)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    image: UploadFile = File(
        ..., description="Handwriting-sample file (JPEG, PNG, or PDF)."
    ),
    depth: Literal["concise", "indepth"] = Form("indepth"),
    quality_label: Literal["high", "medium", "low"] | None = Form(None),
    sample_id: str | None = Form(None),
    output_format: Literal["json", "pdf"] = Form("json"),
) -> AnalyzeResponse | Response:
    """Run :func:`run_analysis` on an uploaded file and return the result.

    Mounted at both ``POST /v1/analyze`` (the primary, versioned route)
    and ``POST /analyze`` (a backward-compatible alias -- both paths are
    served by this exact function, so their responses can never diverge).

    Request: ``multipart/form-data`` with an ``image`` file field (an
    image or a PDF -- see :func:`grafology_ai.input.pdf.is_pdf`, which
    :func:`~grafology_ai.run_analysis.run_analysis` uses internally to
    detect and rasterize PDF uploads; for a PDF, only the first page is
    analyzed) plus optional ``depth`` (``"concise"``/``"indepth"``,
    default ``"indepth"``), ``quality_label``
    (``"high"``/``"medium"``/``"low"``), ``sample_id``, and
    ``output_format`` (``"json"``/``"pdf"``, default ``"json"``) form
    fields -- the same parameters :func:`~grafology_ai.run_analysis.run_analysis`
    takes, plus ``output_format`` selecting how the *successfully analyzed*
    result is returned (see below). An invalid ``depth``, ``quality_label``,
    or ``output_format`` value (anything outside those literal sets) yields
    FastAPI's standard structured ``422`` validation-error response, not a
    hand-rolled error.

    Response body (200 on success) depends on ``output_format``:

    - ``"json"`` (the default -- unchanged from before Phase D3): a JSON
      :class:`grafology_ai.api_models.AnalyzeResponse` body (``sample_id``,
      ``depth``, ``report_markdown``, ``overall_summary``,
      ``has_rejected_validation``, ``validation_results``, ``features``,
      ``findings``, ``strengths``, ``areas_of_attention``).
    - ``"pdf"``: an ``application/pdf`` body -- the same analysis result
      rendered via :func:`grafology_ai.report.pdf.render_report_pdf`
      instead -- with a ``Content-Disposition: attachment;
      filename="report-{name}.pdf"`` header (``{name}`` derived from
      ``sample_id`` if given, else ``"sample"``; see :func:`_pdf_filename`).

    Format only affects how a *successful* analysis is returned; it has no
    effect on upload/read errors below. A file that cannot be
    opened/identified as an image (corrupt data, unsupported format) or a
    PDF that cannot be rasterized (corrupt, encrypted, zero-page) yields
    ``400 Bad Request`` with a descriptive ``detail`` message rather than a
    raw traceback / 500, for either ``output_format``.
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

    if output_format == "pdf":
        pdf_bytes = render_report_pdf(result.findings, sample_id=sample_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{_pdf_filename(sample_id)}"'
            },
        )

    return _build_response(result, sample_id=sample_id)
