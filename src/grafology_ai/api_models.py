"""Pydantic response models for :mod:`grafology_ai.api`'s analyze endpoint.

Phase D2 ("Typed, versioned API contract") replaces the hand-built
``dict[str, Any]`` response :mod:`grafology_ai.api` used to return with a
set of named, typed Pydantic models. The point is entirely about the
*response contract*, not about re-deriving any data: every value placed
into these models still comes straight from
:meth:`grafology_ai.run_analysis.AnalysisResult` and its nested
``to_dict()`` methods (see that module and
``grafology_ai.analysis.features``/``grafology_ai.interpretation.interpret``/
``grafology_ai.validation.validators`` for the single source of truth each
field is copied from) -- this module only adds names, types, and
validation on top so FastAPI's generated OpenAPI schema documents the
response shape with real, named component schemas instead of a bare
``object``.

Structural choice: a fully typed :class:`FeaturesModel` (one field per
:class:`~grafology_ai.analysis.Features` attribute) rather than a loose
``dict[str, Any]``. ``Features`` only has 15 scalar fields plus one
``confidence`` dict, which is small enough that a typed model is not
laborious, and it gives callers of the generated OpenAPI schema real field
names/types/docs instead of an opaque blob -- worth it here given this
endpoint's response is meant to be a stable, documented contract (the
whole point of D2).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- validation results --------------------------------------------------------


class ValidationResultModel(BaseModel):
    """One automated image-quality check result.

    Mirrors :class:`grafology_ai.validation.validators.ValidationResult`
    field-for-field (via its ``to_dict()``); see that class's docstring
    for what each field means.
    """

    model_config = ConfigDict(extra="forbid")

    check_name: str
    verdict: Literal["accept", "reject", "flag"]
    reason: str
    measured_value: float | None = None


# --- features --------------------------------------------------------------------


class FeaturesModel(BaseModel):
    """Every measured, uninterpreted :class:`grafology_ai.analysis.Features` value.

    Field-for-field mirror of ``Features.to_dict()``; see
    :class:`grafology_ai.analysis.features.Features` for what each
    measurement means and how it was computed.
    """

    model_config = ConfigDict(extra="forbid")

    slant_angle_degrees: float
    stroke_width_mean: float
    stroke_width_std: float
    letter_size_estimate: float
    line_spacing_mean: float
    word_spacing_mean: float
    baseline_slope_degrees: float
    margin_left_px: float
    margin_right_px: float
    margin_top_px: float
    margin_bottom_px: float
    ink_density: float
    rhythm_regularity: float
    stroke_connectedness: float
    organization_score: float
    confidence: dict[str, float] = Field(
        description=(
            "Per-field confidence (0-1) for every field above except "
            "'confidence' itself -- see Features.confidence."
        )
    )


# --- findings ----------------------------------------------------------------------


class FindingModel(BaseModel):
    """One interpreted, per-indicator finding.

    Mirrors :class:`grafology_ai.interpretation.interpret.Finding`
    field-for-field, plus two values not stored on ``Finding`` itself but
    always derivable from it: ``label`` (the human-readable indicator name,
    looked up from :data:`grafology_ai.interpretation.INDICATOR_LABELS`)
    and ``confidence_label`` (the qualitative "high"/"medium"/"low" bucket
    for ``confidence``, via
    :func:`grafology_ai.report.generator.confidence_label`).
    """

    model_config = ConfigDict(extra="forbid")

    indicator: str
    label: str
    observation: str
    interpretation: str
    confidence: float
    confidence_label: Literal["high", "medium", "low"]


# --- top-level analyze response -----------------------------------------------------


class AnalyzeResponse(BaseModel):
    """Full response body of ``POST /v1/analyze`` (and its ``/analyze`` alias).

    A superset of the original ``POST /analyze`` response shape: every key
    that endpoint returned before D2 (``sample_id``, ``depth``,
    ``report_markdown``, ``overall_summary``, ``has_rejected_validation``,
    ``validation_results``) is still present with the same meaning, plus
    the newly exposed structured result: ``features`` (every measured
    value, see :class:`FeaturesModel`), ``findings`` (per-indicator
    interpretation, see :class:`FindingModel`), ``strengths``, and
    ``areas_of_attention``.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str | None
    depth: Literal["concise", "indepth"]
    report_markdown: str
    overall_summary: str
    has_rejected_validation: bool
    validation_results: list[ValidationResultModel]
    features: FeaturesModel
    findings: list[FindingModel]
    strengths: list[str]
    areas_of_attention: list[str]
