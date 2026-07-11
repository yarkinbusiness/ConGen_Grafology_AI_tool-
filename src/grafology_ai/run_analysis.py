"""Single entry point: validate -> analyze -> interpret -> report, in one call.

This module is the "Minimal interface" step from the client's technical
proposal ("No complete web app in this phase ... Test tool in a quick
form: CLI script or notebook") and the piece that "proves web-readiness":
:func:`run_analysis` is a single, plain function that chains the four
already-existing pipeline stages --

1. :func:`grafology_ai.validation.validate_sample`
2. :func:`grafology_ai.analysis.analyze`
3. :func:`grafology_ai.interpretation.interpret`
4. :func:`grafology_ai.report.generate_report`

-- into one call over a single handwriting-sample image. It is written
once here and used twice: :mod:`grafology_ai.cli` (a stdlib-``argparse``
console script) and :mod:`grafology_ai.api` (a minimal FastAPI app) are
both thin wrappers around this exact function, not divergent
reimplementations of the pipeline.

Design decision: what happens on a "reject" validation verdict
-----------------------------------------------------------------

:func:`grafology_ai.validation.validate_sample` can return one or more
:class:`~grafology_ai.validation.ValidationResult` entries with
``verdict="reject"`` (e.g. the image is too small, too blurry, too low
contrast, or an unsupported format). :func:`run_analysis` deliberately
does **not** treat a reject verdict as fatal:

- It never raises or short-circuits because of a rejected check --
  automated image-quality checks are a signal for the human reviewer, not
  a hard gate on whether *any* output gets produced. The rest of the
  pipeline (feature analysis, interpretation, report rendering) still
  runs in full, so a graphologist reviewing a flagged sample still gets a
  complete, readable report rather than nothing.
- It never *silently hides* a reject verdict either. Two things make it
  visible:

  1. :attr:`AnalysisResult.validation_results` always carries every
     :class:`~grafology_ai.validation.ValidationResult` verbatim (see
     also the :attr:`AnalysisResult.has_rejected_validation` /
     :attr:`AnalysisResult.rejected_validation_checks` convenience
     properties), so any caller inspecting the result programmatically
     can see exactly which checks failed and why.
  2. When at least one check rejects, :func:`run_analysis` appends one
     hedged, non-diagnostic caveat sentence (see
     :func:`_quality_caveat_text`) to the interpreted
     :class:`~grafology_ai.interpretation.StructuredFindings`'
     ``areas_of_attention`` list *before* rendering the report -- i.e. it
     reuses the interpretation layer's own existing "may benefit from
     closer review" extension point rather than patching rendered
     markdown text after the fact. This keeps
     :mod:`grafology_ai.report.generator` (which this task must not
     modify) as the single place report markdown is actually rendered:
     :attr:`AnalysisResult.report` is always exactly
     ``generate_report(AnalysisResult.findings, sample_id=...)``, so
     ``findings`` and ``report`` can never disagree about whether a
     quality caveat was noted, and callers that re-render (e.g.
     :func:`grafology_ai.report.save_report`) from
     :attr:`AnalysisResult.findings` reproduce the same text.

quality_label is passed straight through to
:func:`~grafology_ai.validation.validate_sample`, which already knows how
to downgrade blur/contrast rejects to "flag" for manifest-declared
``quality="low"`` samples (see its docstring) -- :func:`run_analysis`
does not duplicate that logic.
"""

from __future__ import annotations

import dataclasses
import io
import os
from dataclasses import dataclass
from typing import Union

from PIL import Image

from grafology_ai.analysis import Features, analyze
from grafology_ai.input.pdf import is_pdf, rasterize_pdf
from grafology_ai.interpretation import DISALLOWED_TERMS, Depth, StructuredFindings, interpret
from grafology_ai.report import generate_report
from grafology_ai.validation import ValidationResult, validate_sample

ImageInput = Union[Image.Image, bytes, str, "os.PathLike[str]"]


@dataclass(frozen=True)
class _LoadedSample:
    """Internal: the in-memory image plus format/page metadata `run_analysis` needs.

    Attributes:
        image: The loaded/rasterized :class:`PIL.Image.Image`.
        declared_format: ``"PDF"`` when `image` came from rasterizing a
            PDF page (so :func:`~grafology_ai.validation.validate_sample`
            should trust that declared format rather than re-deriving
            ``image.format``, which is ``None`` for a rasterized page);
            ``None`` for a normally-loaded image, preserving existing
            behavior exactly.
        page_count: Total page count of the source PDF, or ``1`` for a
            non-PDF input.
    """

    image: Image.Image
    declared_format: str | None
    page_count: int


def _load_sample(image: ImageInput) -> _LoadedSample:
    """Load `image` into memory, transparently rasterizing PDF input.

    - An already-loaded :class:`PIL.Image.Image` is used as-is
      (``declared_format=None``, ``page_count=1``) -- unchanged from
      before PDF support existed.
    - A path or raw ``bytes`` is first sniffed with
      :func:`grafology_ai.input.pdf.is_pdf`; if it is a PDF, page 1 is
      rasterized via :func:`grafology_ai.input.pdf.rasterize_pdf` (at the
      default DPI) and ``declared_format="PDF"`` /
      the source's real ``page_count`` are recorded.
      :class:`~grafology_ai.input.pdf.PdfInputError` is allowed to
      propagate uncaught here -- mapping it to a CLI/API-friendly error is
      separate follow-up work, not this function's job (mirroring how
      ``Image.open``'s own exceptions already propagate uncaught below).
    - Otherwise it is loaded normally via ``Image.open`` (from a path, or
      from an in-memory buffer for raw non-PDF ``bytes``),
      ``declared_format=None``, ``page_count=1``.
    """
    if isinstance(image, Image.Image):
        return _LoadedSample(image=image, declared_format=None, page_count=1)

    if is_pdf(image):
        rasterization = rasterize_pdf(image)
        return _LoadedSample(
            image=rasterization.image,
            declared_format="PDF",
            page_count=rasterization.page_count,
        )

    if isinstance(image, (bytes, bytearray, memoryview)):
        pil_image = Image.open(io.BytesIO(image))
    else:
        pil_image = Image.open(image)
    return _LoadedSample(image=pil_image, declared_format=None, page_count=1)


def _assert_language_is_disciplined(text: str) -> None:
    """Defense-in-depth check that ``text`` avoids the project's disallowed terms.

    Mirrors ``grafology_ai.interpretation.interpret._assert_language_is_disciplined``:
    every piece of narrative text this module originates itself (currently
    just :func:`_quality_caveat_text`) is a fixed template, so this should
    never fire in practice, but it makes a future accidental template edit
    fail loudly rather than silently shipping disallowed language.
    """
    lowered = text.lower()
    for term in DISALLOWED_TERMS:
        if term in lowered:
            raise ValueError(
                f"Generated quality-caveat text contains disallowed term {term!r}: {text!r}"
            )


def _quality_caveat_text(rejected: list[ValidationResult]) -> str:
    """Hedged, non-diagnostic ``areas_of_attention`` text for rejected checks.

    Written to the same cautious standard as
    ``grafology_ai.interpretation.interpret``'s own generated text (see
    :data:`~grafology_ai.interpretation.DISALLOWED_TERMS`): it names which
    automated checks did not pass and notes that this may reduce the
    reliability of the reading, without diagnostic language, absolute
    claims, or "weakness"/"problem" framing.
    """
    check_names = ", ".join(result.check_name for result in rejected)
    return (
        f"Automated image-quality checks did not pass for this sample ({check_names}); "
        "this may reduce the reliability of the measurements above, and the "
        "original image may benefit from being recaptured or rescanned before "
        "this reading is relied upon."
    )


@dataclass(frozen=True)
class AnalysisResult:
    """Bundled, internally-consistent output of one :func:`run_analysis` call.

    Attributes:
        validation_results: Every automated image-quality check result
            (see :func:`grafology_ai.validation.validate_sample`), in the
            order that function returns them. Always present in full,
            regardless of verdict -- see the module docstring's "Design
            decision" section for why a "reject" verdict does not stop
            the rest of the pipeline from running.
        features: The raw measured :class:`~grafology_ai.analysis.Features`
            (see :func:`grafology_ai.analysis.analyze`).
        findings: The interpreted
            :class:`~grafology_ai.interpretation.StructuredFindings` (see
            :func:`grafology_ai.interpretation.interpret`). If any
            ``validation_results`` entry rejected, this includes one
            extra ``areas_of_attention`` entry noting that (see the
            module docstring).
        report: The rendered markdown report string -- always exactly
            ``generate_report(findings, sample_id=...)``, so it is
            guaranteed consistent with ``findings`` (e.g.
            ``findings.overall_summary`` always appears verbatim in
            ``report``).
    """

    validation_results: list[ValidationResult]
    features: Features
    findings: StructuredFindings
    report: str

    @property
    def rejected_validation_checks(self) -> list[ValidationResult]:
        """The subset of :attr:`validation_results` with ``verdict == "reject"``."""
        return [result for result in self.validation_results if result.verdict == "reject"]

    @property
    def has_rejected_validation(self) -> bool:
        """Whether any automated image-quality check rejected this sample."""
        return len(self.rejected_validation_checks) > 0


def run_analysis(
    image: ImageInput,
    depth: Depth = "indepth",
    quality_label: str | None = None,
    sample_id: str | None = None,
) -> AnalysisResult:
    """Run the full per-sample pipeline on ``image`` and return a bundled result.

    This is the single function both :func:`grafology_ai.cli.main` and
    :mod:`grafology_ai.api`'s ``POST /analyze`` endpoint call -- see the
    module docstring for why, and for the documented "reject" verdict
    behavior.

    Args:
        image: A :class:`PIL.Image.Image`, a path (``str`` or
            ``os.PathLike``) to an image or PDF file, or raw ``bytes``
            (e.g. an in-memory upload buffer). Path/bytes input is
            sniffed for the PDF magic header (see
            :func:`grafology_ai.input.pdf.is_pdf`); if it is a PDF, page 1
            is rasterized (see :func:`grafology_ai.input.pdf.rasterize_pdf`)
            and used for the rest of the pipeline, and a
            ``check_name="pdf_pages"`` ``"flag"`` result is added to
            :attr:`AnalysisResult.validation_results` when the source PDF
            has more than one page (see the module docstring). Loading a
            non-PDF path/bytes or rasterizing a PDF can both raise (e.g.
            ``FileNotFoundError``/``PIL.UnidentifiedImageError`` for a
            missing/corrupt image,
            :class:`grafology_ai.input.pdf.PdfInputError` for a
            corrupt/unparseable PDF) -- callers that want a friendly,
            non-traceback error message for that case should catch those
            (see :mod:`grafology_ai.cli`, which does exactly this).
        depth: Interpretation/report depth, ``"concise"`` or
            ``"indepth"`` (default). Passed straight through to
            :func:`grafology_ai.interpretation.interpret`.
        quality_label: The manifest-declared quality for this sample
            (``"high"``, ``"medium"``, ``"low"``), or ``None`` if
            unknown. Passed straight through to
            :func:`grafology_ai.validation.validate_sample`.
        sample_id: Optional sample identifier included in the rendered
            report header (see
            :func:`grafology_ai.report.generate_report`).

    Returns:
        An :class:`AnalysisResult` bundling every stage's output.
    """
    loaded = _load_sample(image)
    pil_image = loaded.image

    validation_results = validate_sample(
        pil_image, quality_label=quality_label, declared_format=loaded.declared_format
    )
    if loaded.page_count > 1:
        validation_results = [
            *validation_results,
            ValidationResult(
                check_name="pdf_pages",
                verdict="flag",
                reason=(
                    f"the source PDF has {loaded.page_count} pages; only page 1 "
                    "was rasterized and analyzed, the remaining pages were not "
                    "examined"
                ),
                measured_value=float(loaded.page_count),
            ),
        ]
    features = analyze(pil_image)
    findings = interpret(features, depth=depth)

    rejected = [result for result in validation_results if result.verdict == "reject"]
    if rejected:
        caveat = _quality_caveat_text(rejected)
        _assert_language_is_disciplined(caveat)
        findings = dataclasses.replace(
            findings,
            areas_of_attention=[*findings.areas_of_attention, caveat],
        )

    report = generate_report(findings, sample_id=sample_id)

    return AnalysisResult(
        validation_results=validation_results,
        features=features,
        findings=findings,
        report=report,
    )
