"""Automated image-quality validators for handwriting-sample images.

The client's technical proposal specifies automatic image-quality
acceptance criteria that every sample should meet before later pipeline
stages (feature analyzer, report generator) run on it: adequate
resolution, not blurry, not cropped, sufficient contrast, and no heavy
shadows/aggressive filters. This module implements the checks that can be
assessed directly from pixel data: resolution, blur, contrast, and file
format.

The proposal also allows "dirty" samples through when the dataset manifest
already labels them ``quality="low"`` (see
:mod:`grafology_ai.dataset.schema`). :func:`validate_sample` implements
that downgrade rule: blur/contrast failures become a "flag" instead of a
"reject" for samples the manifest already declares low quality, while
resolution and format failures always reject, since an image that is too
small or in an unusable format cannot be salvaged by a quality label.

All checks accept either a :class:`PIL.Image.Image` or a path to an image
file (``str`` or ``os.PathLike``), for a single consistent interface across
this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

ImageInput = Union[Image.Image, str, "os.PathLike[str]"]

Verdict = Literal["accept", "reject", "flag"]

#: Formats this pipeline currently accepts. Handwriting samples supplied as
#: PDF are common in the client's raw intake but need to be rasterized to
#: a bitmap format first; see the TODO on :func:`check_format`.
SUPPORTED_FORMATS: tuple[str, ...] = ("JPEG", "PNG")

#: Practical proxy for "adequate resolution": real DPI metadata is
#: frequently missing from phone-camera photos (unlike flatbed scans,
#: which usually embed it), and even when present it is often an
#: uninformative firmware default rather than a measurement of the actual
#: capture. Instead of trusting DPI metadata, this check converts a
#: requested "DPI-equivalent" into a minimum pixel dimension using a fixed,
#: conservative reference physical size for a handwriting-sample crop (a
#: few inches on the shorter side -- these are typically crops of a
#: paragraph or a few lines, not a full page). The resulting default
#: threshold (150 * 3 = 450px on the shorter side) is deliberately modest:
#: it is meant to catch obviously-too-small images (thumbnails, aggressive
#: crops) rather than to certify print-shop-grade quality.
DEFAULT_MIN_DPI_EQUIVALENT = 150
RESOLUTION_REFERENCE_INCHES = 3.0

#: Below this Laplacian-variance value an image is considered blurred.
#: Laplacian variance measures the amount of high-frequency detail (edges)
#: in an image; sharp images with well-defined handwriting strokes have
#: high variance, while blurred images have their edges smoothed out and
#: consequently low variance. The absolute value is scale/content
#: dependent, so this default is a practical starting point, not a
#: universal constant.
DEFAULT_MIN_BLUR_VARIANCE = 150.0

#: Below this grayscale standard deviation an image is considered too
#: low-contrast (e.g. faint pencil on gray paper, or a washed-out scan).
DEFAULT_MIN_CONTRAST_STD_DEV = 25.0

# 3x3 discrete Laplacian kernel (4-neighborhood). Convolving a grayscale
# image with this kernel approximates the second spatial derivative, which
# responds strongly at edges and weakly across smooth/blurred regions.
_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of a single automated image-quality check.

    Attributes:
        check_name: Short identifier for the check, e.g. ``"resolution"``,
            ``"blur"``, ``"contrast"``, or ``"format"``.
        verdict: ``"accept"`` if the sample passes the check, ``"reject"``
            if it clearly fails and the sample should be excluded,
            or ``"flag"`` if it fails but is allowed through for manual
            review (e.g. a manifest-declared low-quality sample).
        reason: Human-readable explanation of the verdict.
        measured_value: The numeric quantity the verdict was based on
            (e.g. a pixel dimension, a variance, a standard deviation), or
            ``None`` for checks that are not based on a single scalar
            (e.g. format detection).
    """

    check_name: str
    verdict: Verdict
    reason: str
    measured_value: float | None = None


def _load_image(image: ImageInput) -> Image.Image:
    """Return a :class:`PIL.Image.Image` for either an image or a path."""
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a 2D float64 numpy array of grayscale values."""
    return np.asarray(image.convert("L"), dtype=np.float64)


def _convolve2d_same(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """"Same"-mode 2D convolution of ``array`` with a small ``kernel``.

    A minimal, dependency-free (no scipy) convolution implemented with
    numpy array slicing: the array is edge-padded and then, for each
    nonzero kernel coefficient, a shifted view of the padded array is
    accumulated. This is efficient for the small (3x3) kernels used in
    this module.
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode="edge")

    result = np.zeros_like(array, dtype=np.float64)
    height, width = array.shape
    for i in range(kernel_height):
        for j in range(kernel_width):
            coefficient = kernel[i, j]
            if coefficient == 0:
                continue
            result += coefficient * padded[i : i + height, j : j + width]
    return result


def check_resolution(
    image: ImageInput,
    min_dpi_equivalent: int = DEFAULT_MIN_DPI_EQUIVALENT,
) -> ValidationResult:
    """Check that the image's pixel dimensions are large enough to be legible.

    See the module-level docstring on :data:`DEFAULT_MIN_DPI_EQUIVALENT`
    for why this uses a pixel-dimension proxy rather than DPI metadata.
    """
    pil_image = _load_image(image)
    width, height = pil_image.size
    shorter_side = min(width, height)
    min_pixels = int(min_dpi_equivalent * RESOLUTION_REFERENCE_INCHES)

    if shorter_side < min_pixels:
        return ValidationResult(
            check_name="resolution",
            verdict="reject",
            reason=(
                f"shorter image dimension is {shorter_side}px, below the "
                f"{min_pixels}px minimum (~{min_dpi_equivalent} DPI over a "
                f"{RESOLUTION_REFERENCE_INCHES:g}in crop); handwriting "
                "strokes are unlikely to be legible at this resolution"
            ),
            measured_value=float(shorter_side),
        )
    return ValidationResult(
        check_name="resolution",
        verdict="accept",
        reason=(
            f"shorter image dimension is {shorter_side}px, meeting the "
            f"{min_pixels}px minimum"
        ),
        measured_value=float(shorter_side),
    )


def check_blur(
    image: ImageInput,
    min_variance: float = DEFAULT_MIN_BLUR_VARIANCE,
) -> ValidationResult:
    """Detect blur via a Laplacian-variance sharpness measure.

    The image is converted to grayscale and convolved with a discrete
    Laplacian kernel (see :data:`_LAPLACIAN_KERNEL`); the variance of the
    response is used as a sharpness score. Sharp images with well-defined
    edges (e.g. clean pen strokes) produce a high-variance response;
    blurred images, where edges are smoothed away, produce a low-variance
    response.
    """
    pil_image = _load_image(image)
    grayscale = _to_grayscale_array(pil_image)
    laplacian = _convolve2d_same(grayscale, _LAPLACIAN_KERNEL)
    variance = float(laplacian.var())

    if variance < min_variance:
        return ValidationResult(
            check_name="blur",
            verdict="reject",
            reason=(
                f"Laplacian variance {variance:.2f} is below the sharpness "
                f"threshold {min_variance:g}; the image appears blurred"
            ),
            measured_value=variance,
        )
    return ValidationResult(
        check_name="blur",
        verdict="accept",
        reason=(
            f"Laplacian variance {variance:.2f} meets the sharpness "
            f"threshold {min_variance:g}"
        ),
        measured_value=variance,
    )


def check_contrast(
    image: ImageInput,
    min_std_dev: float = DEFAULT_MIN_CONTRAST_STD_DEV,
) -> ValidationResult:
    """Measure contrast via the standard deviation of grayscale pixel values.

    A near-uniform image (e.g. faint pencil on gray paper, a washed-out
    scan, or heavy shadowing that flattens tonal range) has a low standard
    deviation; an image with a clear distinction between ink and page has
    a high one.
    """
    pil_image = _load_image(image)
    grayscale = _to_grayscale_array(pil_image)
    std_dev = float(grayscale.std())

    if std_dev < min_std_dev:
        return ValidationResult(
            check_name="contrast",
            verdict="reject",
            reason=(
                f"grayscale standard deviation {std_dev:.2f} is below the "
                f"contrast threshold {min_std_dev:g}; the image appears "
                "washed out or low-contrast"
            ),
            measured_value=std_dev,
        )
    return ValidationResult(
        check_name="contrast",
        verdict="accept",
        reason=(
            f"grayscale standard deviation {std_dev:.2f} meets the "
            f"contrast threshold {min_std_dev:g}"
        ),
        measured_value=std_dev,
    )


def check_format(image: ImageInput) -> ValidationResult:
    """Confirm the input is a supported image format (JPEG or PNG).

    Format is determined by actually attempting to identify the image
    (via Pillow), not merely by trusting a file extension, so a
    mislabeled or corrupt file is caught rather than waved through.

    TODO (future work, out of scope for this task): the client's raw
    intake also includes PDF documents. Supporting them will require a
    conversion step -- rasterizing each page to PNG/JPEG at a chosen DPI,
    likely via an added dependency such as ``pdf2image`` or ``pypdfium2``
    -- before any of the checks in this module can run on them. PDFs are
    not supported yet; this check rejects them like any other unsupported
    format.
    """
    try:
        pil_image = image if isinstance(image, Image.Image) else Image.open(image)
        detected_format = pil_image.format
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ValidationResult(
            check_name="format",
            verdict="reject",
            reason=f"could not identify an image format: {exc}",
            measured_value=None,
        )

    if detected_format is None:
        return ValidationResult(
            check_name="format",
            verdict="reject",
            reason=(
                "no format information available (the image was created "
                "in memory rather than loaded from a file); pass a file "
                "path, or an Image opened via Image.open(), to check format"
            ),
            measured_value=None,
        )

    if detected_format.upper() not in SUPPORTED_FORMATS:
        return ValidationResult(
            check_name="format",
            verdict="reject",
            reason=(
                f"format {detected_format!r} is not supported; only "
                f"{', '.join(SUPPORTED_FORMATS)} are supported (PDF "
                "conversion is a documented future TODO)"
            ),
            measured_value=None,
        )

    return ValidationResult(
        check_name="format",
        verdict="accept",
        reason=f"format {detected_format} is supported",
        measured_value=None,
    )


def validate_sample(
    image: ImageInput,
    quality_label: str | None = None,
) -> list[ValidationResult]:
    """Run all automated image-quality checks against a sample.

    Runs :func:`check_format`, :func:`check_resolution`, :func:`check_blur`,
    and :func:`check_contrast`, in that order, and returns one
    :class:`ValidationResult` per check.

    Per the client's technical proposal, "dirty" samples are allowed
    through only when the dataset manifest already labels them as
    ``quality="low"`` (see :mod:`grafology_ai.dataset.schema`). To
    implement that: if ``quality_label == "low"``, a blur or contrast
    check that would otherwise ``"reject"`` is downgraded to ``"flag"``
    instead, so the sample survives for manual review rather than being
    dropped outright. Resolution and format failures always ``"reject"``
    regardless of ``quality_label`` -- a sample that is too small or in an
    unusable format is not usable no matter how it is labeled.

    Args:
        image: A :class:`PIL.Image.Image` or a path to an image file.
        quality_label: The manifest's declared ``quality`` for this
            sample (``"high"``, ``"medium"``, ``"low"``), or ``None`` if
            unknown/not yet labeled.
    """
    pil_image = _load_image(image)

    results = [
        check_format(image),
        check_resolution(pil_image),
        check_blur(pil_image),
        check_contrast(pil_image),
    ]

    if quality_label == "low":
        downgradable = {"blur", "contrast"}
        results = [
            ValidationResult(
                check_name=result.check_name,
                verdict="flag",
                reason=(
                    result.reason
                    + " (downgraded from reject to flag: sample is "
                    "manifest-labeled quality=\"low\", so this defect "
                    "alone does not disqualify it)"
                ),
                measured_value=result.measured_value,
            )
            if result.check_name in downgradable and result.verdict == "reject"
            else result
            for result in results
        ]

    return results
