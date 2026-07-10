"""Automated image-quality validation for handwriting-sample images.

This subpackage implements the automatic image-quality acceptance criteria
from the client's technical proposal (adequate resolution, not blurry,
sufficient contrast, supported file format) as pixel-level checks that run
ahead of the not-yet-built pipeline/feature-analyzer/report-generator
stages. It does not perform dataset-manifest validation -- see
:mod:`grafology_ai.dataset` for that.
"""

from grafology_ai.validation.validators import (
    ValidationResult,
    check_blur,
    check_contrast,
    check_format,
    check_resolution,
    validate_sample,
)

__all__ = [
    "ValidationResult",
    "check_blur",
    "check_contrast",
    "check_format",
    "check_resolution",
    "validate_sample",
]
