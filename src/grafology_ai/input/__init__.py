"""PDF sniffing and rasterization for handwriting-sample intake.

The client's raw intake includes PDF documents alongside JPEG/PNG images.
This subpackage rasterizes PDF pages to bitmap images (see
:mod:`grafology_ai.input.pdf` for the full approach and its documented
simplifications) so that later pipeline stages -- all of which operate on
pixel data -- can run on them. Wiring this into
:mod:`grafology_ai.validation.validators`'s format check and the rest of
the pipeline/CLI/API is separate follow-up work (see ``docs/roadmap.md``,
phase B2), deliberately out of scope here.
"""

from grafology_ai.input.pdf import (
    DEFAULT_PDF_RASTER_DPI,
    PdfInput,
    PdfInputError,
    PdfRasterization,
    is_pdf,
    rasterize_pdf,
)

__all__ = [
    "DEFAULT_PDF_RASTER_DPI",
    "PdfInput",
    "PdfInputError",
    "PdfRasterization",
    "is_pdf",
    "rasterize_pdf",
]
