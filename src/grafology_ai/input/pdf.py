"""PDF sniffing and rasterization for handwriting-sample intake.

The client's raw intake includes PDF documents alongside JPEG/PNG images
(see the TODO on
:func:`grafology_ai.validation.validators.check_format`, which this
module is the eventual implementation of): a PDF has to be converted to a
bitmap page image before any of the validation/analysis pipeline stages
-- all of which operate on pixel data -- can run on it.

This module deliberately does two things and nothing more:

1. :func:`is_pdf` sniffs the actual leading bytes of an input for the
   ``%PDF-`` magic header, exactly mirroring
   :func:`~grafology_ai.validation.validators.check_format`'s "never
   trust a file extension, always inspect real content" stance.
2. :func:`rasterize_pdf` renders exactly one page of a PDF to a
   :class:`PIL.Image.Image`, using `pypdfium2
   <https://github.com/pypdfium2-team/pypdfium2>`_ (a Python binding to
   Google's PDFium rendering engine). ``pypdfium2`` was chosen over
   ``pdf2image`` because it ships PDFium as a prebuilt wheel dependency
   with no external Poppler binary to separately install and manage --
   simpler to deploy alongside this pipeline's other pure-Python/Pillow
   dependencies.

Per the client's intake-handling decision (see ``docs/roadmap.md``,
phase B1): only the *first* page of a multi-page PDF is rasterized by
default (``page_index=0``); a caller that wants a later page passes
``page_index`` explicitly, and :attr:`PdfRasterization.page_count` lets a
caller detect (and, in a later pipeline stage, flag) multi-page intake
rather than silently dropping the extra pages. The default rasterization
resolution is 300 DPI (:data:`DEFAULT_PDF_RASTER_DPI`), a standard
print/scan-quality target.

This module does not perform any of the image-quality checks in
:mod:`grafology_ai.validation.validators` (resolution/blur/contrast) on
the rasterized output -- wiring a rasterized page back through that
validation layer is separate follow-up work (see ``docs/roadmap.md``,
phase B2), deliberately out of scope here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import pypdfium2 as pdfium
from PIL import Image

#: A PDF source: either a path to a file on disk (``str``/``os.PathLike``)
#: or raw in-memory bytes (``bytes``/``bytearray``/``memoryview``) -- both
#: accepted uniformly across this module's functions, since callers may
#: have an on-disk PDF or an in-memory upload buffer (e.g. from
#: ``python-multipart``), the same flexibility
#: :data:`grafology_ai.validation.validators.ImageInput` gives image
#: inputs.
PdfInput = Union[bytes, bytearray, memoryview, str, "os.PathLike[str]"]

#: The literal magic header every valid PDF file begins with.
_PDF_MAGIC = b"%PDF-"

#: Number of leading bytes inspected when sniffing for :data:`_PDF_MAGIC`.
#: The PDF spec tolerates some non-conformant content (e.g. a leading
#: byte-order mark, or bytes prepended by a lossy transfer) before the
#: header, as long as it appears within roughly the first 1024 bytes of
#: the file -- readers are expected to scan for it rather than require it
#: at byte 0, so this module does the same rather than only checking
#: ``source[:5]``.
_MAGIC_SNIFF_WINDOW_BYTES = 1024

#: Default rasterization resolution, in DPI, used by :func:`rasterize_pdf`
#: when the caller doesn't specify one -- 300 DPI is a standard
#: print/scan-quality target and matches the client's decision for this
#: pipeline (see ``docs/roadmap.md``, phase B1).
DEFAULT_PDF_RASTER_DPI = 300

#: PDF page geometry is expressed in points, fixed at 72 points per inch
#: regardless of the DPI a page is eventually rasterized at; ``dpi /
#: _PDF_POINTS_PER_INCH`` is the linear pixels-per-point scale factor
#: :func:`rasterize_pdf` passes to PDFium's renderer.
_PDF_POINTS_PER_INCH = 72.0


class PdfInputError(Exception):
    """A PDF input could not be rasterized.

    Raised for corrupt/unparseable PDF data, an encrypted PDF opened
    without (or with the wrong) credentials, a zero-page PDF, or a
    ``page_index`` outside the document's page range. Always carries a
    clear, human-readable message describing which of those problems
    occurred, rather than letting a raw ``pypdfium2`` exception (or an
    unhandled ``IndexError``) bubble up to the caller -- mirroring how
    :mod:`grafology_ai.validation.validators` always surfaces clear,
    typed errors rather than raw library exceptions.
    """


@dataclass(frozen=True)
class PdfRasterization:
    """The result of rasterizing one page of a PDF document.

    Attributes:
        image: The rendered page as an RGB :class:`PIL.Image.Image`.
        page_count: Total number of pages in the *source* PDF document
            (not merely a flag for "1" vs "more than 1"), so a caller can
            detect -- and, in a later pipeline stage, flag for manual
            follow-up -- multi-page intake per the client's page-1-only
            default (see the module docstring).
        dpi_used: The DPI :attr:`image` was actually rendered at (echoes
            back whatever ``dpi`` :func:`rasterize_pdf` was called with).
    """

    image: Image.Image
    page_count: int
    dpi_used: int


def _read_leading_bytes(source: PdfInput, n: int) -> bytes:
    """Return up to the first `n` bytes of `source`, from bytes or a path.

    Returns an empty ``bytes`` (rather than raising) if `source` is a
    path that does not exist or cannot be opened, so :func:`is_pdf` can
    treat an unreadable path as simply "not a PDF" rather than crashing.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source[:n])
    try:
        with open(source, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def is_pdf(source: PdfInput) -> bool:
    """Return whether `source` is actually PDF content, by sniffing bytes.

    Reads the first :data:`_MAGIC_SNIFF_WINDOW_BYTES` bytes of `source`
    (a path or raw bytes) and checks whether the :data:`_PDF_MAGIC`
    header appears within them -- never trusts a file extension, matching
    :func:`grafology_ai.validation.validators.check_format`'s "never
    trust the extension" stance. Returns ``False`` (rather than raising)
    for an unreadable path or for content that simply isn't a PDF.
    """
    head = _read_leading_bytes(source, _MAGIC_SNIFF_WINDOW_BYTES)
    return _PDF_MAGIC in head


def _open_document(source: PdfInput) -> pdfium.PdfDocument:
    """Open `source` as a `pypdfium2.PdfDocument`, wrapping failures in `PdfInputError`."""
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return pdfium.PdfDocument(bytes(source))
        return pdfium.PdfDocument(str(source))
    except FileNotFoundError as exc:
        raise PdfInputError(f"PDF source not found: {exc}") from exc
    except pdfium.PdfiumError as exc:
        raise PdfInputError(
            f"could not open PDF (corrupt, encrypted, or not a valid PDF): {exc}"
        ) from exc


def rasterize_pdf(
    source: PdfInput,
    dpi: int = DEFAULT_PDF_RASTER_DPI,
    page_index: int = 0,
) -> PdfRasterization:
    """Rasterize one page of a PDF document to an RGB :class:`PIL.Image.Image`.

    Args:
        source: A path to a PDF file, or raw PDF bytes (see
            :data:`PdfInput`). Not required to already be confirmed a PDF
            by :func:`is_pdf` -- an invalid source raises
            :class:`PdfInputError` here regardless.
        dpi: Rasterization resolution. Defaults to
            :data:`DEFAULT_PDF_RASTER_DPI`.
        page_index: Zero-based index of the page to rasterize. Defaults
            to ``0`` (the first page), per the client's page-1-only intake
            decision (see the module docstring) -- callers that want a
            later page pass this explicitly.

    Returns:
        A :class:`PdfRasterization` carrying the rendered page image,
        the source document's total page count, and the DPI used.

    Raises:
        PdfInputError: If the PDF cannot be opened (corrupt, encrypted,
            not a real PDF), has zero pages, if `page_index` is negative
            or is not less than the document's page count, or if the
            requested page fails to load/render.
    """
    document = _open_document(source)
    try:
        page_count = len(document)
        if page_count == 0:
            raise PdfInputError("PDF has zero pages; there is nothing to rasterize")
        if page_index < 0 or page_index >= page_count:
            raise PdfInputError(
                f"page_index {page_index} is out of range for a "
                f"{page_count}-page PDF (valid range: 0..{page_count - 1})"
            )

        try:
            page = document[page_index]
        except pdfium.PdfiumError as exc:
            raise PdfInputError(f"could not load page {page_index}: {exc}") from exc

        try:
            scale = dpi / _PDF_POINTS_PER_INCH
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        except pdfium.PdfiumError as exc:
            raise PdfInputError(f"could not render page {page_index}: {exc}") from exc
        finally:
            page.close()

        return PdfRasterization(image=image, page_count=page_count, dpi_used=dpi)
    finally:
        document.close()
