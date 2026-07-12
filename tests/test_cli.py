"""Tests for grafology_ai.cli (the ``grafology-analyze`` console script).

Exercises the actual CLI entry point (``grafology_ai.cli.main``), not just
``grafology_ai.run_analysis.run_analysis`` directly -- both by calling
``main()`` with an explicit argv list and by invoking it out-of-process
via ``python -m grafology_ai.cli`` through ``subprocess``, so the
``[project.scripts]`` console-script wiring and stdlib ``argparse``
handling are both actually covered.

Fixture image reuses the same deterministic-drawing approach as
``tests/test_run_analysis.py``.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from grafology_ai.cli import main
from grafology_ai.run_analysis import run_analysis

BACKGROUND = 255
INK = 0

# Mirrors tests/test_report_pdf.py's DISCLAIMER_CORE_PHRASE / _extract_pdf_text.
DISCLAIMER_CORE_PHRASE = "not a diagnosis"


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


def _write_sample(tmp_path: Path, name: str = "sample.png") -> Path:
    path = tmp_path / name
    _draw_stroke_grid((800, 1000)).save(path, format="PNG")
    return path


# --- PDF input helpers (Phase B3) -----------------------------------------------
#
# Built the same way as ``tests/test_pdf_input.py`` / ``tests/test_run_analysis.py``:
# Pillow's own ``Image.save(..., format="PDF")``, no extra PDF-generation
# dependency and no checked-in binary fixture.


def _write_pdf_sample(tmp_path: Path, name: str = "sample.pdf") -> Path:
    path = tmp_path / name
    _draw_stroke_grid((800, 1000)).save(path, format="PDF")
    return path


# A corrupt PDF: starts with the real ``%PDF-`` magic header (so `is_pdf`
# would say True) but is not a parseable PDF document beyond that -- the
# same construction ``tests/test_pdf_input.py`` uses.
_CORRUPT_PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nthis is not really a pdf body at all"


def _write_corrupt_pdf(tmp_path: Path, name: str = "corrupt.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(_CORRUPT_PDF_BYTES)
    return path


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Full text of every page of `pdf_bytes`, concatenated in page order.

    Mirrors ``tests/test_report_pdf.py``'s ``_extract_pdf_text`` helper
    (see that file's docstring for why whitespace is collapsed).
    """
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        parts = []
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    parts.append(textpage.get_text_range())
                finally:
                    textpage.close()
            finally:
                page.close()
    finally:
        document.close()
    return " ".join(" ".join(part.split()) for part in parts)


# --- in-process: main() with an explicit argv, stdout path ---------------------


def test_cli_prints_report_to_stdout(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)

    exit_code = main([str(sample_path), "--depth", "indepth", "--sample-id", "cli-001"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Handwriting Analysis Support Report" in captured.out
    assert "cli-001" in captured.out
    assert "## Overall Summary" in captured.out

    # Matches what calling run_analysis() directly produces for the same input.
    direct = run_analysis(sample_path, depth="indepth", sample_id="cli-001")
    assert direct.findings.overall_summary in captured.out


# --- in-process: main() writing to --output ------------------------------------


def test_cli_writes_report_to_output_file(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "report.md"

    exit_code = main(
        [str(sample_path), "--depth", "concise", "--output", str(output_path)]
    )

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Handwriting Analysis Support Report" in content
    assert "Concise" in content

    # Confirms nothing was accidentally also dumped to stdout instead.
    captured = capsys.readouterr()
    assert "# Handwriting Analysis Support Report" not in captured.out
    assert str(output_path) in captured.out

    direct = run_analysis(sample_path, depth="concise")
    assert direct.findings.overall_summary in content


def test_cli_output_file_matches_stdout_content_for_same_args(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "report.md"

    main([str(sample_path), "--sample-id", "match-001", "--output", str(output_path)])
    written = output_path.read_text(encoding="utf-8")

    capsys.readouterr()  # clear
    main([str(sample_path), "--sample-id", "match-001"])
    printed = capsys.readouterr().out

    # printed report has a trailing newline from print(); strip for comparison.
    assert printed.rstrip("\n") == written.rstrip("\n")


# --- bad path: clear error, non-zero exit, no raw traceback --------------------


def test_cli_nonexistent_image_gives_clear_error_and_nonzero_exit(capsys) -> None:
    exit_code = main(["/nonexistent/path/does-not-exist.png"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does-not-exist.png" in captured.err
    assert "Error" in captured.err
    # No raw traceback leaked to the user.
    assert "Traceback" not in captured.err


def test_cli_unopenable_file_gives_clear_error_and_nonzero_exit(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"this is not image data")

    exit_code = main([str(bad_path)])

    assert exit_code != 0


# --- PDF input (Phase B3) --------------------------------------------------------


def test_cli_clean_pdf_prints_report_to_stdout(tmp_path: Path, capsys) -> None:
    pdf_path = _write_pdf_sample(tmp_path)

    exit_code = main([str(pdf_path), "--depth", "indepth", "--sample-id", "pdf-cli-001"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Handwriting Analysis Support Report" in captured.out
    assert "pdf-cli-001" in captured.out

    direct = run_analysis(pdf_path, depth="indepth", sample_id="pdf-cli-001")
    assert direct.findings.overall_summary in captured.out


def test_cli_corrupt_pdf_gives_clear_error_and_nonzero_exit(tmp_path: Path, capsys) -> None:
    corrupt_pdf_path = _write_corrupt_pdf(tmp_path)

    exit_code = main([str(corrupt_pdf_path)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    stderr_lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(stderr_lines) == 1
    assert "Error" in captured.err
    assert "Traceback" not in captured.err


def test_cli_help_mentions_pdf_support() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "PDF" in completed.stdout
    assert "JPEG or PNG" not in completed.stdout


# --- out-of-process: exercises the actual `python -m grafology_ai.cli` entry ----


def test_cli_help_via_subprocess() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "grafology-analyze" in completed.stdout
    assert "--depth" in completed.stdout


def test_cli_subprocess_end_to_end(tmp_path: Path) -> None:
    sample_path = _write_sample(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.cli", str(sample_path), "--depth", "concise"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "# Handwriting Analysis Support Report" in completed.stdout
    assert "Concise" in completed.stdout


def test_cli_subprocess_bad_path_nonzero_exit() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "grafology_ai.cli", "/nonexistent/nope.png"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "Error" in completed.stderr
    assert "Traceback" not in completed.stderr


# --- Phase C2: --format {md,pdf} --------------------------------------------------


def test_cli_format_pdf_with_output_writes_pdf_report(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "r.pdf"

    exit_code = main([str(sample_path), "--format", "pdf", "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    pdf_bytes = output_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    text = _extract_pdf_text(pdf_bytes)
    assert DISCLAIMER_CORE_PHRASE in text

    captured = capsys.readouterr()
    assert str(output_path) in captured.out


def test_cli_format_pdf_without_output_errors_with_exit_code_2(tmp_path: Path, capsys) -> None:
    sample_path = _write_sample(tmp_path)

    exit_code = main([str(sample_path), "--format", "pdf"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    stderr_lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(stderr_lines) == 1
    assert "Error" in captured.err
    assert "Traceback" not in captured.err


def test_cli_pdf_extension_infers_pdf_format(tmp_path: Path) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "r.pdf"

    exit_code = main([str(sample_path), "--output", str(output_path)])

    assert exit_code == 0
    pdf_bytes = output_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")


def test_cli_default_output_without_format_flag_is_markdown(tmp_path: Path) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "report.md"

    exit_code = main([str(sample_path), "--output", str(output_path)])

    assert exit_code == 0
    content_bytes = output_path.read_bytes()
    assert not content_bytes.startswith(b"%PDF")
    content = output_path.read_text(encoding="utf-8")
    assert "# Handwriting Analysis Support Report" in content


def test_cli_explicit_format_md_overrides_pdf_extension_inference(tmp_path: Path) -> None:
    sample_path = _write_sample(tmp_path)
    output_path = tmp_path / "report.pdf"

    exit_code = main([str(sample_path), "--format", "md", "--output", str(output_path)])

    assert exit_code == 0
    content_bytes = output_path.read_bytes()
    assert not content_bytes.startswith(b"%PDF")
    content = output_path.read_text(encoding="utf-8")
    assert "# Handwriting Analysis Support Report" in content
