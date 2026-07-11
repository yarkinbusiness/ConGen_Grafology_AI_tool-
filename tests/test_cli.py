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

from PIL import Image, ImageDraw

from grafology_ai.cli import main
from grafology_ai.run_analysis import run_analysis

BACKGROUND = 255
INK = 0


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
