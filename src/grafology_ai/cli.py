"""``grafology-analyze`` console script: run one handwriting-sample analysis.

This is the "Test tool in a quick form: CLI script" the client's technical
proposal calls for. It is a thin ``argparse`` wrapper around
:func:`grafology_ai.run_analysis.run_analysis` -- the CLI does not
duplicate any pipeline logic, it only parses arguments, calls
``run_analysis``, and presents the result (see
:mod:`grafology_ai.run_analysis`'s module docstring for why
:mod:`grafology_ai.api` is built the same way, over the same function).

Usage::

    grafology-analyze path/to/sample.png
    grafology-analyze path/to/sample.pdf
    grafology-analyze path/to/sample.png --depth concise
    grafology-analyze path/to/sample.png --output report.md
    grafology-analyze path/to/sample.png --output report.pdf
    grafology-analyze path/to/sample.png --format pdf --output report.pdf
    grafology-analyze path/to/sample.png --quality-label low --sample-id S001

By default the report is rendered as markdown (``--format md``). Pass
``--format pdf`` (which requires ``--output``, since raw PDF bytes cannot
usefully be printed to a terminal) to render a PDF report instead. As a
convenience, if ``--output`` is given a path ending in ``.pdf`` and
``--format`` is *not* explicitly specified, PDF output is inferred
automatically -- an explicit ``--format`` always wins over this
inference.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from PIL import UnidentifiedImageError

from grafology_ai.input.pdf import PdfInputError
from grafology_ai.report import save_report, save_report_pdf
from grafology_ai.run_analysis import run_analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grafology-analyze",
        description=(
            "Run the full handwriting-sample analysis pipeline (automated "
            "image-quality validation, classical feature analysis, "
            "interpretation, and markdown report generation) on a single "
            "handwriting sample (image or PDF), end to end. This is a "
            "support tool for professional graphologists, not a "
            "diagnostic tool."
        ),
    )
    parser.add_argument(
        "image",
        help=(
            "Path to the handwriting-sample file (JPEG, PNG, or PDF). "
            "For a PDF, only the first page is analyzed."
        ),
    )
    parser.add_argument(
        "--depth",
        choices=("concise", "indepth"),
        default="indepth",
        help="Interpretation/report depth (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Write the report to this file instead of printing to stdout "
            "(printing is only supported for markdown output). If PATH "
            "ends in '.pdf' and --format is not given explicitly, PDF "
            "output is inferred automatically."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("md", "pdf"),
        default=None,
        help=(
            "Report output format (default: 'md', unless inferred as "
            "'pdf' from a '.pdf' --output path). 'pdf' requires --output."
        ),
    )
    parser.add_argument(
        "--quality-label",
        choices=("high", "medium", "low"),
        default=None,
        help=(
            "Manifest-declared quality label for this sample, if known. "
            "'low' downgrades blur/contrast rejections to a flag for "
            "manual review instead of a hard reject."
        ),
    )
    parser.add_argument(
        "--sample-id",
        metavar="TEXT",
        default=None,
        help="Sample identifier to include in the generated report header.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``grafology-analyze`` console script.

    Returns a process exit code (0 on success) rather than calling
    ``sys.exit`` itself, so it can be invoked directly and asserted on in
    tests without a ``SystemExit`` needing to be caught.

    A missing or unopenable image/PDF file produces a single, clear
    ``stderr`` message and a non-zero exit code -- never a raw traceback.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve the effective format: an explicit `--format` always wins;
    # otherwise infer "pdf" from a `.pdf`-suffixed `--output` path (see
    # module docstring), falling back to the historical default, "md".
    if args.format is not None:
        output_format = args.format
    elif args.output and args.output.lower().endswith(".pdf"):
        output_format = "pdf"
    else:
        output_format = "md"

    if output_format == "pdf" and not args.output:
        # Exit code 2 matches argparse's own convention for a usage error
        # (e.g. a missing required argument), distinguishing this from
        # exit code 1, used below for pipeline/IO errors encountered
        # while actually running the analysis.
        print(
            "Error: --format pdf requires --output PATH (PDF bytes cannot "
            "be printed to stdout).",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_analysis(
            args.image,
            depth=args.depth,
            quality_label=args.quality_label,
            sample_id=args.sample_id,
        )
    except FileNotFoundError:
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        return 1
    except UnidentifiedImageError:
        print(
            f"Error: could not identify '{args.image}' as a supported image file "
            "(it may be corrupt or in an unsupported format).",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"Error: could not open image file '{args.image}': {exc}", file=sys.stderr)
        return 1
    except PdfInputError as exc:
        print(f"Error: could not read PDF file '{args.image}': {exc}", file=sys.stderr)
        return 1

    if args.output:
        # `result.findings` already carries any quality-caveat text
        # `run_analysis` added (see its module docstring), so both
        # `save_report(result.findings, ...)` and
        # `save_report_pdf(result.findings, ...)` reproduce `result.report`'s
        # content exactly (same `findings`, same `sample_id`) -- just
        # rendered to a different format/sink.
        if output_format == "pdf":
            output_path = save_report_pdf(result.findings, args.output, sample_id=args.sample_id)
        else:
            output_path = save_report(result.findings, args.output, sample_id=args.sample_id)
        print(f"Report written to {output_path}")
    else:
        print(result.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
