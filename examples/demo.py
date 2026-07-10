#!/usr/bin/env python3
"""Golden-path demo of the grafology_ai pipeline, end to end.

Run it with::

    python examples/demo.py

after installing the package (``pip install -e ".[dev]"``). No arguments,
no manual setup, and no files outside this repository are required: the
script generates its own sample handwriting-like image with
:func:`grafology_ai.dataset.fixtures.generate_fixture_dataset` (the same
synthetic generator the pipeline's own tests and hardening runs use, since
no real client handwriting dataset exists yet -- see
``docs/pipeline_runbook.md``), then runs the full
:func:`grafology_ai.run_analysis.run_analysis` pipeline against it at both
supported depths (``"concise"`` and ``"indepth"``), printing a short
walkthrough of what happened at each stage plus the full rendered markdown
report.

This is meant to be the "show the client" moment: a single, runnable
script that demonstrates validation, feature analysis, interpretation, and
report generation working together on one sample, without needing a real
dataset, a running server, or any manual setup.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from grafology_ai.dataset.fixtures import generate_fixture_dataset
from grafology_ai.report import save_report
from grafology_ai.run_analysis import run_analysis

#: Fixed seed so the "generated" sample -- and therefore this script's
#: printed output -- is reproducible from one run to the next (see
#: `generate_fixture_dataset`'s docstring: same `count`/`seed` always
#: produces byte-identical images).
DEMO_SEED = 20260710

_RULE = "-" * 78
_DOUBLE_RULE = "=" * 78


def _print_header(text: str) -> None:
    print()
    print(_DOUBLE_RULE)
    print(text)
    print(_DOUBLE_RULE)


def _print_stage(text: str) -> None:
    print()
    print(_RULE)
    print(text)
    print(_RULE)


def main() -> int:
    _print_header("ConGen Grafology AI Tool -- golden-path demo")
    print(
        "This script demonstrates the full support-tool pipeline on one\n"
        "synthetic handwriting-sample image: validate -> analyze -> interpret\n"
        "-> generate_report. It is a support tool for professional\n"
        "graphologists, not a diagnostic tool -- see every report's\n"
        "disclaimer section below."
    )

    work_dir = Path(tempfile.mkdtemp(prefix="grafology_demo_"))
    dataset_dir = work_dir / "fixture_dataset"

    # --- Stage 0: obtain a sample image -------------------------------------
    _print_stage("Stage 0: generate a sample handwriting-like image")
    print(
        "No real client handwriting dataset exists yet (see\n"
        "docs/pipeline_runbook.md), so this demo generates one procedurally\n"
        "rendered sample the same way the test suite and hardening runs do,\n"
        f"via generate_fixture_dataset(seed={DEMO_SEED})."
    )
    entries = generate_fixture_dataset(dataset_dir, count=1, seed=DEMO_SEED)
    entry = entries[0]
    image_path = dataset_dir / "images" / f"{entry.sample_id}.png"
    print(
        f"Generated sample_id={entry.sample_id!r} "
        f"(acquisition_method={entry.acquisition_method!r}, "
        f"manifest quality={entry.quality!r}) at {image_path}"
    )

    for depth in ("concise", "indepth"):
        _print_header(f"Running the pipeline at depth={depth!r}")

        result = run_analysis(
            image_path,
            depth=depth,
            quality_label=entry.quality,
            sample_id=entry.sample_id,
        )

        # --- Stage 1: validation ---------------------------------------------
        _print_stage("Stage 1: automated image-quality validation (validate_sample)")
        for vr in result.validation_results:
            print(f"  [{vr.verdict.upper():6s}] {vr.check_name:10s} -- {vr.reason}")
        if result.has_rejected_validation:
            print(
                "  -> At least one check rejected this sample; run_analysis() "
                "still runs the full pipeline and appends a caveat to the "
                "report instead of stopping (see run_analysis.py's docstring)."
            )
        else:
            print("  -> All automated checks passed.")

        # --- Stage 2: feature analysis ----------------------------------------
        _print_stage("Stage 2: classical image-processing feature analysis (analyze)")
        f = result.features
        print(f"  slant_angle_degrees   = {f.slant_angle_degrees:+.1f} deg")
        print(f"  stroke_width_mean     = {f.stroke_width_mean:.1f} px  (pressure proxy)")
        print(f"  stroke_width_std      = {f.stroke_width_std:.1f} px  (pressure consistency)")
        print(f"  letter_size_estimate  = {f.letter_size_estimate:.1f} px")
        print(f"  line_spacing_mean     = {f.line_spacing_mean:.1f} px")
        print(f"  word_spacing_mean     = {f.word_spacing_mean:.1f} px")
        print(f"  baseline_slope_degrees= {f.baseline_slope_degrees:+.1f} deg")
        print(
            "  margins (L/R/T/B) px  = "
            f"({f.margin_left_px:.0f}, {f.margin_right_px:.0f}, "
            f"{f.margin_top_px:.0f}, {f.margin_bottom_px:.0f})"
        )
        print(f"  ink_density           = {f.ink_density:.3f}")

        # --- Stage 3: interpretation --------------------------------------------
        _print_stage("Stage 3: cautious, non-diagnostic interpretation (interpret)")
        print(f"  depth used            = {result.findings.depth!r}")
        print(f"  indicators covered    = {len(result.findings.findings)}")
        print(f"  strengths noted       = {len(result.findings.strengths)}")
        print(f"  areas of attention    = {len(result.findings.areas_of_attention)}")
        print(f"  overall_summary       = {result.findings.overall_summary}")

        # --- Stage 4: report rendering -----------------------------------------
        _print_stage("Stage 4: markdown report rendering (generate_report / save_report)")
        report_path = work_dir / f"report_{entry.sample_id}_{depth}.md"
        save_report(result.findings, report_path, sample_id=entry.sample_id)
        print(f"  Report written to: {report_path}")
        print(f"  Report length: {len(result.report)} characters")
        print()
        print("  ---- full rendered report follows ----")
        print()
        print(result.report)

    _print_header("Demo complete")
    print(
        "Both depths ran end to end with no errors. Generated image and\n"
        f"saved reports are under: {work_dir}\n"
        "(a temporary directory -- safe to delete; re-run this script any\n"
        "time to regenerate the same reproducible sample and reports)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
