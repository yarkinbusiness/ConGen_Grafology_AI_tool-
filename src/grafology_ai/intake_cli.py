"""``grafology-intake`` console script: day-zero dataset onboarding check.

This is the "is this dataset usable?" companion to `grafology-analyze`
(which analyzes one sample) -- a thin `argparse` wrapper around
`grafology_ai.intake.run_intake`, following the same conventions as
`grafology_ai.cli` (the `grafology-analyze` console script): parse
arguments, call the one real function, present the result, no pipeline
logic duplicated here.

Usage::

    grafology-intake path/to/raw_dataset
    grafology-intake path/to/raw_dataset --output report.json

Exit code is `0` if the dataset has zero rejected samples and no
manifest-level errors, non-zero otherwise -- so this is scriptable in
automation (e.g. a CI check on a freshly dropped client dataset), not just
a human-readable report.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from grafology_ai.intake import IntakeReport, run_intake


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grafology-intake",
        description=(
            "Check whether a raw handwriting-sample dataset (manifest.json + "
            "images/) is usable before any training work starts: ingest it, "
            "run the same automated image-quality checks the rest of the "
            "pipeline uses, and report accepted/flagged/rejected counts plus "
            "an actionable fix-it checklist. This is a validator/checker, "
            "not a labeling tool and not a dataset-collection tool."
        ),
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to the raw dataset directory (containing manifest.json and images/).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Also write a detailed JSON dump of the full intake report "
            "(counts, per-rejected-sample failed checks, manifest errors, "
            "and the fix-it checklist) to this file."
        ),
    )
    return parser


def _print_summary(report: IntakeReport) -> None:
    print(f"Dataset: {report.dataset_dir}")

    if report.manifest_errors:
        print("Manifest-level errors (dataset could not be fully checked):")
        for error in report.manifest_errors:
            print(f"  - [{error.error_type}] {error.message}")
        print()

    print(
        f"Total samples: {report.total_samples}  "
        f"(accepted: {report.accepted}, flagged: {report.flagged}, "
        f"rejected: {report.rejected})"
    )

    if report.rejected_samples:
        print("\nRejected samples:")
        for sample in report.rejected_samples:
            reasons = "; ".join(
                f"{check['check_name']}: {check['reason']}" for check in sample.failed_checks
            )
            print(f"  - {sample.sample_id}: {reasons}")

    print("\nFix-it checklist:")
    if report.fix_it_checklist:
        for item in report.fix_it_checklist:
            print(f"  - {item}")
    else:
        print("  No issues found.")

    print()
    if report.is_clean:
        print("Result: dataset looks usable (no rejected samples, no manifest errors).")
    else:
        print("Result: dataset is NOT ready -- see above.")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``grafology-intake`` console script.

    Returns a process exit code (0 if the dataset has zero rejected
    samples and no manifest-level errors, non-zero otherwise) rather than
    calling ``sys.exit`` itself, so it can be invoked directly and
    asserted on in tests without a ``SystemExit`` needing to be caught.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = run_intake(args.dataset_dir)
    _print_summary(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
            f.write("\n")
        print(f"\nDetailed report written to {args.output}")

    return 0 if report.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
