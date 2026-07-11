# ConGen Grafology AI Tool

ConGen Grafology AI Tool is a Python pipeline that supports professional
graphologists in analyzing handwriting samples. It validates a graphological
image dataset, runs a classical image-processing feature analyzer, produces
cautious, non-diagnostic interpreted findings, and renders those findings
into a structured markdown report.

**This is a support tool, not a diagnostic tool.** It does not determine
anything about a writer's character, health, or psychological state, and it
is not intended to replace a professional graphologist's own judgment --
every report it produces says so explicitly and repeats the point at the
top and bottom of the document.

No real, labeled client handwriting dataset exists yet. Everything in this
repository -- tests, the demo script, and every end-to-end run performed
during development -- is exercised against a synthetic fixture generator
(`grafology_ai.dataset.fixtures`) rather than real samples. See
["The v0 heuristic analyzer, and the road to v1"](#the-v0-heuristic-analyzer-and-the-road-to-v1)
below for what that means in practice and what changes once real data
arrives.

## Setup

Requires Python >= 3.10. Install the package in editable mode with the
development dependencies (`pytest`, `httpx`, needed for the test suite and
for exercising the API via `TestClient`):

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Try it: the demo script

`examples/demo.py` is a self-contained, no-argument walkthrough of the
whole pipeline. It generates one synthetic sample image, runs the full
pipeline against it at both supported depths (`"concise"` and
`"indepth"`), and prints what happened at every stage -- validation
verdicts, measured features, interpretation summary, and the full rendered
markdown report -- plus writes the reports to a temporary directory (its
path is printed at the end) so you can open them directly.

```bash
python examples/demo.py
```

It needs nothing beyond `pip install -e ".[dev]"`: no real dataset, no
running server, no manual setup, and no files outside this repository. It
uses a fixed random seed, so the generated sample (and therefore most of
the printed output) is the same on every run.

## Using the CLI

Installing the package registers a `grafology-analyze` console script -- a
thin wrapper around the same `run_analysis()` function the demo script and
the API both call, so all three ways of running the pipeline can never
diverge in behavior.

```
$ grafology-analyze --help
usage: grafology-analyze [-h] [--depth {concise,indepth}] [--output PATH]
                         [--quality-label {high,medium,low}]
                         [--sample-id TEXT]
                         image

Run the full handwriting-sample analysis pipeline (automated image-quality
validation, classical feature analysis, interpretation, and markdown report
generation) on a single image, end to end. This is a support tool for
professional graphologists, not a diagnostic tool.

positional arguments:
  image                 Path to the handwriting-sample image file (JPEG or
                        PNG).

options:
  -h, --help            show this help message and exit
  --depth {concise,indepth}
                        Interpretation/report depth (default: indepth).
  --output PATH         Write the markdown report to this file instead of
                        printing to stdout.
  --quality-label {high,medium,low}
                        Manifest-declared quality label for this sample, if
                        known. 'low' downgrades blur/contrast rejections to a
                        flag for manual review instead of a hard reject.
  --sample-id TEXT      Sample identifier to include in the generated report
                        header.
```

Example invocation, writing a concise report to a file instead of stdout:

```bash
grafology-analyze path/to/sample.png --depth concise --sample-id S001 --output report.md
```

A missing or unreadable image file produces a single clear `stderr`
message and a non-zero exit code, never a raw Python traceback.

## Using the API

`grafology_ai.api` defines a minimal FastAPI `app` object with one route,
`POST /analyze` -- again a thin wrapper around the same `run_analysis()`
function. This exists to prove the pipeline is "web-ready", not to ship a
production server: **no ASGI server is bundled or started by this
repository**, and the project does not depend on one (e.g. `uvicorn`) --
only on `fastapi` itself.

Today, exercise it via FastAPI's `TestClient` (this is exactly how
`tests/test_api.py` does it):

```python
from fastapi.testclient import TestClient
from grafology_ai.api import app

client = TestClient(app)
with open("path/to/sample.png", "rb") as f:
    response = client.post(
        "/analyze",
        files={"image": ("sample.png", f, "image/png")},
        data={"depth": "concise"},
    )
data = response.json()
print(data["overall_summary"])
print(data["report_markdown"])
```

`POST /analyze` accepts `multipart/form-data` with an `image` file field
plus optional `depth` (`"concise"`/`"indepth"`), `quality_label`
(`"high"`/`"medium"`/`"low"`), and `sample_id` fields, and returns a JSON
body with `sample_id`, `depth`, `report_markdown`, `overall_summary`,
`has_rejected_validation`, and the full list of `validation_results`. An
unreadable/corrupt upload returns `400 Bad Request` with a descriptive
`detail` message rather than a raw traceback.

A future real deployment would run this same `app` object under a
standard ASGI server, e.g.:

```bash
pip install uvicorn
uvicorn grafology_ai.api:app --host 0.0.0.0 --port 8000
```

(`uvicorn` is illustrative here, not a dependency of this project --
adding a real server, auth, rate limiting, etc. is future work, not
something this repository currently does.)

## Architecture: the pipeline stages

Every entry point (`run_analysis()`, the CLI, the API) chains the same
four stages, in the same order, over one image:

```
validate_sample()  ->  analyze()  ->  interpret()  ->  generate_report()
```

| Stage | Function | Lives in | What it does |
|---|---|---|---|
| 1. Validate | `validate_sample()` | `src/grafology_ai/validation/validators.py` | Automated image-quality checks (format, resolution, blur, contrast). Returns a list of accept/reject/flag verdicts; never blocks the rest of the pipeline (see design note below). |
| 2. Analyze | `analyze()` | `src/grafology_ai/analysis/features.py` | Classical image-processing feature extraction (Pillow + numpy only): slant, stroke width (pressure proxy), letter size, line/word spacing, baseline slope, margins, ink density, rhythm regularity, stroke continuity, and overall layout organization -- plus a per-field confidence score. Returns a `Features` record of plain numbers, no interpretation attached. |
| 3. Interpret | `interpret()` | `src/grafology_ai/interpretation/interpret.py` | Maps `Features` onto the indicator vocabulary in `docs/labeling_rubric.md` and attaches short, hedged, non-diagnostic narrative text to each one. Supports two depths (`"concise"`, `"indepth"`). Enforces its own language discipline against a denylist (`DISALLOWED_TERMS`) of diagnostic/absolute/evaluative language. Returns a `StructuredFindings` record (per-indicator findings, strengths, areas of attention, an overall summary). |
| 4. Report | `generate_report()` / `save_report()` | `src/grafology_ai/report/generator.py` | Renders a `StructuredFindings` into a fixed-section markdown report (header, disclaimer, overall summary, per-indicator findings with confidence, strengths, areas of attention, closing note) as-is -- it never rewrites or embellishes the interpretation layer's text. |

These four stages are chained by a single function,
`run_analysis()` (`src/grafology_ai/run_analysis.py`), which both
`grafology_ai.cli` and `grafology_ai.api` call directly rather than
reimplementing any pipeline logic themselves. A design note worth
knowing: a `"reject"` validation verdict (e.g. the image is too blurry)
does not stop the rest of the pipeline -- the sample still gets a full,
readable report, with an added caveat noting which checks did not pass,
so a graphologist reviewing a flagged sample gets a complete reading
rather than nothing.

Two additional subpackages support the dataset side, independent of the
per-sample analysis pipeline above:

- **`src/grafology_ai/dataset/`** -- the manifest schema (`schema.py`,
  `ManifestEntry`) and the synthetic fixture generator (`fixtures.py`,
  `generate_fixture_dataset()`) used throughout this repo's tests and the
  demo script in place of a real dataset.
- **`src/grafology_ai/pipeline/`** -- dataset-level ingest, validation,
  deterministic train/val/test split, and versioned export
  (`run_pipeline()`). See `docs/pipeline_runbook.md` for the full
  operational how-to, and `docs/labeling_rubric.md` for the indicator
  vocabulary graphologists use to label ground truth.

## The v0 heuristic analyzer, and the road to v1

**`analyze()` today is a classical image-processing heuristic, not a
trained machine-learning model.** It measures slant, stroke width,
spacing, baseline, margins, rhythm regularity, stroke continuity, and
overall layout organization directly from pixels using from-scratch
Otsu thresholding, run-length analysis, and a projection-profile shear
search -- documented assumptions and simplifications throughout
`src/grafology_ai/analysis/features.py`. It is a *stand-in*, used because
**no real, labeled client handwriting dataset exists yet** to train a
model on. Every threshold in the interpretation layer
(`src/grafology_ai/interpretation/interpret.py`) is similarly a
deliberately chosen, documented, illustrative constant -- not a value
calibrated against labeled data, because there is no labeled data yet to
calibrate against.

This is a deliberate, swappable seam, not an accident:

- `analyze()`'s contract is `Features` -- a plain, frozen dataclass of
  numeric measurements plus a per-field confidence score (see
  `src/grafology_ai/analysis/features.py`). Any future analyzer, heuristic
  or trained, just needs to produce a `Features` record with the same
  shape.
- `interpret()`'s contract is `Features -> StructuredFindings`. A trained
  model that reproduces (or extends) this same `Features` shape can be
  dropped in behind `analyze()`'s current position without changing
  `interpret()`, `generate_report()`, `run_analysis()`, the CLI, or the
  API at all.

**Once a labeled client dataset arrives**, `src/grafology_ai/pipeline/` is
already built to prepare it: `ingest_dataset()` reads a raw
`manifest.json` + `images/` directory, `validate_dataset()` runs the same
automated quality checks and marks accepted/flagged/rejected samples,
`split_dataset()` deterministically partitions accepted samples into
train/val/test sets, and `export_snapshot()` writes a versioned,
content-hashed snapshot directory -- all chainable in one call via
`run_pipeline()` (see `docs/pipeline_runbook.md`). `docs/labeling_rubric.md`
defines the indicator vocabulary graphologists would use to label that
dataset's ground truth, in the same terms `interpret()` already reports
in, so human labels and the heuristic's current estimates are comparable
on the same footing from day one.

At that point, a trained model becomes a second implementation of the
`Features`-producing seam above (or extends `interpret()`'s indicator
coverage) -- evaluated against, and eventually replacing, today's
heuristic baseline -- without the CLI, API, or report generator needing to
change.

## Project layout

```
src/grafology_ai/
  validation/    Stage 1: automated image-quality checks
  analysis/      Stage 2: classical feature extraction (the "v0 heuristic")
  interpretation/Stage 3: cautious, non-diagnostic interpretation
  report/        Stage 4: markdown report rendering
  dataset/       Manifest schema + synthetic fixture generator
  pipeline/      Dataset ingest / validate / split / versioned export
  run_analysis.py  Single entry point chaining stages 1-4
  cli.py         `grafology-analyze` console script
  api.py         Minimal FastAPI `/analyze` endpoint
examples/
  demo.py        Golden-path, no-argument demo (see above)
docs/
  labeling_rubric.md    Indicator vocabulary for graphologist ground-truth labels
  pipeline_runbook.md   Operational how-to for the dataset pipeline
tests/           116 tests covering every module above, run via `pytest`
```
