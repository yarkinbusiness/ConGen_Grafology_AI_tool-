# Pipeline Runbook

This is the operational runbook for the dataset pipeline step of the
client's technical proposal ("Pipeline setup (ingest/split/export/
versioning)"), implemented in `src/grafology_ai/pipeline/`. It covers
generating a fixture dataset to develop and test against, running the
pipeline end to end, what the output looks like, and how versioning works.

No real client dataset exists yet. Every command below is written against
the synthetic fixture generator (`grafology_ai.dataset.fixtures`); once a
real dataset exists, the same commands apply unchanged as long as it is
laid out as `manifest.json` + `images/` (see "Input layout" below).

## 1. Generate a fixture dataset

```python
from pathlib import Path
from grafology_ai.dataset.fixtures import generate_fixture_dataset

generate_fixture_dataset(Path("/tmp/raw_dataset"), count=20, seed=0)
```

This writes:

```
/tmp/raw_dataset/
  manifest.json          # array of ManifestEntry.to_dict() records
  images/
    synthetic-0000.png
    synthetic-0001.png
    ...
```

`count` and `seed` fully determine the output: the same `(count, seed)`
pair always produces byte-identical images and manifest content, no
matter where you write it. Use this to reliably reproduce a dataset for
testing the pipeline below.

## 2. Run the pipeline end to end

The simplest way to run ingest -> validate -> split -> export in one call:

```python
from pathlib import Path
from grafology_ai.pipeline import run_pipeline

result = run_pipeline(
    input_dir=Path("/tmp/raw_dataset"),
    export_dir=Path("/tmp/exports"),
    seed=0,               # seeds the deterministic train/val/test split
    train_ratio=0.70,      # defaults shown; all three must sum to 1.0
    val_ratio=0.15,
    test_ratio=0.15,
)

print(result.version)        # e.g. "v_3f9a1c2b7e4d5a6b"
print(result.snapshot_dir)   # /tmp/exports/v_3f9a1c2b7e4d5a6b
```

Each stage is also usable on its own, if you need to inspect or act on an
intermediate result (e.g. look at rejected samples before deciding whether
to export, or try different split ratios without re-running validation):

```python
from grafology_ai.pipeline import (
    ingest_dataset, validate_dataset, partition_by_status,
    split_dataset, export_snapshot,
)

ingested = ingest_dataset(Path("/tmp/raw_dataset"))
validated = validate_dataset(ingested)               # runs validate_sample() per image
included, rejected = partition_by_status(validated)  # rejected = any "reject" verdict

split = split_dataset(
    [sample.entry for sample in included],
    seed=0, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
)

result = export_snapshot(Path("/tmp/exports"), validated, split)
```

Note that `export_snapshot` takes *all* validated samples (accepted,
flagged, and rejected), not just the split -- it needs the rejected
samples to write `rejected_manifest.json` and the full set to write
`validation_report.json`.

### Input layout

`ingest_dataset(input_dir)` expects the same layout
`generate_fixture_dataset` produces:

- `input_dir/manifest.json`: a JSON array of manifest entries (see
  `grafology_ai.dataset.schema.ManifestEntry` / `manifest.schema.json`).
- `input_dir/images/`: one image file per entry, named `{sample_id}.*`
  (extension is not assumed -- ingest locates the file by globbing on
  `sample_id`, so PNG or JPEG both work).

Ingest fails fast with a clear `FileNotFoundError` if any manifest entry's
image is missing, and with a `ValueError` if a `sample_id` matches more
than one file (ambiguous) or the manifest fails schema validation.

### Accept / reject / flag

Each sample runs through
`grafology_ai.validation.validators.validate_sample()` (format,
resolution, blur, contrast checks), passing the manifest's `quality`
field through as `quality_label`. The dataset-level disposition is
derived from the per-check verdicts:

- **rejected**: any check verdict is `"reject"`. Excluded from the
  train/val/test splits; recorded in `rejected_manifest.json`.
- **flagged**: no check rejected the sample, but at least one verdict is
  `"flag"` (e.g. a manifest-declared `quality="low"` sample whose blur/
  contrast defect was downgraded from reject to flag). **Included** in
  the splits, but marked for manual review in `validation_report.json`.
- **accepted**: every check verdict is `"accept"`.

## 3. Output structure

```
/tmp/exports/
  v_3f9a1c2b7e4d5a6b/                 # version = content hash, see below
    train_manifest.json               # array of ManifestEntry.to_dict()
    val_manifest.json                 # array of ManifestEntry.to_dict()
    test_manifest.json                # array of ManifestEntry.to_dict()
    rejected_manifest.json            # array of entry dicts + "rejection_reasons"
    validation_report.json            # summary counts + full per-sample results
    images/
      synthetic-0000.png              # copied from the source images/ dir
      synthetic-0003.png              # (only for samples in a split --
      ...                             #  rejected samples' images are not copied)
```

- `train_manifest.json` / `val_manifest.json` / `test_manifest.json`:
  each entry's full `ManifestEntry.to_dict()`, sorted by `sample_id`.
- `rejected_manifest.json`: each rejected entry's `to_dict()` plus a
  `"rejection_reasons"` list of the specific checks that rejected it
  (`check_name`, `reason`, `measured_value`).
- `validation_report.json`: `{"summary": {...}, "samples": [...]}` --
  `summary` has total/accepted/flagged/rejected counts and a per-check
  accept/reject/flag breakdown; `samples` has every sample's `sample_id`,
  status, and full list of per-check results.
- `images/`: a copy (not a reference) of every accepted or flagged
  sample's image file, named `{sample_id}{original extension}`. Copying
  makes the snapshot a standalone artifact that does not depend on the
  source raw-dataset directory continuing to exist or stay unchanged.

## 4. Versioning and re-running

The version identifier (`v_<16 hex chars>`, the snapshot's directory name
under `export_dir`) is a truncated SHA-256 hash of everything that
determines the snapshot's content: every sample's manifest fields,
validation results and status, split assignment, and raw image bytes --
**not** a counter or a timestamp.

Practical consequences:

- **Re-running with unchanged input is a safe no-op you can detect.**
  Running the pipeline again against the same raw dataset (or an
  independently regenerated fixture dataset with the same `seed`/`count`)
  produces the same version identifier and a byte-identical snapshot
  directory. If the target snapshot directory already exists, it is
  overwritten with a fresh (byte-identical) copy rather than skipped or
  duplicated.
- **Any content change produces a new version.** A different fixture
  seed, a different image, a different manifest field, or a different
  split (different `seed`/ratios passed to `run_pipeline`) changes the
  hash and therefore the version identifier -- there is never ambiguity
  about whether two snapshots hold the same data.
- **Comparing versions is comparing content.** Two version strings equal
  means the two snapshots' `train_manifest.json`, `val_manifest.json`,
  `test_manifest.json`, `rejected_manifest.json`, `validation_report.json`,
  and image bytes are all identical -- there is no case where the version
  matches but the content differs, or vice versa.

To pin a specific snapshot for downstream use (e.g. training), record its
full `result.version` string (or the `result.snapshot_dir` path) alongside
whatever consumes it.

## 5. Day-Zero Procedure: what to do the moment a real client dataset arrives

This section is the operational checklist for the day a real, client-
provided dataset is dropped somewhere on disk (as `manifest.json` +
`images/`, per "Input layout" above). It chains together four tools built
across milestones 1 and 2 -- `grafology-intake`, `grafology_ai.assessment`,
`grafology_ai.pipeline.run_pipeline` (section 2 above), and
`grafology_ai.training`/`grafology_ai.evaluation` -- in the order you should
run them. **Every command below was actually executed, verbatim, against a
`generate_fixture_dataset()`-produced dataset as a dry run** (24 synthetic
samples, `seed=0`) before being written down here; the sample counts/hashes
quoted are the real output from that run, not illustrative numbers. Against
a real dataset the counts and version hashes will differ, but the commands
and their shape will not.

### 5.1. Run `grafology-intake` on the raw dataset

```bash
grafology-intake /path/to/raw_dataset
```

Check the exit code (`0` = clean, non-zero = not ready) and read the
fix-it checklist. Real output from the dry run (`grafology-intake` against
the fixture dataset):

```
Dataset: /tmp/dayzero_dryrun/raw_dataset
Total samples: 24  (accepted: 19, flagged: 2, rejected: 3)

Rejected samples:
  - synthetic-0007: blur: Laplacian variance 136.65 is below the sharpness threshold 150; the image appears blurred
  - synthetic-0009: blur: Laplacian variance 147.81 is below the sharpness threshold 150; the image appears blurred
  - synthetic-0017: blur: Laplacian variance 110.14 is below the sharpness threshold 150; the image appears blurred

Fix-it checklist:
  - 3 samples are too blurry to use -- recapture with a steadier hand or better camera focus.

Result: dataset is NOT ready -- see above.
```

Exit code was `1` (non-zero), because 3 samples were rejected. This is the
expected, normal outcome for a first real drop -- it does not block the
next step. If the checklist reveals a *structural* problem (a missing
`manifest.json`, a schema-invalid entry, a missing image file --
`report.manifest_errors`), fix those before proceeding; a handful of
rejected-for-quality samples (blur/contrast/resolution/format) does not
need to block moving on to the technical assessment, since
`grafology_ai.assessment` and `run_pipeline` both handle rejected samples
gracefully (excluded from splits, but visible in the reports).

Add `--output report.json` to also write the full machine-readable report
(counts, per-rejected-sample failed checks, manifest errors, fix-it
checklist) to disk for the record.

### 5.2. Produce the technical assessment deliverable

This is the "first technical assessment of the available dataset,
highlighting strengths, limitations and areas for improvement" the
client's technical proposal calls for as an up-front deliverable.

```python
from pathlib import Path
from grafology_ai.assessment import assess_dataset, render_assessment_report, save_assessment_report

assessment = assess_dataset(Path("/path/to/raw_dataset"))
report_markdown = render_assessment_report(assessment)
save_assessment_report(assessment, Path("/path/to/dataset_assessment.md"))
```

`assess_dataset` reuses `grafology-intake`'s `run_intake` internally for
all quality/validation judgment (do not re-derive accepted/flagged/
rejected counts by hand), and adds composition breakdowns (acquisition
method, quality, language), per-indicator label coverage against the
9-indicator rubric in `docs/labeling_rubric.md`, and a comparison of
dataset size against the technical proposal's own thresholds
(`grafology_ai.assessment.PROTOTYPE_MINIMUM = 50`,
`grafology_ai.assessment.STABILIZING_THRESHOLD = 300`). Real output from
the dry run (report head, `render_assessment_report` on the same 24-sample
fixture dataset):

```
# Dataset Assessment Report
**Dataset directory:** /tmp/dayzero_dryrun/raw_dataset

## Composition

**Total samples:** 24

**By acquisition method:**
- photo: 8 (33%)
- scan: 16 (67%)

**By quality:**
- high: 17 (71%)
- low: 2 (8%)
- medium: 5 (21%)

**By language:**
- en: 24 (100%)

## Quality & Validation Summary

**Accepted:** 19  **Flagged:** 2  **Rejected:** 3

**Fix-it checklist:**
- 3 samples are too blurry to use -- recapture with a steadier hand or better camera focus.

## Label Coverage

0 of 24 sample(s) have at least one labeled indicator; 0 of 24 sample(s) are fully labeled across all 9 rubric indicators.

| Indicator | Coverage |
|---|---|
| Pressure | 0% |
| Slant | 0% |
...
```

The dry-run fixture dataset has zero label coverage (fixtures always
produce empty `labels` -- see `grafology_ai.dataset.fixtures`), so the
"Size vs. Thresholds", "Strengths", "Limitations", and "Recommendations"
sections below the excerpt above are exactly the kind of "not ready for
training yet" read this report exists to surface early. Send (or attach)
this rendered markdown to the client as the technical-assessment
deliverable.

### 5.3. Run the pipeline to produce a versioned dataset snapshot

Once the dataset has been assessed (it does not need to be perfect --
rejected/flagged samples are handled automatically), run the existing
ingest/validate/split/export pipeline documented in full in section 2
above:

```python
from pathlib import Path
from grafology_ai.pipeline import run_pipeline

result = run_pipeline(
    input_dir=Path("/path/to/raw_dataset"),
    export_dir=Path("/path/to/exports"),
    seed=0,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
)
print(result.version)
print(result.snapshot_dir)
```

Real output from the dry run:

```
version: v_12344fa9a2f4ba26
snapshot_dir: /tmp/dayzero_dryrun/exports/v_12344fa9a2f4ba26
```

...with `train_manifest.json`, `val_manifest.json`, `test_manifest.json`,
`rejected_manifest.json`, `validation_report.json`, and an `images/`
directory written under that snapshot directory, exactly as described in
section 3 above. Record `result.version` for the record -- it is what
training (next step) should be run against.

### 5.4. Once real labels exist for at least one rubric indicator: training and evaluation

`grafology_ai.training.train_baseline()` and
`grafology_ai.evaluation.compare_baseline_to_heuristic()` already work end
to end (see their module docstrings and `tests/test_training.py`/
`tests/test_evaluation.py`) -- but today, both are wired only to the
**"pressure"** indicator and to **synthetic, fabricated labels**
(`grafology_ai.training.synthetic_labels.generate_synthetic_pressure_labels`),
because no real labeled data exists yet. Both functions' actual signatures
already accept real data with no code changes needed:

- `train_baseline(features_and_labels: list[tuple[Features, str]], seed=0)`
  takes a plain list of `(Features, label)` pairs -- it has no dependency
  on `synthetic_labels` at all. The only thing that needs to change is
  **what `label` comes from**: instead of
  `generate_synthetic_pressure_label(features)`, read
  `entry.labels.get("pressure")` from a manifest entry (or snapshot
  `train_manifest.json`) whose `labels` dict a graphologist has actually
  filled in, and only include pairs where that value is non-empty.
- `compare_baseline_to_heuristic(model, features_and_labels)` takes the
  same `(Features, ground_truth)` pair shape (`ground_truth` may be `None`
  for still-unlabeled samples) -- again no change needed beyond sourcing
  `ground_truth` from real manifest labels instead of
  `generate_synthetic_pressure_labels`.

What **would** need to change once real labels exist for indicators other
than pressure: `grafology_ai.training.baseline.FEATURE_NAMES` (currently
`("stroke_width_mean", "stroke_width_std")`, chosen because it is
`analyze()`'s most direct pressure proxy) and `INDICATOR` (`"pressure"`)
are today hardcoded to the one indicator the M2.3 plumbing proof targets --
extending the baseline to e.g. "slant" or "letter_size" means picking the
right `Features` fields for that indicator and either parameterizing
`train_baseline`/`predict` or adding a sibling model. That is real
experimentation work for the training-budget hours in the technical
proposal, not something to pre-build against synthetic data.

The dry run below simulates this transition exactly: it takes the
`train_manifest.json` snapshot from step 5.3, writes a real-looking
`"pressure"` label into 10 of its 15 train-split entries (standing in for
a graphologist's in-progress labeling of a partial dataset -- this
label-injection step only exists in the dry run; against a real dataset
the manifest would already carry the labels), then trains and evaluates
against those labels via `entry.labels.get("pressure")` -- no
`synthetic_labels` import anywhere in this step.

```python
import json
from pathlib import Path
from grafology_ai.analysis import analyze
from grafology_ai.training import train_baseline, save_model
from grafology_ai.evaluation import compare_baseline_to_heuristic, render_comparison_report

train_manifest_path = Path("/path/to/exports/<version>/train_manifest.json")
images_dir = train_manifest_path.parent / "images"
with train_manifest_path.open() as f:
    train_entries = json.load(f)

features_and_labels = []
for entry in train_entries:
    label = entry.get("labels", {}).get("pressure")   # <-- real graphologist label
    if not label:
        continue
    features = analyze(images_dir / f"{entry['sample_id']}.png")
    features_and_labels.append((features, label))

model = train_baseline(features_and_labels, seed=0)
saved = save_model(model, Path("/path/to/models"))

comparison = compare_baseline_to_heuristic(model, features_and_labels)
report_markdown = render_comparison_report(comparison)
```

Real output from the dry run:

```
Injected 'pressure' labels into 10 of 15 train-split entries.
Training pairs with a real 'pressure' label: 10
Trained model: indicator=pressure, classes=('firm', 'light', 'moderate')
Saved model: version=v_34dc734112440cbd at /tmp/dayzero_dryrun/models/v_34dc734112440cbd.json
n_samples=10, n_ground_truth_available=10
agreement_rate=100.00%
```

`render_comparison_report()`'s output carries its own prominent "plumbing
proof, not a real accuracy claim" disclaimer near the top of the document
-- keep that disclaimer intact if the comparison report is shared onward,
until real, graphologist-labeled data and a real (fine-tuned, not
baseline) model exist to evaluate.
