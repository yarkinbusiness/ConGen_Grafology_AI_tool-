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
