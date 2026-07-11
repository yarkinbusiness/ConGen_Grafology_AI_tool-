# ConGen Grafology AI Tool — Implementation Roadmap (Gaps 2–5)

**Scope:** close the four confirmed gaps against the client brief — PDF input (gap 2), independent rhythm / stroke-continuity / overall-organization measurement (gap 3), PDF report export (gap 4), and backend web-readiness for an externally designed frontend (gap 5, backend half only). **Explicitly out of scope:** any visual UI/frontend work (handled separately via Claude Design), auth systems, user accounts, billing, background job queues, and the trained AI model.

**On gap 1 (trained model) — acknowledgment only:** blocked on the client dataset (see `docs/client_status_update.md`). Nothing in this roadmap disturbs the swappable seam: `analyze()` → `Features` → `interpret()` (`src/grafology_ai/analysis/features.py`, `src/grafology_ai/interpretation/interpret.py`). Phase A *extends* `Features` with three new fields; that keeps the seam ready — the day data arrives, the new fields are labelable against the same rubric entries (`docs/labeling_rubric.md` already defines Rhythm, Stroke Continuity, and Overall Organization) and `src/grafology_ai/pipeline/` + `src/grafology_ai/training/` need no changes (the baseline names its input fields explicitly: `training/baseline.py::FEATURE_NAMES = ("stroke_width_mean", "stroke_width_std")`).

**Execution model:** each task below is written to be handed verbatim to a worker agent and graded by an independent brain reviewer per `docs/agent-patterns/brain-worker-pattern.md` §2. Acceptance bars are itemized and independently checkable (re-run `pytest`, re-read the diff). Baseline: the suite currently collects **160 tests**, all passing; every bar below includes "full existing suite still passes" implicitly, with any *expected* test-helper updates called out.

**Phase ordering rationale (one paragraph):** Phase A changes the core data shapes (`Features`, `StructuredFindings`) that everything downstream renders and serializes — do it *before* the API contract is stabilized in Phase D, or the contract gets frozen twice. Phase B (PDF input) is independent of A and touches the input seam only. Phase C (PDF export) needs no new data model (`StructuredFindings` already is the stable report data model, and `report/generator.py` proves rendering from it works) but is sequenced after B because B's rasterizer (`pypdfium2`) doubles as the PDF-text-extraction tool Phase C's tests need to verify rendered output. Phase D last: it consolidates everything (new indicators, PDF in, PDF out) into one documented, versioned API contract, so it must not run before the shapes it freezes exist.

---

## Phase A — Complete the measurement vocabulary: rhythm, stroke continuity, overall organization (gap 3)

**Rationale:** the brief names rhythm, continuity of stroke, and overall organization as first-class analysis targets. Today `ink_density` is a single documented "rough proxy" for the first two (`analysis/features.py` line ~184) and the interpretation layer's `"rhythm"` indicator is keyed directly to it (`interpret.py::INDICATOR_CONFIDENCE_KEY`); organization has no measure at all. This phase must precede Phase D (contract freeze) because it changes `Features` and the findings list. Constraint carried from the existing code: Pillow + numpy only, no OpenCV/scipy, deterministic, all documented-constant thresholds — match the conventions already in `features.py`.

**Cross-cutting note for A1–A3:** `Features` has no field defaults and is constructed positionally/by-kwargs in `tests/test_evaluation.py::_make_features` and in `analysis/features.py::_empty_features`/`analyze()`. Each task adds its field *before* `confidence`, updates `_empty_features`, `_build_confidence`, and the test helper. `FEATURE_CONFIDENCE_KEYS` is derived dynamically from the dataclass, so existing confidence-coverage tests will automatically demand the new keys — that's the safety net working as designed.

### A1 — Rhythm metric (`rhythm_regularity`)

- **Build:** a `rhythm_regularity: float` field in `Features` measuring regularity of repetition across the sample per the rubric's Rhythm definition — e.g. an inverse-coefficient-of-variation composite over (a) per-line-band ink-run heights, (b) word-gap lengths (`_estimate_word_spacing` already collects the gap list — refactor it to expose gaps rather than re-deriving), and (c) per-band stroke widths, normalized to [0, 1] where 1 = highly regular. Exact formula is the worker's choice but must be documented with named constants and derived only from quantities the module already computes or can compute with existing helpers.
- **Files:** `src/grafology_ai/analysis/features.py`, `tests/test_analysis.py`, `tests/test_evaluation.py` (helper only).
- **Acceptance bar:**
  1. `Features` has `rhythm_regularity`; `FEATURE_CONFIDENCE_KEYS` includes it; `analyze()` populates both value and confidence; `_empty_features` returns 0.0/0.0 for it.
  2. ≥5 new passing tests in `tests/test_analysis.py`: (i) a synthetic image with uniform stroke spacing/size scores strictly higher than one with deliberately varied spacing/size (both constructed in-test with PIL, same overall ink amount); (ii) value always in [0.0, 1.0] and finite across the existing fixture set (`generate_fixture_dataset`); (iii) blank image → 0.0 value, 0.0 confidence; (iv) determinism — two `analyze()` calls on the same image return identical values; (v) confidence in [0, 1].
  3. Full suite passes; only permitted existing-test change is adding the new kwarg to `_make_features`-style helpers.

### A2 — Stroke-continuity metric (`stroke_connectedness`)

- **Build:** a `stroke_connectedness: float` field in [0, 1] per the rubric's Stroke Continuity definition (joined vs. lifted/broken strokes). Suggested approach consistent with the existing run-length machinery: within each detected line band (`_detect_line_bands`), compare the count of ink-column-run segments to the count of Otsu-split "word" clusters (`_estimate_word_spacing`'s split already distinguishes intra-word from inter-word gaps) — many segments per word ⇒ broken/printed; few ⇒ connected. Higher = more connected.
- **Files:** `src/grafology_ai/analysis/features.py`, `tests/test_analysis.py`, `tests/test_evaluation.py` (helper only).
- **Acceptance bar:**
  1. Field + confidence wired as in A1 (dataclass, `analyze()`, `_empty_features`, `_build_confidence`).
  2. ≥5 new passing tests: (i) a synthetic "connected" image (continuous horizontal strokes per line) scores strictly higher than a "broken" one (the same strokes rendered dashed/segmented); (ii)–(v) same bounds/blank/determinism/confidence checks as A1.
  3. Full suite passes.

### A3 — Overall-organization metric (`organization_score`)

- **Build:** an `organization_score: float` field in [0, 1] per the rubric's Overall Organization definition, composited from measurable layout properties already derivable from `_detect_line_bands` and the ink mask: consistency of inter-band spacing, consistency of per-band left-edge start column (line-start alignment), and consistency of per-band baseline slope (reuse `_estimate_baseline_slope`'s per-band fits — refactor to expose them rather than recomputing). Weights and normalization as documented named constants.
- **Files:** `src/grafology_ai/analysis/features.py`, `tests/test_analysis.py`, `tests/test_evaluation.py` (helper only).
- **Acceptance bar:**
  1. Field + confidence wired as in A1. Confidence must scale with the number of detected line bands (an organization read off one line is near-meaningless) — a single-band image must report confidence < 0.5.
  2. ≥5 new passing tests: (i) an "organized" synthetic image (evenly spaced, left-aligned, level lines) scores strictly higher than a "disorganized" one (uneven spacing, ragged starts, mixed slopes); (ii)–(v) as A1, plus the single-band low-confidence check.
  3. Full suite passes.

### A4 — Interpretation layer: re-key rhythm, add two new indicators

- **Build:** in `src/grafology_ai/interpretation/interpret.py`: (1) re-key the existing `"rhythm"` indicator from `ink_density` to `rhythm_regularity` (`INDICATOR_CONFIDENCE_KEY`), rewrite `_candidate_rhythm` around regular/moderate/irregular buckets with new documented thresholds, and rename its label to plain `"Rhythm"`; (2) add `"stroke_continuity"` and `"organization"` indicators — `INDICATOR_ORDER` (→ 12 entries), `INDICATOR_LABELS`, `INDICATOR_CONFIDENCE_KEY`, one `_candidate_*` builder each with ≥2 buckets, hedged full+brief text in the established voice, appended to `_CANDIDATE_BUILDERS`. Keep `ink_density` in `Features` (it stays useful raw data) — it simply no longer backs an indicator. `CONCISE_INDICATOR_COUNT` stays 4. `report/generator.py` needs **no** change (it renders whatever findings arrive and looks labels up via `INDICATOR_LABELS`).
- **Files:** `src/grafology_ai/interpretation/interpret.py`, `tests/test_interpretation.py`; expect small assertion updates in `tests/test_report.py`/`tests/test_run_analysis.py` if they enumerate indicators.
- **Acceptance bar:**
  1. `interpret(features, depth="indepth")` returns exactly 12 findings, one per `INDICATOR_ORDER` entry, including `"stroke_continuity"` and `"organization"`; `depth="concise"` still returns exactly 4.
  2. Every bucket branch of the three new/rewritten builders is exercised by at least one test (construct `Features` records that land in each bucket), and every generated `observation`/`interpretation`/strength/attention string from those branches passes the existing `DISALLOWED_TERMS` scan — the existing scan test must demonstrably cover the new branches (extend its input matrix, don't just rely on default fixtures).
  3. `Finding.confidence` for the new indicators is a verbatim carry-over from the corresponding `Features.confidence` key (assert equality in a test).
  4. ≥8 new passing tests in `tests/test_interpretation.py` covering points 1–3; full suite passes.

### A5 — Documentation sweep

- **Build:** update `README.md` (stage-2 feature list and the "rough proxy" sentence — the proxy caveat is now obsolete), and the `Features`/`ink_density` docstrings' claim that ink density proxies rhythm/continuity. `docs/labeling_rubric.md` needs no change (it already defines all three indicators). Update `docs/pipeline_runbook.md` §5.2 if it cites "9 rubric indicators" against label coverage — verify `assessment.py`'s rubric-indicator list and keep code/doc consistent.
- **Files:** `README.md`, `src/grafology_ai/analysis/features.py` (docstrings), `src/grafology_ai/assessment.py` (if it hardcodes the indicator list), `docs/pipeline_runbook.md`.
- **Acceptance bar:** no occurrence of "rough proxy for rhythm" language remains (`grep` clean); README's analyzer row lists the three new measurements; if `assessment.py` enumerates rubric indicators, its list matches `docs/labeling_rubric.md`'s 9 sections and `tests/test_assessment.py` still passes; full suite passes.

**Sequencing:** A1, A2, A3 are mutually independent conceptually, but each touches `features.py` — run serially in the gated loop to avoid merge conflicts in one file. A4 requires all of A1–A3. A5 requires A4.

---

## Phase B — PDF input support (gap 2)

**Rationale:** the brief says "image or PDF"; `SUPPORTED_FORMATS = ("JPEG", "PNG")` in `src/grafology_ai/validation/validators.py`, whose `check_format` TODO already names the intended approach (rasterize via `pdf2image` or `pypdfium2`). Independent of Phase A; must precede Phase C (its library doubles as C's test-verification tool) and Phase D (the frozen API contract must include PDF uploads).

**New dependency — recommendation:** **`pypdfium2`** (Apache-2.0 / BSD-3-Clause, self-contained wheels bundling PDFium, no system binary). Justification against alternatives: `pdf2image` requires the external `poppler` binary installed on every host — a real deployment liability for a repo that currently `pip install`s clean; `PyMuPDF` is AGPL-3.0, unacceptable default for a client commercial tool. `pypdfium2` also provides text extraction, reused by Phase C tests. **Client decision needed before B1 starts (see decision log): page policy and rasterization DPI.** Roadmap default pending that decision: rasterize **page 1 only at 300 DPI**, and when a PDF has >1 page, proceed on page 1 and surface a `"flag"`-verdict validation result saying so.

### B1 — PDF rasterization module

- **Build:** new module `src/grafology_ai/input/__init__.py` + `src/grafology_ai/input/pdf.py`: `is_pdf(path_or_bytes) -> bool` (sniff the `%PDF-` magic, don't trust extensions — matching `check_format`'s "never trust the extension" stance), and `rasterize_pdf(source, dpi=DEFAULT_PDF_RASTER_DPI, page_index=0) -> PdfRasterization` where the returned record carries the `PIL.Image`, `page_count`, and the DPI used. Named constants for DPI and any caps. Corrupt/encrypted/zero-page PDFs raise a single dedicated exception type (e.g. `PdfInputError`) with a human-readable message. Add `pypdfium2>=4` to `pyproject.toml` dependencies.
- **Test fixtures:** generate PDFs in-test with Pillow itself (`Image.save(path, format="PDF")`, multi-page via `save_all=True, append_images=[...]`) — no extra dev dependency.
- **Files:** `src/grafology_ai/input/pdf.py` (new), `pyproject.toml`, `tests/test_pdf_input.py` (new).
- **Acceptance bar:**
  1. ≥7 new passing tests: (i) single-page PDF of a fixture image rasterizes to a `PIL.Image` whose pixel dimensions scale with requested DPI (assert 300 DPI output is ~2× the 150 DPI output, ±1px); (ii) multi-page PDF: `page_count` correct and `page_index` selects the right page (pages made visually distinct); (iii) non-PDF bytes → `is_pdf` False; (iv) truncated/corrupt `%PDF-` bytes → `PdfInputError` (not a raw pypdfium2 exception); (v) determinism — two rasterizations byte-identical (`np.array_equal`); (vi) parity — `analyze()` on the rasterized page of a synthetic sample yields the same slant bucket (per `interpret.py` thresholds) as `analyze()` on the source PNG; (vii) `page_index` out of range → `PdfInputError`.
  2. `pip install -e ".[dev]"` still succeeds; full suite passes.

### B2 — Pipeline/validation integration: rasterized images must not be spuriously rejected

- **Build:** the load seam. In `src/grafology_ai/run_analysis.py::_load_image` (and mirroring in `validators.py` as needed): if the input is a path/bytes that `is_pdf` identifies, rasterize page 1 and run the pipeline on the result. Critical detail: a rasterized image has `PIL.Image.format = None`, so `check_format` as written rejects it ("no format information available"). Fix by threading an explicit `declared_format: str | None` through `validate_sample` → `check_format`: when the pipeline itself produced the image from a valid PDF, the format check returns `accept` with reason naming PDF + rasterization DPI; add `"PDF"` to a documented accepted-sources constant. Multi-page PDFs additionally append a `"flag"`-verdict `ValidationResult` (`check_name="pdf_pages"`) stating only page 1 was analyzed. Remove/rewrite the obsolete TODO in `check_format`.
- **Files:** `src/grafology_ai/run_analysis.py`, `src/grafology_ai/validation/validators.py`, `tests/test_run_analysis.py`, `tests/test_validation.py`.
- **Acceptance bar:**
  1. `run_analysis("sample.pdf")` returns a complete `AnalysisResult` with **zero** `"reject"` verdicts attributable to format for a clean single-page PDF (assert `has_rejected_validation is False` for a high-quality fixture rendered to PDF).
  2. Multi-page PDF → exactly one `"flag"` result with `check_name="pdf_pages"`, and the pipeline still completes.
  3. Corrupt PDF path → `PdfInputError` propagates (for the CLI/API layers to map in B3).
  4. ≥6 new passing tests covering points 1–3 plus: quality checks (blur/contrast) still run on the rasterized image and can still reject a genuinely blurry PDF; PNG/JPEG behavior is byte-for-byte unchanged (existing tests untouched). Full suite passes.

### B3 — CLI and API PDF acceptance

- **Build:** `src/grafology_ai/cli.py`: accept PDF paths (help text updated from "JPEG or PNG"), map `PdfInputError` to a single clear stderr line + exit 1, matching the existing error style. `src/grafology_ai/api.py`: `POST /analyze` sniffs uploaded bytes; PDF → rasterize then run; corrupt PDF → `400` with descriptive `detail`. Update `README.md` CLI/API sections.
- **Files:** `src/grafology_ai/cli.py`, `src/grafology_ai/api.py`, `README.md`, `tests/test_cli.py`, `tests/test_api.py`.
- **Acceptance bar:**
  1. ≥6 new passing tests: (i) CLI on a clean PDF exits 0 and the produced report contains `"# Handwriting Analysis Support Report"`; (ii) CLI on corrupt PDF exits 1 with a single stderr line, no traceback; (iii) `--help` no longer says images only; (iv) `TestClient` PDF upload returns 200 with the same JSON keys as an image upload; (v) corrupt PDF upload → 400 with `detail`; (vi) multi-page PDF upload → 200 and `validation_results` contains the `pdf_pages` flag entry.
  2. Full suite passes.

**Sequencing:** B1 → B2 → B3, strictly serial.

---

## Phase C — PDF report export (gap 4)

**Rationale:** the brief requires "a well-organized report that can be exported as a PDF"; today `report/generator.py` renders markdown only and its docstring explicitly defers PDF. The stable report data model the PDF needs **already exists** — `StructuredFindings` — so the renderer works from it directly (never by parsing markdown), exactly as `generate_report()` does; that keeps the two output formats structurally locked together and unable to diverge. Sequenced after B so tests can verify rendered PDFs by extracting text with the already-added `pypdfium2`.

**New dependency — recommendation:** **`fpdf2`** (pure Python, zero system dependencies, good Unicode, actively maintained) — but its **LGPL-3.0 license is a client decision** before C1 starts. The alternative is **`reportlab`** (BSD-3-Clause, equally pure-pip) if the client wants a maximally permissive license; the task is written renderer-agnostic so the choice swaps in cleanly. Do **not** choose `weasyprint` (requires pango/cairo system libraries) — same deployment objection as poppler.

### C1 — PDF renderer module

- **Build:** `src/grafology_ai/report/pdf.py`: `render_report_pdf(findings: StructuredFindings, sample_id: str | None = None) -> bytes` and `save_report_pdf(findings, output_path, sample_id=None) -> Path` (mirroring `save_report`'s parent-dir creation and return contract). Renders the **same seven fixed sections in the same order** as `generate_report()` (header, prominent disclaimer near the top, overall summary, per-indicator findings with confidence label + percentage, strengths, areas of attention, closing note), with all interpretation-layer text verbatim — this module originates layout only, never narrative, per `generator.py`'s "thin renderer" doctrine. Neutral, professional styling only (client branding awaits the external design work). Export the new functions from `src/grafology_ai/report/__init__.py`; add the chosen library to `pyproject.toml`; update `generator.py`'s "PDF is out of scope" docstring paragraphs to point here.
- **Files:** `src/grafology_ai/report/pdf.py` (new), `src/grafology_ai/report/__init__.py`, `src/grafology_ai/report/generator.py` (docstring only), `pyproject.toml`, `tests/test_report_pdf.py` (new).
- **Acceptance bar:**
  1. ≥8 new passing tests, verifying via `pypdfium2` text extraction of the rendered bytes: (i) output starts with `%PDF`; (ii) extracted text contains the disclaimer sentence "not a diagnosis", the verbatim `overall_summary`, every finding's indicator label from `INDICATOR_LABELS`, and every strengths/areas entry, for a full `indepth` `run_analysis` result on a fixture image; (iii) section headings appear in the fixed order (assert index ordering in extracted text); (iv) empty strengths/areas render the same empty-note sentences `generator.py` uses; (v) works at both depths; (vi) sample_id appears when given, no `"None"` literal when not; (vii) non-ASCII narrative characters survive extraction (degree values like `+18.0` and the em-dash-style `--` text render); (viii) `save_report_pdf` creates parents and returns the `Path`.
  2. `render_report_pdf` takes no markdown string anywhere in its implementation (brain verifies by reading the diff).
  3. Full suite passes.

### C2 — CLI PDF export

- **Build:** `src/grafology_ai/cli.py`: `--format {md,pdf}` (default `md`, fully backward compatible); `--format pdf` requires `--output` (PDF bytes to stdout is useless — clear stderr error + exit 2 otherwise); convenience inference: `--output x.pdf` without `--format` selects pdf. Update `--help` and `README.md`.
- **Files:** `src/grafology_ai/cli.py`, `README.md`, `tests/test_cli.py`.
- **Acceptance bar:** ≥4 new passing tests: (i) `--format pdf --output r.pdf` exits 0 and the file starts with `%PDF` and extracts the disclaimer text; (ii) `--format pdf` without `--output` exits non-zero with one stderr line; (iii) `--output r.pdf` alone produces a PDF; (iv) default behavior without new flags byte-identical to before (existing markdown tests unmodified). Full suite passes.

**Sequencing:** C1 → C2. C1 depends on Phase B (test tooling) and benefits from Phase A being done first (new indicators covered by the "every finding label appears" assertion automatically). API-side PDF delivery is deliberately deferred to D3 to keep all API-contract changes in one phase.

---

## Phase D — Backend web-readiness for the external frontend (gap 5, backend half)

**Rationale:** the frontend will be designed and built externally; this codebase's job is a stable, documented, deployable JSON API it can consume. Today `api.py` returns a hand-built dict (no response models, so the OpenAPI schema FastAPI auto-generates is untyped), there is no health endpoint, no CORS, no server dependency, and no contract document. This phase runs **last** because it freezes shapes produced in A–C. **Explicit non-goals, restated:** no auth, no accounts, no rate limiting implementation, no async job queue — each is flagged as a client decision below, and the contract doc records the posture.

### D1 — Canonical JSON serialization of the analysis result

- **Build:** `to_dict()` methods (or a `src/grafology_ai/serialization.py` module — worker's structural choice, but one canonical place) covering `Features`, `Finding`, `StructuredFindings`, `ValidationResult`, and `AnalysisResult`, producing plain JSON-safe types. This is the single source the API response, and any future consumer, serializes from — the API stops hand-building its dict.
- **Files:** `src/grafology_ai/serialization.py` (new) or methods on the dataclasses in `analysis/features.py`, `interpretation/interpret.py`, `validation/validators.py`, `run_analysis.py`; `tests/test_serialization.py` (new).
- **Acceptance bar:** ≥5 new passing tests: (i) `json.dumps(result_dict)` succeeds for a full `run_analysis` result on a fixture image at both depths; (ii) every `Features` dataclass field (including Phase A's three new ones) appears by name, keys verified programmatically against `Features.__dataclass_fields__`; (iii) same programmatic completeness check for `StructuredFindings`/`Finding`/`ValidationResult`; (iv) determinism — two serializations of the same result are `==`; (v) no non-JSON types leak (round-trip through `json.loads(json.dumps(...))`). Full suite passes.

### D2 — Typed, versioned API contract

- **Build:** in `src/grafology_ai/api.py` (or a new `api/` package with `models.py`): Pydantic response models for the analyze response; mount the endpoint at **`POST /v1/analyze`** with `POST /analyze` kept as a thin alias for backward compatibility (or a deprecation redirect — worker documents the choice). Expand the response to carry the full structured result via D1: existing keys unchanged (`sample_id`, `depth`, `report_markdown`, `overall_summary`, `has_rejected_validation`, `validation_results`) **plus** `features` (all measured values + per-field confidence) and `findings` (per-indicator: `indicator`, human `label`, `observation`, `interpretation`, `confidence`, `confidence_label` reusing `report/generator.py::confidence_label`), `strengths`, `areas_of_attention`. Invalid `depth`/`quality_label` values must produce FastAPI's structured `422`.
- **Files:** `src/grafology_ai/api.py` (+ optional `api/models.py`), `tests/test_api.py`.
- **Acceptance bar:** ≥8 new passing tests: (i) `/v1/analyze` 200 response contains every field above, values equal to a direct `run_analysis` call on the same bytes; (ii) `/analyze` still works and returns a superset of the pre-existing keys (all current `tests/test_api.py` assertions pass unmodified); (iii) `app.openapi()` contains named component schemas for the response models (assert schema names present in `openapi.json` via `TestClient`); (iv) `depth=bogus` → 422; (v) `confidence_label` agrees with `confidence` per the thresholds in `generator.py`; (vi) both depths; (vii) PDF upload (Phase B) returns the same shape; (viii) corrupt upload still → 400 with `detail`. Full suite passes.

### D3 — PDF report delivery over the API

- **Build:** `output_format` form field on `/v1/analyze` (`"json"` default | `"pdf"`): `"pdf"` returns `application/pdf` bytes from C1's `render_report_pdf`, with `Content-Disposition: attachment; filename="report-{sample_id|sample}.pdf"`.
- **Files:** `src/grafology_ai/api.py`, `tests/test_api.py`.
- **Acceptance bar:** ≥4 new passing tests: (i) `output_format=pdf` → 200, `content-type` `application/pdf`, body starts `%PDF`, extracted text contains the disclaimer and the given `sample_id`; (ii) `Content-Disposition` filename present and sensible with and without `sample_id`; (iii) default remains JSON (no `output_format` → identical shape to D2); (iv) `output_format=bogus` → 422. Full suite passes.

### D4 — Operational surface: health, CORS, upload cap

- **Build:** (1) `GET /v1/health` → `{"status": "ok", "version": <package version from importlib.metadata>}`. (2) `CORSMiddleware` configured from env var `GRAFOLOGY_CORS_ORIGINS` (comma-separated); **default: no CORS** (secure default; the frontend's real origin is a client decision). (3) Upload size cap: reject bodies over `GRAFOLOGY_MAX_UPLOAD_BYTES` (default 15 MiB — provisional pending client decision) with `413` and a descriptive `detail`, enforced by checking the read bytes in the endpoint (no ASGI-server-level assumption).
- **Files:** `src/grafology_ai/api.py`, `tests/test_api.py`.
- **Acceptance bar:** ≥6 new passing tests: (i) health 200 with both keys, version matches `importlib.metadata.version("grafology-ai")`; (ii) with env set (monkeypatched before app construction — worker must make CORS config re-readable/testable), a request with `Origin` gets the ACAO header; (iii) without env, no ACAO header; (iv) oversized upload → 413 with `detail` naming the limit; (v) at-limit upload passes; (vi) limit env override respected. Full suite passes.

### D5 — Deployability: server extra + serve entry point

- **Build:** `pyproject.toml`: optional extra `server = ["uvicorn"]`; new console script `grafology-serve` (`src/grafology_ai/serve.py`) — a thin argparse wrapper (`--host` default `127.0.0.1`, `--port` default `8000`) that imports uvicorn lazily and errors with a clear "install with `pip install 'grafology-ai[server]'`" message if absent. Update `README.md`'s "Using the API" section (remove the "uvicorn is illustrative, not a dependency" caveat in favor of the real extra).
- **Files:** `pyproject.toml`, `src/grafology_ai/serve.py` (new), `README.md`, `tests/test_serve.py` (new).
- **Acceptance bar:** ≥3 new passing tests: (i) `main(["--port", "9999"])` with `uvicorn.run` monkeypatched asserts it is called with `grafology_ai.api:app` (or the app object), host/port as given — the test never binds a socket; (ii) missing-uvicorn path (monkeypatch import failure) exits non-zero with the install hint on stderr; (iii) `--help` exits 0. `pip install -e ".[dev,server]"` resolves. Full suite passes.

### D6 — API contract document (the frontend's ground truth)

- **Build:** `docs/api_contract.md`: every endpoint (`/v1/analyze` JSON + PDF modes, `/v1/health`, legacy `/analyze`), exact request fields with allowed values/defaults, the full response schema field-by-field, every error status (`400` unreadable file, `413` too large, `422` invalid params) with example bodies, the processing model (**synchronous** — request blocks for the pipeline, typically a few seconds; stated explicitly, with the async-jobs question logged as future/client decision), upload constraints (formats JPEG/PNG/PDF, size cap, PDF page-1 policy), and the security posture (no auth/rate limiting yet — deliberate, documented non-goals pending client decision). Plus a **consistency test**: programmatically assert every response-model field name (from D2's Pydantic models) and every documented status code appears in the doc text, so the doc cannot silently rot.
- **Files:** `docs/api_contract.md` (new), `tests/test_api_contract_doc.py` (new), `README.md` (link).
- **Acceptance bar:** doc exists covering every item above (brain verifies the checklist item-by-item against the doc); the consistency test passes and demonstrably fails if a model field name is removed from the doc (worker shows the red/green run as evidence); full suite passes.

**Sequencing:** D1 → D2 → (D3, D4 in either order) → D5 → D6. All of Phase D after A, B, C.

---

## Client decision log (needed before the flagged task starts)

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| 1 | PDF input page policy: page 1 only (flag extra pages) vs. analyze all pages (multi-report?) | B1 | Page 1 + flag; revisit multi-page later |
| 2 | PDF rasterization DPI | B1 | 300 DPI |
| 3 | PDF renderer license: `fpdf2` (LGPL-3.0) vs `reportlab` (BSD-3) | C1 | `fpdf2` unless license policy forbids LGPL |
| 4 | PDF report styling/branding | C1 | Neutral now; branding after external design work lands |
| 5 | CORS: the frontend's real origin(s) | D4 (config value only, not the task) | Env-configured, default closed |
| 6 | Upload size cap value | D4 | 15 MiB default |
| 7 | Sync vs async processing | D6 wording | Stay synchronous (pipeline runs in seconds); log async as future work |
| 8 | Auth / rate limiting timing | none now (explicit non-goal) | Decide before any public deployment; document posture in D6 |
| 9 | `pypdfium2` acceptable (Apache-2.0/BSD-3, bundles PDFium) | B1 | Yes — permissive, no system deps |

## New third-party dependencies (complete list)

- `pypdfium2` — PDF rasterization in, PDF text extraction in tests (Phase B; reused C/D).
- `fpdf2` **or** `reportlab` — PDF report rendering (Phase C; client decision #3).
- `uvicorn` — **optional extra only** (`[server]`, Phase D5). No other additions; core stays Pillow + numpy + fastapi + python-multipart.

---

## Closing: minimal facts a UI/UX designer needs to design against this backend

1. **Endpoints:** `POST /v1/analyze` (multipart upload) and `GET /v1/health`; processing is **synchronous** — one request in, full result out in seconds; no polling, job IDs, or progress states exist.
2. **Accepted inputs:** one file per request — JPEG, PNG, or PDF (page 1 analyzed; extra pages produce a visible flag notice); max upload size (default 15 MiB).
3. **Request options:** `depth` = `"concise"` | `"indepth"` (default indepth); optional `sample_id` (free text, echoed into report header); optional `quality_label` = `"high"|"medium"|"low"`; `output_format` = `"json"` | `"pdf"`.
4. **Success response (JSON) structure:** `sample_id`, `depth`, `report_markdown` (complete renderable report), `overall_summary`, `features` (numeric measurements + per-field 0–1 confidence), `findings[]` (each: `indicator` slug, human `label`, `observation`, `interpretation`, `confidence` 0–1, `confidence_label` high/medium/low), `strengths[]`, `areas_of_attention[]`, `has_rejected_validation`, `validation_results[]` (each: `check_name`, `verdict` accept/reject/flag, `reason`, `measured_value`).
5. **Report content structure (fixed section order, both formats):** title/header (sample ID + depth) → prominent non-diagnostic disclaimer → Overall Summary → per-indicator Findings (12 indicators in-depth, 4 concise; each with observation, interpretation, confidence) → Strengths → Areas of Attention → closing disclaimer note.
6. **Key product behavior to design for:** a failed quality check does **not** block the report — the user always gets a full report plus visible quality caveats (`validation_results` verdicts + an extra areas-of-attention entry); rejection UI should read as "review advised," not "analysis failed."
7. **Error states:** `400` (file unreadable/corrupt, with human-readable `detail`), `413` (too large), `422` (invalid option values, FastAPI structured body). No auth of any kind (no login/token states to design yet).
8. **PDF export:** same analysis request with `output_format=pdf` returns a downloadable `application/pdf` (filename derived from `sample_id`) — export is a per-analysis action, not a stored-document library (no persistence exists server-side).
9. **Tone constraints binding on any UI copy:** cautious, non-diagnostic, no absolute claims — the backend hard-enforces a denylist (`DISALLOWED_TERMS`) on all generated text; UI copy should match that register (no "magical"/sensationalist framing, per the brief).
