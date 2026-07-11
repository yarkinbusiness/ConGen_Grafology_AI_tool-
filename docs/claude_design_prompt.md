# Claude Design brief — ConGen Grafology AI Tool

Paste everything below into Claude Design as the project prompt.

---

## What this product is

Design a web platform for the **ConGen Grafology AI Tool**, a support tool that helps professional graphologists, HR consultants, and trainers get a fast, structured *first read* of a handwriting sample. It is explicitly **not** a diagnostic tool and never claims to be — it produces a cautious, descriptive support report, never a definitive judgment about the person. The tone must communicate reliability, accuracy, and professionalism, and must actively avoid anything "magical," mystical, or sensationalist (no crystal-ball framing, no fortune-telling visuals, no overclaiming language in UI copy — even microcopy like button labels and empty states should read as measured and respectful, matching the calm, hedged register of the generated reports themselves).

Target users: professional graphologists, HR professionals, career/personal-development consultants, trainers, and other users with a genuine interest in graphology who want a first structured pass before applying their own expertise.

## Core user flow to design

1. **Upload** — a single handwriting sample as an image (JPEG/PNG) or PDF. Drag-and-drop plus a traditional file picker. Show accepted formats and a size limit (default 15 MB) up front, not just as an error after the fact.
2. **Options (optional, not required to proceed)** — choice of analysis depth: **Concise** (quick overview) vs **In-depth** (full section-by-section breakdown); an optional free-text sample/reference ID field.
3. **Processing** — this is a **synchronous** request: the user waits roughly a few seconds while the backend runs the full pipeline. Design a calm, confidence-inspiring processing/loading state (no fake progress bars implying multi-stage AI "thinking" theatrics — keep it honest and simple, e.g. a single steady indicator).
4. **Report view** — a well-organized, readable report with a **fixed structure that must be preserved exactly, in this order**:
   - Header (sample ID if given, chosen depth)
   - A prominent, unmissable disclaimer near the top stating this is a support tool, not a diagnosis, and does not replace a professional graphologist's judgment (this disclaimer is non-negotiable and must be visually present, not hidden in fine print or a tooltip)
   - Overall Summary
   - Per-indicator Findings — up to 12 indicators in-depth (4 in concise mode): pressure, slant, letter size, spacing, baseline movement, margins, rhythm, stroke continuity, overall organization, plus a couple more layout-derived indicators. Each finding has: an indicator label, a plain-language observation, an interpretive note (personality/communication/emotional/energy/organizational/social themes, always hedged — "may suggest," "can indicate," never "is" or "means"), and a confidence level (High/Medium/Low — design a clear, non-alarming way to show measurement confidence per indicator, since this is a heuristic analysis and confidence varies per sample)
   - Strengths (a short list)
   - Areas of Attention (a short list — frame neutrally, never as "problems" or "weaknesses")
   - A closing disclaimer note, mirroring the top one
5. **Export** — a clear action to export the report as a downloadable PDF, matching the same fixed section structure as the on-screen report (the PDF is not a separate design — same content, same order, exported).
6. **Error / edge states to design explicitly**:
   - Corrupt or unreadable file upload (clear, human, non-technical error message)
   - File too large
   - A sample that fails one or more automated quality checks (blur, contrast, low resolution) — **important product behavior**: this does *not* block the report. The user still gets a complete report, with a visible, non-alarming caveat noting which quality checks did not pass and that results should be reviewed with that in mind. Design this as "review advised," never as a hard failure/red error screen.
   - A multi-page PDF upload — only the first page is analyzed; show a visible, friendly note about this, not a silent limitation.

## Content and copy constraints (binding on any UI text you write)

- No diagnostic, absolute, or clinical language anywhere in generated or interface copy: avoid words like "disorder," "diagnosis," "always," "never," "you are," "this means you." Findings are written as possibilities and tendencies, never certainties.
- No sensationalist or "AI magic" framing — no glowing orbs, no mystical iconography, no "unlock your personality" style marketing copy. This is a professional analytical tool, closer in feel to a diagnostic-imaging report UI or a professional assessment platform than a consumer personality-quiz app.
- The non-diagnostic disclaimer must be genuinely visible at both the top and bottom of every report view, not just a footer link.
- Confidence indicators (High/Medium/Low per finding) should be shown honestly and unobtrusively — this is a classical heuristic analysis today, not a mature trained model, so avoid any visual language that overstates certainty (no "99% match" style badges).

## Technical facts to design against (do not deviate — this is what the real backend does)

- One file per request; no accounts, no login, no saved history — each analysis is a single, self-contained session (no persistence exists server-side today, so don't design a "past reports" library unless told this is a future phase).
- Depth options are exactly two: `concise` and `indepth`. There is no sliding scale or custom selection of which indicators to include.
- Processing is synchronous with no job IDs or background progress — design accordingly (no "check back later" flows).
- PDF export is a per-report action producing a downloadable file; it is not a stored/shareable link.
- No user accounts or authentication exist in this phase — do not design login/signup screens unless explicitly asked to plan for a later phase.

## Visual tone direction

Professional, calm, trustworthy — closer to a clinical/analytical report tool (think structured lab-report or assessment-platform aesthetics) than a playful consumer app. Legible typography suited to dense structured text (many short paragraphs per report), clear visual hierarchy between observation vs. interpretation vs. confidence per finding, and a restrained color palette that avoids anything overtly mystical (steer clear of purples/starfields/crystal-ball imagery commonly associated with "fortune telling" framing). Accessibility matters: this tool will be used by consultants and professionals in working contexts, so prioritize readability and clarity over decoration.

## Deliverables requested

- Upload screen (empty state, drag-active state, file-selected state, error states)
- Depth/options selection (as part of upload, or a distinct step — your call)
- Processing/loading state
- Report view — both concise and in-depth variants, showing the fixed section order above, including at least 2–3 example findings with confidence indicators, a strengths list, and an areas-of-attention list
- Quality-flagged report variant (showing the "review advised" caveat treatment)
- PDF export action/affordance
- Error states: corrupt file, oversized file, multi-page-PDF notice
