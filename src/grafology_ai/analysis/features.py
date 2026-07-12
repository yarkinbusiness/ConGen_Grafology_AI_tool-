"""Classical image-processing feature extraction for handwriting samples.

No trainable model exists yet (there is no real, labeled client dataset to
train one on -- see :mod:`grafology_ai.dataset.fixtures`). This module is a
heuristic stand-in: it measures observable, numeric properties directly
from pixels using classical image processing (Pillow + numpy only, no
OpenCV/scipy), producing a :class:`Features` record per sample. It later
either feeds a real trained model as input features, or serves as a
baseline the trained model is evaluated against.

This module deliberately stops at *measurement*. It reports numbers (a
slant angle, a stroke-width statistic, a margin in pixels) without
attaching graphological interpretation or judgment language to them --
turning measurements into indicator labels/interpretations is a separate,
later concern (see ``docs/labeling_rubric.md`` for the indicator
vocabulary this module's measurements are meant to eventually support).

Pipeline overview (see :func:`analyze`):

1. Binarize the image into an ink/background mask with a from-scratch
   Otsu threshold (:func:`_otsu_threshold`).
2. Detect approximate text "lines" from the row-wise ink-density profile
   (:func:`_detect_line_bands`), which drives line spacing and baseline
   slope.
3. Detect approximate "strokes" via run-length analysis of the binarized
   mask -- horizontal runs (row-wise) approximate local stroke thickness,
   vertical runs (column-wise) approximate stroke/letter height -- rather
   than full 2D connected-component labeling, which would require either
   an added dependency or an expensive from-scratch flood fill. This is a
   deliberate simplification; see :func:`_run_lengths_rowwise`.
4. Estimate slant via a projection-profile shear search
   (:func:`_estimate_slant_degrees`): the candidate shear angle whose
   correction makes ink pixels stack into the sharpest column histogram is
   taken as the dominant stroke angle.
5. Compute margins and ink density directly from the ink bounding box.

Every measurement is approximate by construction -- this is a heuristic
baseline, not a diagnostic tool -- and the confidence heuristic in
:func:`_build_confidence` is documented at its definition.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from PIL import Image

ImageInput = Union[Image.Image, str, "os.PathLike[str]"]

# --- Otsu / binarization -------------------------------------------------

#: Number of histogram bins used by the from-scratch Otsu threshold search.
#: 256 matches the natural resolution of 8-bit grayscale pixel values; for
#: non-pixel-value inputs (e.g. gap lengths in pixels, see
#: :func:`_estimate_word_spacing`) this still gives ample resolution for
#: typical handwriting-sample dimensions.
_OTSU_HISTOGRAM_BINS = 256

# --- Slant estimation ------------------------------------------------------

#: Below this many ink pixels, a projection-profile slant estimate is
#: considered unreliable (too few strokes to establish a dominant angle),
#: and :func:`_estimate_slant_degrees` short-circuits to 0.0 degrees.
_MIN_INK_PIXELS_FOR_SLANT = 30

#: Candidate shear angles (degrees from vertical) searched when estimating
#: slant. +/-60 degrees comfortably brackets any handwriting slant that
#: could plausibly be called "handwriting" rather than sideways text; 1
#: degree steps give ample resolution given the generous tolerance this
#: measurement is documented to have.
_SLANT_CANDIDATE_DEGREES: np.ndarray = np.arange(-60.0, 60.0 + 1e-9, 1.0)

#: Fixed histogram bin width (pixels) used when scoring each candidate
#: shear angle. A *fixed* width (rather than a fixed bin *count* over a
#: range that itself grows with the shear) is important: it keeps the
#: peakiness score comparable across candidate angles instead of biasing
#: the search toward extreme angles whose wider sheared-x range would
#: otherwise be binned more coarsely.
_SLANT_BIN_WIDTH_PX = 2.0

#: Ink pixels are randomly subsampled to at most this many points before
#: the slant search, since the search cost is O(n_candidates * n_pixels)
#: and dense samples can have far more ink pixels than are needed to
#: estimate one dominant angle. The subsample is drawn from a
#: fixed-seed Generator so :func:`analyze` stays deterministic.
_MAX_INK_PIXELS_FOR_SLANT_SAMPLE = 20_000
_SLANT_SAMPLE_SEED = 0

# --- Run-length statistics (stroke width / letter size) --------------------

#: Run lengths (in pixels) are trimmed to this percentile range before
#: computing mean/std, once enough runs exist (see
#: _MIN_RUNS_FOR_PERCENTILE_TRIM), to reduce the influence of a handful of
#: outlier runs (e.g. two strokes that happen to touch on one row).
_RUN_PERCENTILE_TRIM = (5.0, 95.0)
_MIN_RUNS_FOR_PERCENTILE_TRIM = 20

# --- Line-band detection (line spacing / baseline slope) -------------------

#: Minimum smoothing window (rows) applied to the row ink-density profile
#: before thresholding it into line bands, so that small intra-word gaps
#: (a lifted pen between letters) don't fragment one visual line into
#: several detected bands. Scales with image height (see
#: :func:`_detect_line_bands`) but never drops below this floor.
_LINE_SMOOTH_MIN_WINDOW = 3

#: A row is considered part of a text line once its smoothed ink count
#: exceeds this fraction of the profile's peak smoothed value.
_LINE_BAND_THRESHOLD_FRACTION = 0.15

#: Minimum number of ink-containing columns within a line band required
#: before attempting a baseline-slope line fit for that band; fitting a
#: line through fewer points is not meaningful.
_MIN_COLUMNS_FOR_BASELINE_FIT = 5

# --- Word-spacing detection -------------------------------------------------

#: Minimum number of horizontal column-gaps (within line bands) required
#: before attempting to Otsu-split them into "letter" vs "word" gaps; with
#: fewer gaps than this, the split is not meaningful and all detected gaps
#: are simply averaged instead (see :func:`_estimate_word_spacing`).
_MIN_GAPS_FOR_OTSU_SPLIT = 4

# --- Confidence scaling ------------------------------------------------------

#: "Full confidence" sample-size thresholds used by :func:`_confidence_scale`
#: for each statistically-estimated indicator: confidence ramps linearly
#: from 0 up to 1.0 as the number of underlying observations (ink pixels,
#: runs, gaps, line bands, baseline-fit columns) approaches the threshold,
#: reflecting that a measurement built from very few observations is less
#: reliable than one built from many. These are practical, not derived
#: from any formal statistical procedure.
_CONF_FULL_AT_SLANT_PIXELS = 1000
_CONF_FULL_AT_RUNS = 50
_CONF_FULL_AT_LINE_GAPS = 3
_CONF_FULL_AT_WORD_GAPS = 5
_CONF_FULL_AT_BASELINE_COLUMNS = 100

# --- Rhythm regularity -------------------------------------------------------

#: Minimum number of observations (per-band heights, per-band widths, or
#: word gaps) a single ingredient of the :func:`_build_rhythm_regularity`
#: composite needs before its coefficient-of-variation score is considered
#: meaningful at all -- a "spread" computed from 0 or 1 points isn't a
#: regularity measurement, it's noise. Ingredients with fewer observations
#: than this are dropped from the composite (see the weight-renormalization
#: in :func:`_build_rhythm_regularity`) rather than forced to a fabricated
#: score.
_RHYTHM_MIN_OBSERVATIONS = 2

#: Equal weights for the three rhythm ingredients -- per-line-band ink-run
#: (letter) height, word-gap length, and per-line-band stroke width. The
#: labeling rubric's Rhythm definition ("how evenly stroke shapes, sizes,
#: and spacing repeat") does not prioritize any one of these over the
#: others, so they are weighted equally by default; when an ingredient is
#: unusable (see :data:`_RHYTHM_MIN_OBSERVATIONS`) the remaining weights are
#: renormalized rather than treating the missing ingredient as "perfectly
#: regular".
_RHYTHM_WEIGHT_LINE_HEIGHT = 1.0 / 3.0
_RHYTHM_WEIGHT_WORD_GAP = 1.0 / 3.0
_RHYTHM_WEIGHT_STROKE_WIDTH = 1.0 / 3.0

#: "Full confidence" sample-size threshold for the per-line-band ingredients
#: of rhythm_regularity (letter height, stroke width): confidence ramps up
#: as the number of line bands that contributed a usable value approaches
#: this many, matching the spirit of :data:`_CONF_FULL_AT_LINE_GAPS` (a
#: regularity read off very few bands is not well supported). The word-gap
#: ingredient reuses :data:`_CONF_FULL_AT_WORD_GAPS` since it is built from
#: the exact same gap list as ``word_spacing_mean``.
_CONF_FULL_AT_RHYTHM_BANDS = 3

# --- stroke connectedness -----------------------------------------------------

#: Segments-per-word ratio (ink-column-run segments in a line band, divided
#: by that band's estimated word count -- see
#: :func:`_stroke_connectedness_per_band`) at or above which
#: stroke_connectedness saturates at 0.0 (maximally broken/lifted). A ratio
#: of 1.0 (every word drawn as one unbroken run) is, by definition, the
#: maximally connected case (score 1.0); this constant marks the opposite
#: end of the scale. A typical short handwritten word rendered as roughly
#: four separate disconnected pieces (e.g. every letter drawn as its own
#: printed, unjoined stroke) already reads as about as broken as
#: handwriting gets, so ratios at or beyond it are treated as equally
#: "fully broken" rather than driving the score arbitrarily negative. The
#: score decays linearly between these two ratios (1.0 -> 1.0 score,
#: this constant -> 0.0 score).
_STROKE_CONNECTEDNESS_MAX_SEGMENTS_PER_WORD = 4.0

#: "Full confidence" sample-size threshold for stroke_connectedness: total
#: ink-column-run segments, summed across every line band that yielded a
#: usable word-count estimate. Matches the spirit of :data:`_CONF_FULL_AT_RUNS`
#: -- a connectedness score built from a handful of segments is much less
#: reliable than one built from many.
_CONF_FULL_AT_STROKE_SEGMENTS = 20

# --- overall organization ------------------------------------------------------

#: Minimum number of per-band observations (inter-band gaps, per-band
#: left-edge columns, or per-band baseline slopes) a single ingredient of
#: the :func:`_build_organization_score` composite needs before its
#: consistency sub-score is considered meaningful at all -- the same
#: reasoning as :data:`_RHYTHM_MIN_OBSERVATIONS`: a "spread" computed from
#: 0 or 1 points isn't a consistency measurement, it's noise. An
#: ingredient with fewer observations than this is dropped from the
#: composite and the remaining weights renormalized (see
#: :func:`_build_organization_score`), the same pattern
#: :func:`_build_rhythm_regularity` uses.
_ORG_MIN_OBSERVATIONS = 2

#: Equal weights for the three organization ingredients -- inter-band
#: vertical spacing consistency, per-band left-edge (line-start) column
#: alignment consistency, and per-band baseline-slope consistency. The
#: labeling rubric's Overall Organization definition ("alignment of
#: lines, use of space... apparent planning of layout") does not
#: prioritize any one of these over the others, so -- matching the
#: equal-weighting-with-renormalization choice already made for
#: :data:`_RHYTHM_WEIGHT_LINE_HEIGHT` and friends -- they are weighted
#: equally by default; when an ingredient is unusable (see
#: :data:`_ORG_MIN_OBSERVATIONS`) the remaining weights are renormalized
#: rather than treating a missing ingredient as "perfectly organized".
_ORG_WEIGHT_BAND_SPACING = 1.0 / 3.0
_ORG_WEIGHT_LEFT_EDGE = 1.0 / 3.0
_ORG_WEIGHT_BASELINE_SLOPE = 1.0 / 3.0

#: A per-band left-edge standard deviation at or beyond this fraction of
#: the ink content's bounding-box width is treated as maximally ragged
#: line-start alignment (score 0.0 in :func:`_build_organization_score`);
#: the ingredient decays linearly from 1.0 (std == 0, every line starts at
#: exactly the same column) down to 0.0 as the std approaches this
#: fraction of content width. Expressing the threshold as a *fraction of
#: content width* -- rather than a fixed pixel count -- keeps it
#: meaningful regardless of image resolution or how wide the handwriting
#: sample happens to be, the same concern :data:`_SLANT_BIN_WIDTH_PX`'s
#: docstring raises about fixed-vs-scaled measurements.
_ORG_LEFT_EDGE_MAX_STD_FRACTION = 0.25

#: A per-band baseline-slope standard deviation (degrees) at or beyond
#: this many degrees is treated as maximally wandering/inconsistent
#: line-to-line baselines (score 0.0 in :func:`_build_organization_score`);
#: decays linearly from 1.0 (std == 0, every line's baseline trend is
#: identical) down to 0.0 as the std approaches this many degrees.
#: Degrees are already resolution-independent (unlike a raw pixel
#: threshold), so no additional normalization is needed here. Chosen much
#: tighter than the +/-60 degree :data:`_SLANT_CANDIDATE_DEGREES` search
#: range used for overall stroke slant, since line-to-line baseline
#: *wander* is a subtler, smaller-magnitude organization signal than
#: overall stroke slant.
_ORG_BASELINE_SLOPE_STD_MAX_DEGREES = 8.0

#: "Full confidence" line-band-count threshold for organization_score:
#: confidence ramps up via :func:`_confidence_scale` as the number of
#: *detected* line bands (not any per-pixel/per-run observation count)
#: approaches this many. Organization is inherently a cross-line
#: comparison -- reading it off a single detected line is close to
#: meaningless, since there is nothing to compare that line against -- so
#: this threshold is deliberately set high enough that a single detected
#: band always lands strictly below 0.5 confidence (1 / 4 == 0.25).
_CONF_FULL_AT_ORGANIZATION_BANDS = 4


@dataclass(frozen=True)
class Features:
    """Measurable graphological indicators extracted from a handwriting image.

    Every field is a plain, directly-measured numeric quantity -- no
    interpretation or judgment language attached (see the module
    docstring). Turning these into graphological indicator labels (e.g.
    "right slant", "heavy pressure") is deliberately out of scope here.

    Attributes:
        slant_angle_degrees: Average angle of stroke segments relative to
            vertical. Positive means a rightward lean (top of strokes
            shifted right relative to their base), negative means a
            leftward lean, matching the sign convention used by
            :mod:`grafology_ai.dataset.fixtures`.
        stroke_width_mean: Mean ink-stroke thickness in pixels, a
            pressure proxy (thicker strokes are read from the image as
            heavier pen pressure; true pressure cannot be recovered from a
            static image, only inferred from stroke weight).
        stroke_width_std: Standard deviation of stroke thickness, in
            pixels -- a proxy for how *consistent* pressure is across the
            sample.
        letter_size_estimate: Proxy for average character/stroke height,
            in pixels.
        line_spacing_mean: Average vertical whitespace gap between
            detected text lines, in pixels.
        word_spacing_mean: Average horizontal gap between detected
            word/stroke clusters within a line, in pixels.
        baseline_slope_degrees: Trend of the baseline across a line, in
            degrees. Positive means the baseline rises (moves toward
            smaller row/pixel-y, i.e. up the page) left-to-right;
            negative means it falls.
        margin_left_px: Whitespace to the left of the ink content's
            bounding box, in pixels.
        margin_right_px: Whitespace to the right of the ink content's
            bounding box, in pixels.
        margin_top_px: Whitespace above the ink content's bounding box,
            in pixels.
        margin_bottom_px: Whitespace below the ink content's bounding
            box, in pixels.
        ink_density: Fraction of foreground (ink) pixels within the
            content bounding box, in [0, 1] -- a raw ink-coverage
            measurement. It is still computed and populated here, but it
            no longer backs any interpretation-layer indicator: the
            ``"rhythm"`` and ``"stroke_continuity"`` indicators are keyed
            off the dedicated ``rhythm_regularity`` and
            ``stroke_connectedness`` fields below instead (see
            ``docs/labeling_rubric.md``'s "Rhythm" / "Stroke Continuity"
            indicators).
        rhythm_regularity: Regularity of repetition across the sample, in
            [0, 1] where 1.0 means highly regular/rhythmic, per
            ``docs/labeling_rubric.md``'s "Rhythm" indicator ("how evenly
            stroke shapes, sizes, and spacing repeat"). An
            inverse-coefficient-of-variation composite over per-line-band
            ink-run (letter) heights, word-gap lengths, and per-line-band
            stroke widths -- see :func:`_build_rhythm_regularity`.
        stroke_connectedness: How joined/continuous vs. lifted/broken
            handwriting strokes are, in [0, 1] where 1.0 means strokes
            read as fully joined (few pen lifts) and 0.0 means strokes
            read as fully segmented/printed, per
            ``docs/labeling_rubric.md``'s "Stroke Continuity" indicator.
            Within each detected line band, compares the count of
            ink-column-run segments to an Otsu-split estimate of how many
            "words" those segments form (more segments per word implies
            more pen lifts) -- see :func:`_stroke_connectedness_per_band`
            and :func:`_build_stroke_connectedness`.
        organization_score: Overall layout organization of the sample, in
            [0, 1] where 1.0 means a highly organized layout (evenly
            spaced lines, consistently aligned line starts, level
            baselines line to line), per ``docs/labeling_rubric.md``'s
            "Overall Organization" indicator ("alignment of lines, use of
            space, and apparent planning of layout"). A weighted composite
            of three per-line-band consistency ingredients -- inter-band
            vertical spacing, left-edge (line-start) column alignment, and
            baseline-slope consistency (reusing
            :func:`_estimate_baseline_slope`'s per-band fits) -- see
            :func:`_build_organization_score`. Unlike every other
            statistically-estimated field, this field's confidence scales
            with the *number of detected line bands* rather than a
            per-pixel/per-run count, since organization is inherently a
            cross-line comparison a single line cannot support -- see
            :data:`_CONF_FULL_AT_ORGANIZATION_BANDS`.
        confidence: Maps each of the above field names to a 0-1
            confidence score. See :func:`_build_confidence` for the
            heuristic (in short: measurements built from very few
            underlying observations -- e.g. a near-empty image with few
            detected strokes or lines -- get a low score).
    """

    slant_angle_degrees: float
    stroke_width_mean: float
    stroke_width_std: float
    letter_size_estimate: float
    line_spacing_mean: float
    word_spacing_mean: float
    baseline_slope_degrees: float
    margin_left_px: float
    margin_right_px: float
    margin_top_px: float
    margin_bottom_px: float
    ink_density: float
    rhythm_regularity: float
    stroke_connectedness: float
    organization_score: float
    confidence: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-safe ``dict`` of every field, by name.

        Every field here is already a JSON-safe scalar or a ``dict[str,
        float]`` (:attr:`confidence`), so this is a flat conversion --
        ``confidence`` is copied into a new plain ``dict`` rather than
        referencing the original (see the module-level serialization
        contract in :mod:`grafology_ai.run_analysis`, whose
        :meth:`~grafology_ai.run_analysis.AnalysisResult.to_dict` composes
        this method for the nested ``features`` entry). Kept as an
        explicit field-by-field mapping (rather than
        ``dataclasses.asdict``) so a future field addition is caught by
        the completeness check in ``tests/test_serialization.py`` instead
        of silently round-tripping through a generic recursive helper.
        """
        return {
            "slant_angle_degrees": self.slant_angle_degrees,
            "stroke_width_mean": self.stroke_width_mean,
            "stroke_width_std": self.stroke_width_std,
            "letter_size_estimate": self.letter_size_estimate,
            "line_spacing_mean": self.line_spacing_mean,
            "word_spacing_mean": self.word_spacing_mean,
            "baseline_slope_degrees": self.baseline_slope_degrees,
            "margin_left_px": self.margin_left_px,
            "margin_right_px": self.margin_right_px,
            "margin_top_px": self.margin_top_px,
            "margin_bottom_px": self.margin_bottom_px,
            "ink_density": self.ink_density,
            "rhythm_regularity": self.rhythm_regularity,
            "stroke_connectedness": self.stroke_connectedness,
            "organization_score": self.organization_score,
            "confidence": dict(self.confidence),
        }


#: The subset of :class:`Features` fields the `confidence` dict must cover
#: (i.e. every field except `confidence` itself).
FEATURE_CONFIDENCE_KEYS: tuple[str, ...] = tuple(
    name for name in Features.__dataclass_fields__ if name != "confidence"
)


# --- image loading -----------------------------------------------------------


def _load_image(image: ImageInput) -> Image.Image:
    """Return a :class:`PIL.Image.Image` for either an image or a path."""
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a 2D float64 numpy array of grayscale values."""
    return np.asarray(image.convert("L"), dtype=np.float64)


# --- Otsu threshold (generic: pixel values or run/gap lengths) -------------


def _otsu_threshold(values: np.ndarray) -> float:
    """Return an Otsu threshold splitting ``values`` into two classes.

    A minimal, dependency-free (no scipy/skimage) implementation: build a
    histogram, then pick the bin boundary that maximizes between-class
    variance. This is written generically over any 1D numeric array, not
    just 0-255 grayscale pixels, so it can also split a distribution of
    gap lengths into "small" and "large" clusters (see
    :func:`_estimate_word_spacing`).

    Returns the input's minimum value if ``values`` is empty or constant
    (there is nothing to split).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    v_min, v_max = float(values.min()), float(values.max())
    if v_min == v_max:
        return v_min

    hist, bin_edges = np.histogram(values, bins=_OTSU_HISTOGRAM_BINS, range=(v_min, v_max))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    sum_total = float(np.sum(hist * bin_centers))

    weight_background = 0.0
    sum_background = 0.0
    best_threshold = float(bin_centers[0])
    best_variance = -1.0

    for i in range(hist.size):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background += hist[i] * bin_centers[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = float(bin_centers[i])

    return best_threshold


def _binarize(grayscale: np.ndarray) -> np.ndarray:
    """Binarize a grayscale array into a boolean ink mask (True = ink).

    Uses :func:`_otsu_threshold` to split pixels into two classes, then
    assumes ink is the *minority* class by pixel count -- a safe
    assumption for handwriting samples, where ink coverage is sparse
    relative to background/page, regardless of whether ink is rendered
    dark-on-light or light-on-dark.

    A perfectly uniform image (no tonal variation at all) is treated as
    containing no ink, since Otsu's threshold is degenerate in that case.
    """
    if grayscale.size == 0 or grayscale.max() == grayscale.min():
        return np.zeros_like(grayscale, dtype=bool)

    threshold = _otsu_threshold(grayscale)
    low_mask = grayscale <= threshold
    high_mask = ~low_mask
    return low_mask if low_mask.sum() <= high_mask.sum() else high_mask


# --- run-length analysis (stroke width / letter size / word gaps) ----------


def _runs_1d(values: np.ndarray) -> np.ndarray:
    """Return an (n_runs, 2) array of [start, end) index pairs of True runs.

    ``end`` is exclusive, so a run's length is ``end - start``.
    """
    if values.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate(([False], values, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return np.stack([starts, ends], axis=1)


def _run_lengths_rowwise(mask: np.ndarray) -> np.ndarray:
    """Lengths of every True-run within each row of a 2D boolean array.

    Each row is treated independently (a run never spans across a row
    boundary): every row is edge-padded with False before differencing, so
    a run that reaches a row's last column is still closed there. Fully
    vectorized (no per-row Python loop) via a single pad + diff + where
    over the whole array: ``np.where`` scans in row-major order, so the
    rising-edge ("start") and falling-edge ("end") indices it returns are
    already in matching row-by-row, left-to-right order and can be paired
    positionally.

    Applying this to ``mask.T`` gives column-wise (vertical) run lengths
    instead.
    """
    if mask.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    diff = np.diff(padded.astype(np.int8), axis=1)
    _, start_cols = np.where(diff == 1)
    _, end_cols = np.where(diff == -1)
    return (end_cols - start_cols).astype(np.int64)


def _robust_run_stats(runs: np.ndarray, scale_factor: float) -> tuple[float, float, int]:
    """Mean/std of ``runs`` (scaled by ``scale_factor``), trimmed of outliers.

    ``scale_factor`` corrects for the fact that a run-length measured
    along a fixed row/column through a *slanted* stroke overstates the
    stroke's true perpendicular thickness/length by roughly
    ``1 / cos(slant)``; callers pass ``cos(slant_angle)`` to undo that.

    Once enough runs exist, the extreme 5th/95th percentiles are dropped
    before computing statistics, so that a small number of runs
    contaminated by two strokes touching (an accidental horizontal/
    vertical merge) don't dominate the mean. Returns ``(0.0, 0.0, 0)`` for
    an empty input.
    """
    if runs.size == 0:
        return 0.0, 0.0, 0

    values = runs.astype(np.float64) * scale_factor
    if values.size >= _MIN_RUNS_FOR_PERCENTILE_TRIM:
        low, high = np.percentile(values, _RUN_PERCENTILE_TRIM)
        trimmed = values[(values >= low) & (values <= high)]
        if trimmed.size > 0:
            values = trimmed

    return float(values.mean()), float(values.std()), int(runs.size)


# --- slant estimation --------------------------------------------------------


def _estimate_slant_degrees(
    ys: np.ndarray, xs: np.ndarray, bands: list[tuple[int, int]]
) -> tuple[float, int]:
    """Estimate the dominant stroke slant via a projection-profile shear search.

    Crucially, this estimates slant *within each detected line band*
    (see :func:`_detect_line_bands`) and combines the per-line estimates,
    rather than searching over the whole image's ink pixels at once. A
    whole-image search confounds real within-line slant with the
    (unrelated) vertical offset between separate lines: shearing by
    ``tan(candidate) * y`` shifts each line by a different amount since
    each line sits at a different y, which can spuriously "align" text
    across lines at the wrong candidate angle -- especially for
    repetitive synthetic content -- and swamp the much smaller, real
    within-line signal. Restricting the search to one line band's narrow
    y-range at a time avoids that. Per-band angles are combined with a
    pixel-count-weighted average.

    If no line bands were detected (e.g. very sparse ink), falls back to
    a single whole-image search over all ink pixels.

    Returns ``(angle_degrees, n_pixels_used)``; the pixel count feeds the
    confidence heuristic.
    """
    if not bands:
        return _estimate_slant_for_pixels(ys, xs)

    angles: list[float] = []
    weights: list[int] = []
    for start, end in bands:
        in_band = (ys >= start) & (ys < end)
        if not np.any(in_band):
            continue
        angle, n = _estimate_slant_for_pixels(ys[in_band], xs[in_band])
        if n >= _MIN_INK_PIXELS_FOR_SLANT:
            angles.append(angle)
            weights.append(n)

    if not angles:
        return _estimate_slant_for_pixels(ys, xs)

    mean_angle = float(np.average(angles, weights=weights))
    return mean_angle, int(sum(weights))


def _estimate_slant_for_pixels(ys: np.ndarray, xs: np.ndarray) -> tuple[float, int]:
    """Projection-profile shear search for the dominant slant of one pixel set.

    For a stroke leaning at angle ``theta`` from vertical, a pixel at
    ``(x, y)`` on that stroke satisfies ``x + tan(theta) * y == constant``
    along the whole stroke (this matches the sign convention drawn by
    :mod:`grafology_ai.dataset.fixtures`: positive ``theta`` means the top
    of a stroke, at smaller y, is shifted to larger x, i.e. a rightward
    lean). So for each candidate angle, this shears every ink pixel's x
    coordinate by ``tan(candidate) * y`` and scores how "peaky" (sharply
    concentrated into narrow columns) the resulting 1D histogram is; the
    candidate whose shear best concentrates ink into columns -- meaning it
    best cancels out the real slant -- is taken as the slant estimate.

    This is a coarse statistical estimate over ink pixels, not a Hough
    transform over individual line segments, and assumes a single
    dominant slant across the given pixel set (mixed slants would average
    out rather than being reported distinctly).

    Returns ``(angle_degrees, n_pixels_used)``. Returns ``(0.0, n)``
    without searching if there are too few ink pixels to trust an
    estimate.
    """
    n = xs.size
    if n < _MIN_INK_PIXELS_FOR_SLANT:
        return 0.0, n

    if n > _MAX_INK_PIXELS_FOR_SLANT_SAMPLE:
        rng = np.random.default_rng(_SLANT_SAMPLE_SEED)
        sample_idx = rng.choice(n, size=_MAX_INK_PIXELS_FOR_SLANT_SAMPLE, replace=False)
        sample_ys = ys[sample_idx].astype(np.float64)
        sample_xs = xs[sample_idx].astype(np.float64)
    else:
        sample_ys = ys.astype(np.float64)
        sample_xs = xs.astype(np.float64)

    best_angle = 0.0
    best_score = -1.0
    for angle in _SLANT_CANDIDATE_DEGREES:
        theta = math.radians(float(angle))
        sheared_x = sample_xs + math.tan(theta) * sample_ys
        span = float(sheared_x.max() - sheared_x.min())
        n_bins = max(1, int(math.ceil(span / _SLANT_BIN_WIDTH_PX)))
        counts, _ = np.histogram(sheared_x, bins=n_bins)
        score = float(np.sum(counts.astype(np.float64) ** 2))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle, n


# --- line-band detection (line spacing / baseline slope) -------------------


def _detect_line_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Detect approximate text-line row bands from the row ink-density profile.

    Sums ink pixels per row, smooths that profile with a small moving
    average (so a lifted pen between letters/words doesn't fragment one
    visual line into several bands), then treats any row whose smoothed
    density exceeds :data:`_LINE_BAND_THRESHOLD_FRACTION` of the profile's
    peak as "inside a line". Contiguous runs of such rows are returned as
    ``[start, end)`` row-index bands, in top-to-bottom order.

    Returns an empty list if the mask has no ink at all.
    """
    if mask.shape[0] == 0:
        return []
    row_counts = mask.sum(axis=1).astype(np.float64)
    if row_counts.max() <= 0:
        return []

    window = max(_LINE_SMOOTH_MIN_WINDOW, mask.shape[0] // 100)
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(row_counts, kernel, mode="same")

    threshold = _LINE_BAND_THRESHOLD_FRACTION * smoothed.max()
    line_rows = smoothed > threshold
    runs = _runs_1d(line_rows)
    return [(int(start), int(end)) for start, end in runs]


def _band_gaps(bands: list[tuple[int, int]]) -> list[float]:
    """Return the individual whitespace gaps (rows) between consecutive line bands.

    Extracted as its own function (rather than inlined in
    :func:`_mean_band_gap`) so callers that need the raw list of gap
    lengths -- not just their mean -- can reuse it without re-deriving it;
    the same "expose the list, make the mean a thin wrapper" pattern
    :func:`_word_gaps` uses relative to :func:`_estimate_word_spacing`.
    See :func:`_build_organization_score`, which needs the spread of
    inter-band gaps, not merely their average.

    Returns an empty list if fewer than two bands were detected, or if no
    positive gap exists between any pair of consecutive bands.
    """
    if len(bands) < 2:
        return []
    gaps = [
        bands[i + 1][0] - bands[i][1]
        for i in range(len(bands) - 1)
        if bands[i + 1][0] - bands[i][1] > 0
    ]
    return [float(g) for g in gaps]


def _mean_band_gap(bands: list[tuple[int, int]]) -> tuple[float, int]:
    """Mean whitespace gap (rows) between consecutive line bands.

    Thin wrapper around :func:`_band_gaps`. Returns ``(0.0, 0)`` if fewer
    than two bands were detected, or no positive gap exists -- a spacing
    measurement needs at least two lines to measure a gap between.
    """
    gaps = _band_gaps(bands)
    if not gaps:
        return 0.0, 0
    return float(np.mean(gaps)), len(gaps)


def _baseline_slopes_per_band(
    mask: np.ndarray, bands: list[tuple[int, int]]
) -> tuple[list[float], list[int]]:
    """Per-line-band pixel-space baseline slope and its column-count weight.

    For each detected line band, finds the bottom-most ink row at every
    ink-containing column (a descender-insensitive proxy for that column's
    position on the baseline) and fits a line to (column, bottom_row) via
    least squares -- the same per-band fit :func:`_estimate_baseline_slope`
    combines into one whole-sample estimate. Extracted as its own function
    (the same "expose the per-band list, make the aggregate a thin
    wrapper" pattern as :func:`_word_gaps` relative to
    :func:`_estimate_word_spacing`) so callers that need the *spread* of
    per-band slopes -- not just their combined average -- can reuse it
    without re-deriving it; see :func:`_build_organization_score`.

    A band with fewer than :data:`_MIN_COLUMNS_FOR_BASELINE_FIT`
    ink-containing columns is skipped entirely (not padded with a
    fabricated 0.0 slope), matching :func:`_per_band_run_stats`'s
    convention for unmeasurable bands.

    Returns ``(slopes, weights)``, parallel lists of one entry per band
    that had enough ink-containing columns to fit a line: raw
    pixel-space slopes (rows of vertical shift per column, *before* the
    degree conversion and sign negation :func:`_estimate_baseline_slope`
    applies), and each fit's ink-containing-column count (usable both as
    a combination weight and as an observation-count confidence input).
    """
    slopes: list[float] = []
    weights: list[int] = []

    for start, end in bands:
        band = mask[start:end, :]
        if band.shape[0] == 0:
            continue
        col_has_ink = band.any(axis=0)
        cols = np.nonzero(col_has_ink)[0]
        if cols.size < _MIN_COLUMNS_FOR_BASELINE_FIT:
            continue

        flipped = band[::-1, :]
        first_true_from_bottom = np.argmax(flipped, axis=0)
        bottom_row_local = band.shape[0] - 1 - first_true_from_bottom
        bottom_rows = bottom_row_local[cols].astype(np.float64) + start

        slope_pixel = float(np.polyfit(cols.astype(np.float64), bottom_rows, 1)[0])
        slopes.append(slope_pixel)
        weights.append(int(cols.size))

    return slopes, weights


def _estimate_baseline_slope(
    mask: np.ndarray, bands: list[tuple[int, int]]
) -> tuple[float, int]:
    """Estimate baseline trend (rise/fall across a line), in degrees.

    Thin wrapper around :func:`_baseline_slopes_per_band`: combines its
    per-band pixel-space slopes with a weighted average (weighted by the
    number of columns each band's fit used), then converts from a
    pixel-space slope to degrees, negated so that a baseline that rises
    left-to-right (row index *decreases* as column increases, since row 0
    is the top of the image) is reported as a *positive* degree value.

    Returns ``(0.0, 0)`` if no band had enough ink-containing columns
    (:data:`_MIN_COLUMNS_FOR_BASELINE_FIT`) to fit a line.
    """
    slopes, weights = _baseline_slopes_per_band(mask, bands)
    if not slopes:
        return 0.0, 0

    mean_slope_pixel = float(np.average(slopes, weights=weights))
    return -math.degrees(math.atan(mean_slope_pixel)), int(sum(weights))


# --- word spacing ------------------------------------------------------------


def _word_gaps(mask: np.ndarray, bands: list[tuple[int, int]]) -> list[float]:
    """Return the individual "word"-scale horizontal gap lengths (pixels).

    Within each line band, projects ink onto columns (does this column
    contain any ink within the band?) and measures the horizontal gaps
    between consecutive ink-column runs -- i.e. between stroke/letter
    clusters. Handwriting has two gap scales (tight intra-word/
    letter-to-letter gaps, and wider inter-word gaps); once enough gaps
    are collected, :func:`_otsu_threshold` splits them into these two
    clusters and only the larger ("word") cluster is returned. With too
    few gaps to split meaningfully, all detected gaps are returned as a
    fallback.

    Extracted as its own function (rather than inlined in
    :func:`_estimate_word_spacing`) so callers that need the raw list of
    gap lengths -- not just their mean -- can reuse it without
    re-deriving it; see :func:`_build_rhythm_regularity`, which needs the
    spread of word-gap lengths, not merely their average.

    Returns an empty list if no gaps were found at all.
    """
    all_gaps: list[int] = []
    for start, end in bands:
        band = mask[start:end, :]
        if band.shape[0] == 0:
            continue
        col_has_ink = band.any(axis=0)
        clusters = _runs_1d(col_has_ink)
        if clusters.shape[0] < 2:
            continue
        gaps = clusters[1:, 0] - clusters[:-1, 1]
        all_gaps.extend(int(g) for g in gaps if g > 0)

    if not all_gaps:
        return []

    gaps_array = np.array(all_gaps, dtype=np.float64)
    if gaps_array.size < _MIN_GAPS_FOR_OTSU_SPLIT:
        return [float(g) for g in gaps_array]

    threshold = _otsu_threshold(gaps_array)
    word_gaps = gaps_array[gaps_array > threshold]
    if word_gaps.size == 0:
        word_gaps = gaps_array
    return [float(g) for g in word_gaps]


def _estimate_word_spacing(
    mask: np.ndarray, bands: list[tuple[int, int]]
) -> tuple[float, int]:
    """Estimate average horizontal word-gap (pixels) within detected lines.

    Thin wrapper around :func:`_word_gaps`: averages the word-scale gap
    list it returns. Returns ``(0.0, 0)`` if no gaps were found at all.
    """
    gaps = _word_gaps(mask, bands)
    if not gaps:
        return 0.0, 0
    return float(np.mean(gaps)), len(gaps)


# --- rhythm regularity --------------------------------------------------------


def _per_band_run_stats(
    mask: np.ndarray, bands: list[tuple[int, int]], slant_correction: float
) -> tuple[list[float], list[float]]:
    """Per-line-band mean ink-run height and stroke width, one entry per band.

    Reuses the same run-length machinery as the whole-image
    ``letter_size_estimate``/``stroke_width_mean`` measurements
    (:func:`_run_lengths_rowwise`, :func:`_robust_run_stats`), but scoped
    to one line band at a time: column-wise (vertical) runs within a band
    approximate that band's letter/ink-run height, row-wise (horizontal)
    runs approximate that band's stroke width. This is what lets
    :func:`_build_rhythm_regularity` measure regularity *across* bands
    (does letter size/stroke width stay consistent line to line?) rather
    than only ever seeing one whole-sample average.

    A band that yields no measurable runs is simply skipped (not padded
    with a 0.0) -- an unmeasured band should not be scored as
    "irregular".

    Returns ``(heights, widths)``, each a plain list of per-band means
    (possibly of different lengths, since a band could yield a usable
    horizontal-run measurement but not a vertical one, or vice versa).
    """
    heights: list[float] = []
    widths: list[float] = []
    for start, end in bands:
        band = mask[start:end, :]
        if band.shape[0] == 0:
            continue

        row_runs = _run_lengths_rowwise(band)
        row_runs = row_runs[row_runs > 0]
        width_mean, _width_std, width_n = _robust_run_stats(row_runs, slant_correction)
        if width_n > 0:
            widths.append(width_mean)

        col_runs = _run_lengths_rowwise(band.T)
        col_runs = col_runs[col_runs > 0]
        height_mean, _height_std, height_n = _robust_run_stats(col_runs, slant_correction)
        if height_n > 0:
            heights.append(height_mean)

    return heights, widths


def _inverse_cv_score(values: list[float]) -> float:
    """Inverse-coefficient-of-variation regularity score for one measurement list.

    Computes ``1 / (1 + CV)`` where ``CV = std / mean`` is the coefficient
    of variation of ``values``: 1.0 when every value is identical (zero
    spread, perfectly regular), approaching 0.0 as the spread grows large
    relative to the mean (increasingly irregular). Using ``CV`` (a
    *relative* spread measure) rather than raw standard deviation matters
    because the three rhythm ingredients live on different scales (pixel
    heights, pixel gaps, pixel widths) -- CV makes them comparable before
    they're combined into one composite.

    Returns 0.0 if there are fewer than :data:`_RHYTHM_MIN_OBSERVATIONS`
    values, or if the mean is not positive (no meaningful spread to
    measure in either case).
    """
    if len(values) < _RHYTHM_MIN_OBSERVATIONS:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0:
        return 0.0
    cv = float(arr.std()) / mean
    return float(1.0 / (1.0 + cv))


def _build_rhythm_regularity(
    height_values: list[float], word_gap_values: list[float], width_values: list[float]
) -> float:
    """Composite rhythm_regularity score in [0, 1] (1.0 = highly regular).

    A weighted average of three :func:`_inverse_cv_score` sub-scores --
    per-line-band ink-run (letter) height, word-gap length, and
    per-line-band stroke width -- per ``docs/labeling_rubric.md``'s
    Rhythm definition ("how evenly stroke shapes, sizes, and spacing
    repeat across the sample"). Weights are :data:`_RHYTHM_WEIGHT_LINE_HEIGHT`,
    :data:`_RHYTHM_WEIGHT_WORD_GAP`, :data:`_RHYTHM_WEIGHT_STROKE_WIDTH`.

    An ingredient with too few observations to score
    (:data:`_RHYTHM_MIN_OBSERVATIONS`, checked via ``_inverse_cv_score``
    returning 0.0 in that case) is dropped entirely and the remaining
    weights are renormalized, rather than letting a missing ingredient
    silently count as "perfectly regular" (a 0.0 default would instead
    wrongly count as "perfectly irregular", also wrong). If no ingredient
    has enough observations at all, returns 0.0.
    """
    ingredients = (
        (height_values, _RHYTHM_WEIGHT_LINE_HEIGHT),
        (word_gap_values, _RHYTHM_WEIGHT_WORD_GAP),
        (width_values, _RHYTHM_WEIGHT_STROKE_WIDTH),
    )
    usable = [
        (_inverse_cv_score(values), weight)
        for values, weight in ingredients
        if len(values) >= _RHYTHM_MIN_OBSERVATIONS
    ]
    total_weight = sum(weight for _score, weight in usable)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(score * weight for score, weight in usable)
    return float(min(1.0, max(0.0, weighted_sum / total_weight)))


# --- stroke connectedness -----------------------------------------------------


def _stroke_connectedness_per_band(
    mask: np.ndarray, bands: list[tuple[int, int]]
) -> tuple[list[float], list[int]]:
    """Per-line-band (score, segment_count) for the stroke_connectedness composite.

    Within each line band, ink is projected onto columns and grouped into
    ink-column-run "segments" -- runs of columns that touch ink somewhere
    in the band, separated by columns with no ink at all. This is the same
    clustering :func:`_word_gaps` performs; it is recomputed here (rather
    than calling :func:`_word_gaps`) because that function only returns
    the pooled word-scale gap *values*, not the per-band segment counts
    and per-band word-gap counts this composite needs.

    The gaps between consecutive segments are classified into intra-word
    ("letter") vs inter-word ("word") gaps via one :func:`_otsu_threshold`
    split computed over *every* band's gaps pooled together -- mirroring
    :func:`_word_gaps`'s split (so the two features agree on what counts as
    a "word" gap), but computed once globally rather than per band, since
    a single band rarely has enough gaps on its own to support a reliable
    split. A band's estimated word count is then ``(number of word-level
    gaps in that band) + 1``; comparing that to the band's total segment
    count gives a segments-per-word ratio -- 1.0 means every word was
    drawn as one continuous run (fully connected), larger ratios mean
    words were broken into multiple lifted-pen pieces. The ratio is mapped
    to a [0, 1] score via :data:`_STROKE_CONNECTEDNESS_MAX_SEGMENTS_PER_WORD`.

    If there are too few gaps overall (:data:`_MIN_GAPS_FOR_OTSU_SPLIT`) to
    support a meaningful split, every gap is treated as word-level
    (mirroring :func:`_word_gaps`'s own fallback for the same case), which
    makes every segment its own "word" -- a deliberately neutral (maximally
    connected) default for when there isn't enough evidence to say a
    segment boundary represents a broken stroke rather than genuine word
    spacing.

    Returns two parallel lists, one entry per band that contained at least
    one segment: per-band connectedness scores in [0, 1], and per-band
    segment counts (used both as aggregation weights in
    :func:`_build_stroke_connectedness` and as the raw quantity
    :func:`_build_confidence` scales confidence on). Bands with no ink at
    all are skipped entirely (not scored as "disconnected").
    """
    band_clusters: list[np.ndarray] = []
    band_gaps: list[list[int]] = []
    all_gaps: list[int] = []

    for start, end in bands:
        band = mask[start:end, :]
        if band.shape[0] == 0:
            continue
        col_has_ink = band.any(axis=0)
        clusters = _runs_1d(col_has_ink)
        if clusters.shape[0] == 0:
            continue
        band_clusters.append(clusters)
        if clusters.shape[0] < 2:
            band_gaps.append([])
            continue
        gaps = [int(g) for g in (clusters[1:, 0] - clusters[:-1, 1]) if g > 0]
        band_gaps.append(gaps)
        all_gaps.extend(gaps)

    if not band_clusters:
        return [], []

    use_word_level_fallback = len(all_gaps) < _MIN_GAPS_FOR_OTSU_SPLIT
    threshold = (
        _otsu_threshold(np.array(all_gaps, dtype=np.float64)) if all_gaps else None
    )

    scores: list[float] = []
    weights: list[int] = []
    span = _STROKE_CONNECTEDNESS_MAX_SEGMENTS_PER_WORD - 1.0
    for clusters, gaps in zip(band_clusters, band_gaps):
        n_segments = int(clusters.shape[0])
        if use_word_level_fallback or threshold is None:
            n_words = n_segments
        else:
            n_word_gaps = sum(1 for g in gaps if g > threshold)
            n_words = n_word_gaps + 1
        if n_words <= 0:
            continue

        ratio = n_segments / n_words
        if span > 0:
            score = 1.0 - (ratio - 1.0) / span
        else:
            score = 1.0 if ratio <= 1.0 else 0.0
        scores.append(float(min(1.0, max(0.0, score))))
        weights.append(n_segments)

    return scores, weights


def _build_stroke_connectedness(scores: list[float], weights: list[int]) -> float:
    """Weighted-average stroke_connectedness composite in [0, 1].

    Combines :func:`_stroke_connectedness_per_band`'s per-band scores with
    a segment-count-weighted average, so a band whose estimate rests on
    many measured segments (more evidence) influences the overall score
    more than one built from only a couple of strokes. Returns 0.0 if no
    band produced a usable score.
    """
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return float(min(1.0, max(0.0, weighted_sum / total_weight)))


# --- overall organization ------------------------------------------------------


def _band_left_edges(mask: np.ndarray, bands: list[tuple[int, int]]) -> list[float]:
    """Leftmost ink-containing column (pixels) within each detected line band.

    One value per band that contains any ink at all -- a proxy for where
    each line "starts" on the page, used by
    :func:`_build_organization_score` to measure how consistently lines
    are left-aligned to one another (the labeling rubric's "alignment of
    lines" language). A band with no ink at all (should not normally
    occur, since bands are themselves detected from an ink-density
    profile) is skipped rather than padded with a fabricated value.
    """
    edges: list[float] = []
    for start, end in bands:
        band = mask[start:end, :]
        if band.shape[0] == 0:
            continue
        col_has_ink = band.any(axis=0)
        cols = np.nonzero(col_has_ink)[0]
        if cols.size == 0:
            continue
        edges.append(float(cols.min()))
    return edges


def _threshold_consistency_score(spread: float, max_spread: float) -> float:
    """Linear-decay consistency score: 1.0 at ``spread == 0``, 0.0 at/beyond ``max_spread``.

    Used by the line-start-alignment and baseline-slope-consistency
    ingredients of :func:`_build_organization_score`, whose underlying
    quantities (a column position in pixels, a slope in degrees) don't
    share a natural positive "typical scale" the way the rhythm
    ingredients' coefficient-of-variation composite
    (:func:`_inverse_cv_score`) relies on -- a column position can
    legitimately be near zero, and a slope can be zero or negative, either
    of which would make a CV-style ``std / mean`` ratio undefined or
    misleading. A simple spread-vs-threshold linear decay avoids that,
    mirroring the shape of decay :func:`_stroke_connectedness_per_band`
    already uses for its segments-per-word ratio.

    Returns 0.0 if ``max_spread`` is not positive (nothing to normalize
    against).
    """
    if max_spread <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - spread / max_spread)))


def _build_organization_score(
    band_gap_values: list[float],
    left_edge_values: list[float],
    baseline_slope_degree_values: list[float],
    content_width_px: float,
) -> float:
    """Composite organization_score in [0, 1] (1.0 = highly organized layout).

    A weighted average of three consistency sub-scores, per
    ``docs/labeling_rubric.md``'s Overall Organization definition
    ("alignment of lines, use of space... apparent planning of layout"):

    - Inter-band vertical spacing consistency: :func:`_inverse_cv_score`
      over the list of gaps between consecutive detected line bands (the
      same technique :func:`_build_rhythm_regularity` uses for its
      ingredients -- a gap length is a naturally positive quantity with a
      meaningful "typical scale", its own mean).
    - Left-edge (line-start) alignment consistency: how tightly clustered
      each band's leftmost ink column is, scored by
      :func:`_threshold_consistency_score` against
      :data:`_ORG_LEFT_EDGE_MAX_STD_FRACTION` of the content width.
    - Baseline-slope consistency: how tightly clustered each band's
      baseline slope (in degrees, reusing
      :func:`_baseline_slopes_per_band`'s per-band fits) is, scored by
      :func:`_threshold_consistency_score` against
      :data:`_ORG_BASELINE_SLOPE_STD_MAX_DEGREES`.

    Weights are :data:`_ORG_WEIGHT_BAND_SPACING`,
    :data:`_ORG_WEIGHT_LEFT_EDGE`, :data:`_ORG_WEIGHT_BASELINE_SLOPE`
    (equal by default -- see their docstring for the reasoning). An
    ingredient with fewer than :data:`_ORG_MIN_OBSERVATIONS` observations
    is dropped from the composite and the remaining weights renormalized,
    the same pattern :func:`_build_rhythm_regularity` uses. Returns 0.0 if
    no ingredient has enough observations at all (e.g. 0 or 1 detected
    line bands).
    """
    scores_and_weights: list[tuple[float, float]] = []

    if len(band_gap_values) >= _ORG_MIN_OBSERVATIONS:
        scores_and_weights.append(
            (_inverse_cv_score(band_gap_values), _ORG_WEIGHT_BAND_SPACING)
        )

    if len(left_edge_values) >= _ORG_MIN_OBSERVATIONS and content_width_px > 0:
        left_edge_std = float(np.asarray(left_edge_values, dtype=np.float64).std())
        max_std = _ORG_LEFT_EDGE_MAX_STD_FRACTION * content_width_px
        scores_and_weights.append(
            (_threshold_consistency_score(left_edge_std, max_std), _ORG_WEIGHT_LEFT_EDGE)
        )

    if len(baseline_slope_degree_values) >= _ORG_MIN_OBSERVATIONS:
        slope_std = float(np.asarray(baseline_slope_degree_values, dtype=np.float64).std())
        scores_and_weights.append(
            (
                _threshold_consistency_score(slope_std, _ORG_BASELINE_SLOPE_STD_MAX_DEGREES),
                _ORG_WEIGHT_BASELINE_SLOPE,
            )
        )

    total_weight = sum(weight for _score, weight in scores_and_weights)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(score * weight for score, weight in scores_and_weights)
    return float(min(1.0, max(0.0, weighted_sum / total_weight)))


# --- confidence ---------------------------------------------------------------


def _confidence_scale(n: int, full_at: int) -> float:
    """Linearly ramp confidence from 0 to 1.0 as ``n`` approaches ``full_at``."""
    if full_at <= 0:
        return 0.0
    return float(min(1.0, max(0.0, n / full_at)))


def _build_confidence(
    *,
    slant_n: int,
    width_n: int,
    height_n: int,
    line_gap_n: int,
    word_gap_n: int,
    baseline_n: int,
    rhythm_height_band_n: int,
    rhythm_width_band_n: int,
    stroke_connectedness_segment_n: int,
    organization_band_n: int,
    has_ink: bool,
) -> dict[str, float]:
    """Build the per-field confidence heuristic described on :class:`Features`.

    Two kinds of indicators get two kinds of confidence:

    - Statistically-estimated indicators (slant, stroke width, letter
      size, line/word spacing, baseline slope) scale with how many
      underlying observations (ink pixels, runs, gaps, fitted columns)
      the estimate was built from, via :func:`_confidence_scale` --
      few observations (e.g. a near-empty image with barely any detected
      strokes or lines) means low confidence.
    - Directly-measured geometric indicators (margins, ink density) are
      near-exact once *any* ink is found (they're a bounding box and a
      pixel-count ratio, not a statistical estimate), so they get a flat
      high/low split on whether any ink exists at all.

    ``rhythm_regularity`` is a third, composite case: it is the unweighted
    mean of three per-ingredient confidences (line-band letter height,
    word-gap length reusing ``word_gap_n``, line-band stroke width),
    deliberately *not* renormalized over only the usable ingredients the
    way :func:`_build_rhythm_regularity`'s value itself is. That
    asymmetry is intentional: a rhythm score built from only one usable
    ingredient (e.g. plenty of word gaps but too few detected line bands)
    should read as *less* trustworthy than one built from all three, even
    though the value's own renormalization means it doesn't look
    "wrong" -- confidence is where that missing evidence should show up.

    ``stroke_connectedness`` follows the same statistically-estimated
    pattern as stroke width/letter size: it scales with
    ``stroke_connectedness_segment_n``, the total count of ink-column-run
    segments that contributed a usable per-band score in
    :func:`_stroke_connectedness_per_band` -- few segments (few detected
    line bands / sparse ink) means the segments-per-word ratio the score
    is built from is not well supported.

    ``organization_score`` is a fourth, deliberately different case: its
    confidence does *not* scale with any per-pixel/per-run observation
    count the way every other statistically-estimated field above does.
    Instead it scales directly with ``organization_band_n`` (the number of
    *detected line bands*), via :func:`_confidence_scale` against
    :data:`_CONF_FULL_AT_ORGANIZATION_BANDS`. This is intentional:
    organization is inherently a cross-line comparison (spacing between
    lines, alignment of one line's start against another's, slope
    consistency line to line), so no amount of ink/runs/columns *within* a
    single line can make an organization read off that one line
    trustworthy -- the threshold is chosen so a single detected band
    always lands strictly below 0.5 confidence.
    """
    geometric_confidence = 1.0 if has_ink else 0.0
    rhythm_confidence = (
        _confidence_scale(rhythm_height_band_n, _CONF_FULL_AT_RHYTHM_BANDS)
        + _confidence_scale(word_gap_n, _CONF_FULL_AT_WORD_GAPS)
        + _confidence_scale(rhythm_width_band_n, _CONF_FULL_AT_RHYTHM_BANDS)
    ) / 3.0
    confidence = {
        "slant_angle_degrees": _confidence_scale(slant_n, _CONF_FULL_AT_SLANT_PIXELS),
        "stroke_width_mean": _confidence_scale(width_n, _CONF_FULL_AT_RUNS),
        "stroke_width_std": _confidence_scale(width_n, _CONF_FULL_AT_RUNS),
        "letter_size_estimate": _confidence_scale(height_n, _CONF_FULL_AT_RUNS),
        "line_spacing_mean": _confidence_scale(line_gap_n, _CONF_FULL_AT_LINE_GAPS),
        "word_spacing_mean": _confidence_scale(word_gap_n, _CONF_FULL_AT_WORD_GAPS),
        "baseline_slope_degrees": _confidence_scale(baseline_n, _CONF_FULL_AT_BASELINE_COLUMNS),
        "margin_left_px": geometric_confidence,
        "margin_right_px": geometric_confidence,
        "margin_top_px": geometric_confidence,
        "margin_bottom_px": geometric_confidence,
        "ink_density": geometric_confidence,
        "rhythm_regularity": rhythm_confidence,
        "stroke_connectedness": _confidence_scale(
            stroke_connectedness_segment_n, _CONF_FULL_AT_STROKE_SEGMENTS
        ),
        "organization_score": _confidence_scale(
            organization_band_n, _CONF_FULL_AT_ORGANIZATION_BANDS
        ),
    }
    # Belt-and-suspenders clamp: every value must land in [0, 1].
    return {key: min(1.0, max(0.0, value)) for key, value in confidence.items()}


def _empty_features(width: int, height: int) -> Features:
    """Return a sane, NaN-free :class:`Features` for a blank/near-empty image.

    All numeric measurements default to 0.0 (there is no content to
    measure) and every confidence score is 0.0 (there is nothing to be
    confident about), rather than raising or returning NaN/infinite
    values. ``width``/``height`` are accepted for symmetry with the
    non-empty path but are not otherwise needed, since 0.0 is used
    uniformly rather than e.g. reporting margins equal to the full image
    extent.
    """
    del width, height  # unused; kept for call-site symmetry/documentation
    zero_confidence = {key: 0.0 for key in FEATURE_CONFIDENCE_KEYS}
    return Features(
        slant_angle_degrees=0.0,
        stroke_width_mean=0.0,
        stroke_width_std=0.0,
        letter_size_estimate=0.0,
        line_spacing_mean=0.0,
        word_spacing_mean=0.0,
        baseline_slope_degrees=0.0,
        margin_left_px=0.0,
        margin_right_px=0.0,
        margin_top_px=0.0,
        margin_bottom_px=0.0,
        ink_density=0.0,
        rhythm_regularity=0.0,
        stroke_connectedness=0.0,
        organization_score=0.0,
        confidence=zero_confidence,
    )


def analyze(image: ImageInput) -> Features:
    """Extract heuristic graphological :class:`Features` from a handwriting image.

    Accepts a :class:`PIL.Image.Image` or a path to an image file (``str``
    or ``os.PathLike``), matching the interface used across this project
    (see :mod:`grafology_ai.validation.validators`).

    This is a classical-image-processing heuristic, not a trained model:
    it makes simplifying assumptions (a single dominant slant per sample,
    descender-insensitive baselines, run-length proxies for stroke
    width/letter size rather than full connected-component labeling --
    see the module docstring) and its measurements should be read as
    approximate, directionally-meaningful signals rather than precise
    ground truth. Every returned value is finite (no NaN/inf); a blank or
    near-empty image (no detectable ink) returns zeroed measurements with
    zero confidence rather than raising.
    """
    pil_image = _load_image(image)
    grayscale = _to_grayscale_array(pil_image)
    height, width = grayscale.shape

    ink_mask = _binarize(grayscale)
    ys, xs = np.nonzero(ink_mask)
    n_ink = int(xs.size)

    if n_ink == 0:
        return _empty_features(width, height)

    # Margins + ink density, from the ink content's bounding box.
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    margin_left = float(x_min)
    margin_right = float(width - 1 - x_max)
    margin_top = float(y_min)
    margin_bottom = float(height - 1 - y_max)

    bbox_mask = ink_mask[y_min : y_max + 1, x_min : x_max + 1]
    bbox_area = bbox_mask.shape[0] * bbox_mask.shape[1]
    ink_density = float(bbox_mask.sum()) / bbox_area if bbox_area > 0 else 0.0

    # Line bands drive line spacing, baseline slope, word spacing, and (see
    # below) slant -- detected first since slant is estimated per band.
    bands = _detect_line_bands(ink_mask)

    # Slant (needed before stroke-width/letter-size: their run lengths are
    # corrected by the estimated slant, see _robust_run_stats).
    slant_angle_degrees, slant_n = _estimate_slant_degrees(ys, xs, bands)
    slant_correction = math.cos(math.radians(slant_angle_degrees))

    # Stroke width (pressure proxy): row-wise horizontal run lengths.
    row_runs = _run_lengths_rowwise(ink_mask)
    row_runs = row_runs[row_runs > 0]
    stroke_width_mean, stroke_width_std, width_n = _robust_run_stats(
        row_runs, slant_correction
    )

    # Letter size proxy: column-wise vertical run lengths.
    col_runs = _run_lengths_rowwise(ink_mask.T)
    col_runs = col_runs[col_runs > 0]
    letter_size_estimate, _letter_size_std, height_n = _robust_run_stats(
        col_runs, slant_correction
    )

    line_spacing_mean, line_gap_n = _mean_band_gap(bands)
    baseline_slope_degrees, baseline_n = _estimate_baseline_slope(ink_mask, bands)

    word_spacing_mean, word_gap_n = _estimate_word_spacing(ink_mask, bands)

    # The raw word-gap list (not just its mean) is also one of the three
    # rhythm_regularity ingredients below -- its *spread* matters there.
    word_gaps = _word_gaps(ink_mask, bands)

    # Rhythm regularity: per-line-band letter height / stroke width plus the
    # word-gap list collected above, combined via an inverse-CV composite
    # (see _build_rhythm_regularity).
    rhythm_heights, rhythm_widths = _per_band_run_stats(ink_mask, bands, slant_correction)
    rhythm_regularity = _build_rhythm_regularity(rhythm_heights, word_gaps, rhythm_widths)

    # Stroke connectedness: per-line-band segments-per-word ratio, combined
    # via a segment-count-weighted average (see _build_stroke_connectedness).
    connectedness_scores, connectedness_weights = _stroke_connectedness_per_band(
        ink_mask, bands
    )
    stroke_connectedness = _build_stroke_connectedness(
        connectedness_scores, connectedness_weights
    )

    # Overall organization: inter-band spacing / left-edge / baseline-slope
    # consistency across line bands (see _build_organization_score).
    # _baseline_slopes_per_band is called again here (rather than reusing
    # the per-band fits _estimate_baseline_slope computed internally above)
    # to keep _estimate_baseline_slope a self-contained, independently
    # correct thin wrapper -- the extra per-band linear fits are cheap.
    band_gaps = _band_gaps(bands)
    band_left_edges = _band_left_edges(ink_mask, bands)
    baseline_slopes_px, _baseline_slope_weights = _baseline_slopes_per_band(ink_mask, bands)
    baseline_slope_degrees_per_band = [
        -math.degrees(math.atan(slope)) for slope in baseline_slopes_px
    ]
    content_width_px = float(x_max - x_min)
    organization_score = _build_organization_score(
        band_gaps, band_left_edges, baseline_slope_degrees_per_band, content_width_px
    )

    confidence = _build_confidence(
        slant_n=slant_n,
        width_n=width_n,
        height_n=height_n,
        line_gap_n=line_gap_n,
        word_gap_n=word_gap_n,
        baseline_n=baseline_n,
        rhythm_height_band_n=len(rhythm_heights),
        rhythm_width_band_n=len(rhythm_widths),
        stroke_connectedness_segment_n=int(sum(connectedness_weights)),
        organization_band_n=len(bands),
        has_ink=True,
    )

    return Features(
        slant_angle_degrees=slant_angle_degrees,
        stroke_width_mean=stroke_width_mean,
        stroke_width_std=stroke_width_std,
        letter_size_estimate=letter_size_estimate,
        line_spacing_mean=line_spacing_mean,
        word_spacing_mean=word_spacing_mean,
        baseline_slope_degrees=baseline_slope_degrees,
        margin_left_px=margin_left,
        margin_right_px=margin_right,
        margin_top_px=margin_top,
        margin_bottom_px=margin_bottom,
        ink_density=ink_density,
        rhythm_regularity=rhythm_regularity,
        stroke_connectedness=stroke_connectedness,
        organization_score=organization_score,
        confidence=confidence,
    )
