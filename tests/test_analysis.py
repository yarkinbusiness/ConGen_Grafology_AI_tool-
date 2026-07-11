"""Tests for grafology_ai.analysis.features.

Every fixture image here is generated programmatically with Pillow, with
a *known, controlled ground truth* for exactly one property at a time --
these are fresh, purpose-built images, not the general-purpose
"handwriting-like" fixture from :mod:`grafology_ai.dataset.fixtures` or
``tests/test_validation.py`` (those vary several properties at once and
don't expose ground truth, so they can't verify directional correctness
of a single measurement).

- ``_draw_stroke_grid`` draws a deterministic grid of short strokes
  ("letters") arranged in lines, following the same drawing convention as
  :func:`grafology_ai.dataset.fixtures._draw_handwriting_like_strokes`:
  each stroke runs from ``(x, y)`` (bottom) to ``(x + dx, y -
  stroke_height)`` (top) with ``dx = stroke_height * tan(slant_deg)``, so
  ``slant_deg > 0`` leans right and ``slant_deg < 0`` leans left --
  matching :attr:`~grafology_ai.analysis.Features.slant_angle_degrees`'s
  documented sign convention. Varying its keyword arguments in isolation
  gives known-ground-truth control over slant, stroke width, letter
  height, and line spacing.
- ``_draw_word_grid`` additionally groups strokes into "words" separated
  by a wider gap than the intra-word letter gap, for word-spacing tests.
- ``_draw_pushed_content`` confines strokes to a sub-region of the
  canvas, for margin tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from grafology_ai.analysis import FEATURE_CONFIDENCE_KEYS, Features, analyze
from grafology_ai.dataset.fixtures import generate_fixture_dataset

BACKGROUND = 255
INK = 0


def _draw_stroke_grid(
    size: tuple[int, int],
    *,
    slant_deg: float = 0.0,
    stroke_width: int = 4,
    stroke_height: int = 16,
    stroke_spacing: int = 14,
    line_spacing: int = 40,
    margin: int = 30,
) -> Image.Image:
    """A deterministic grid of short strokes arranged in lines.

    See the module docstring for the sign convention on ``slant_deg``.
    """
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


def _draw_word_grid(
    size: tuple[int, int],
    *,
    word_gap: int,
    letter_gap: int = 6,
    stroke_width: int = 4,
    stroke_height: int = 16,
    letters_per_word: int = 4,
    line_spacing: int = 40,
    margin: int = 30,
) -> Image.Image:
    """A grid of "words" (letter clusters) separated by ``word_gap`` pixels.

    Within a word, consecutive letters are ``letter_gap`` pixels apart
    (small and fixed); after a word, an additional ``word_gap`` pixels are
    inserted before the next word starts -- two distinct gap scales, the
    way real handwriting has tighter intra-word spacing than inter-word
    spacing.
    """
    image = Image.new("L", size, color=BACKGROUND)
    draw = ImageDraw.Draw(image)

    y = margin + stroke_height
    while y <= size[1] - margin:
        x = margin
        while x <= size[0] - margin:
            for _ in range(letters_per_word):
                if x > size[0] - margin:
                    break
                draw.line([(x, y), (x, y - stroke_height)], fill=INK, width=stroke_width)
                x += letter_gap
            x += word_gap
        y += line_spacing
    return image.convert("RGB")


def _draw_pushed_content(
    size: tuple[int, int],
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    stroke_width: int = 4,
    stroke_height: int = 14,
    stroke_spacing: int = 14,
    line_spacing: int = 30,
) -> Image.Image:
    """Draw a stroke grid confined to ``x_range``/``y_range`` sub-region.

    Used to give the rest of the canvas a large, known-empty margin on
    whichever side(s) ``x_range``/``y_range`` don't reach.
    """
    image = Image.new("L", size, color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    x0, x1 = x_range
    y0, y1 = y_range

    y = y0 + stroke_height
    while y <= y1:
        x = x0
        while x <= x1:
            draw.line([(x, y), (x, y - stroke_height)], fill=INK, width=stroke_width)
            x += stroke_spacing
        y += line_spacing
    return image.convert("RGB")


def _blank_image(size: tuple[int, int] = (300, 300)) -> Image.Image:
    return Image.new("RGB", size, color=(255, 255, 255))


def _draw_rhythm_grid(
    size: tuple[int, int],
    *,
    line_specs: list[tuple[int, int, int]],
    words_per_line: int = 6,
    letters_per_word: int = 4,
    letter_gap: int = 6,
    line_pitch: int = 50,
    margin: int = 30,
) -> Image.Image:
    """A grid of "words" with independently controllable per-line rhythm ingredients.

    ``line_specs`` is one ``(stroke_width, stroke_height, word_gap)`` triple
    per drawn line -- i.e. direct, known-ground-truth control over the
    three quantities the ``rhythm_regularity`` composite measures
    (per-line stroke width, per-line letter/ink-run height, and the
    word-gap length used between every word on that line). Every line
    draws exactly ``words_per_line * letters_per_word`` strokes regardless
    of ``line_specs`` (word/letter *counts* are fixed, only their
    thickness/height/spacing vary), so passing triples whose
    ``stroke_width * stroke_height`` product is held constant across all
    lines keeps total ink pixel area matched across differently-*shaped*
    images -- letting a test isolate regularity (the spread of these
    quantities across lines) from density (how much ink there is overall).
    """
    image = Image.new("L", size, color=BACKGROUND)
    draw = ImageDraw.Draw(image)

    for i, (stroke_width, stroke_height, word_gap) in enumerate(line_specs):
        y = margin + i * line_pitch + stroke_height
        x = margin
        for _word in range(words_per_line):
            for _letter in range(letters_per_word):
                draw.line([(x, y), (x, y - stroke_height)], fill=INK, width=stroke_width)
                x += letter_gap
            x += word_gap
    return image.convert("RGB")


# --- slant --------------------------------------------------------------


def test_slant_right_lean_measures_positive_angle() -> None:
    image = _draw_stroke_grid((560, 400), slant_deg=20.0)
    features = analyze(image)
    assert features.slant_angle_degrees > 0
    assert abs(features.slant_angle_degrees - 20.0) <= 10.0


def test_slant_left_lean_measures_negative_angle() -> None:
    image = _draw_stroke_grid((560, 400), slant_deg=-20.0)
    features = analyze(image)
    assert features.slant_angle_degrees < 0
    assert abs(features.slant_angle_degrees - (-20.0)) <= 10.0


# --- stroke width / pressure --------------------------------------------


def test_thick_strokes_measure_higher_stroke_width_than_thin() -> None:
    thin = _draw_stroke_grid((560, 400), stroke_width=2)
    thick = _draw_stroke_grid((560, 400), stroke_width=10)

    thin_features = analyze(thin)
    thick_features = analyze(thick)

    assert thick_features.stroke_width_mean > thin_features.stroke_width_mean + 3.0


# --- line spacing ---------------------------------------------------------


def test_wide_line_spacing_measures_higher_than_tight() -> None:
    tight = _draw_stroke_grid((560, 400), stroke_height=14, line_spacing=30)
    wide = _draw_stroke_grid((560, 900), stroke_height=14, line_spacing=100)

    tight_features = analyze(tight)
    wide_features = analyze(wide)

    assert tight_features.line_spacing_mean > 0
    assert wide_features.line_spacing_mean > tight_features.line_spacing_mean * 2


# --- word spacing -----------------------------------------------------------


def test_wide_word_spacing_measures_higher_than_tight() -> None:
    tight = _draw_word_grid((600, 300), word_gap=10)
    wide = _draw_word_grid((600, 300), word_gap=70)

    tight_features = analyze(tight)
    wide_features = analyze(wide)

    assert wide_features.word_spacing_mean > tight_features.word_spacing_mean * 2


# --- margins ------------------------------------------------------------


def test_margins_reflect_horizontal_asymmetry() -> None:
    # Content confined to the right side of the canvas: large left margin,
    # small right margin.
    image = _draw_pushed_content(
        (600, 300), x_range=(420, 570), y_range=(30, 270)
    )
    features = analyze(image)

    assert features.margin_left_px > features.margin_right_px
    assert features.margin_left_px - features.margin_right_px > 200


def test_margins_reflect_vertical_asymmetry() -> None:
    # Content confined to the top of the canvas: small top margin, large
    # bottom margin.
    image = _draw_pushed_content(
        (400, 600), x_range=(30, 370), y_range=(20, 120)
    )
    features = analyze(image)

    assert features.margin_bottom_px > features.margin_top_px
    assert features.margin_bottom_px - features.margin_top_px > 200


# --- ink density -------------------------------------------------------------


def test_dense_ink_measures_higher_density_than_sparse() -> None:
    sparse = _draw_stroke_grid((500, 400), stroke_width=2, stroke_spacing=40, stroke_height=10)
    dense = _draw_stroke_grid((500, 400), stroke_width=8, stroke_spacing=10, stroke_height=16)

    sparse_features = analyze(sparse)
    dense_features = analyze(dense)

    assert 0.0 <= sparse_features.ink_density <= 1.0
    assert 0.0 <= dense_features.ink_density <= 1.0
    assert dense_features.ink_density > sparse_features.ink_density * 2


# --- confidence -----------------------------------------------------------


def test_confidence_has_entry_for_every_covered_feature_field() -> None:
    image = _draw_stroke_grid((500, 400))
    features = analyze(image)

    assert set(features.confidence.keys()) == set(FEATURE_CONFIDENCE_KEYS)
    assert "confidence" not in features.confidence
    for value in features.confidence.values():
        assert 0.0 <= value <= 1.0


def test_confidence_is_low_when_few_strokes_detected() -> None:
    sparse_image = Image.new("L", (200, 200), color=BACKGROUND)
    draw = ImageDraw.Draw(sparse_image)
    draw.line([(100, 100), (102, 90)], fill=INK, width=1)

    features = analyze(sparse_image.convert("RGB"))

    assert features.confidence["slant_angle_degrees"] < 0.2


# --- blank image handling ---------------------------------------------------


def test_analyze_blank_image_does_not_crash() -> None:
    features = analyze(_blank_image())
    assert isinstance(features, Features)


def test_analyze_blank_image_returns_finite_defaults() -> None:
    features = analyze(_blank_image())

    numeric_fields = (
        features.slant_angle_degrees,
        features.stroke_width_mean,
        features.stroke_width_std,
        features.letter_size_estimate,
        features.line_spacing_mean,
        features.word_spacing_mean,
        features.baseline_slope_degrees,
        features.margin_left_px,
        features.margin_right_px,
        features.margin_top_px,
        features.margin_bottom_px,
        features.ink_density,
    )
    for value in numeric_fields:
        assert math.isfinite(value)


def test_analyze_blank_image_has_low_confidence_everywhere() -> None:
    features = analyze(_blank_image())
    for value in features.confidence.values():
        assert 0.0 <= value <= 0.2


# --- general API / structure ------------------------------------------------


def test_features_is_frozen() -> None:
    features = analyze(_draw_stroke_grid((300, 200)))
    with pytest.raises(Exception):
        features.slant_angle_degrees = 5.0  # type: ignore[misc]


def test_analyze_accepts_path_and_image_identically(tmp_path: Path) -> None:
    image = _draw_stroke_grid((300, 200))
    path = tmp_path / "sample.png"
    image.save(path, format="PNG")

    from_image = analyze(image)
    from_path = analyze(path)
    from_str_path = analyze(str(path))

    assert from_image.slant_angle_degrees == pytest.approx(
        from_path.slant_angle_degrees, abs=1e-6
    )
    assert from_path.slant_angle_degrees == pytest.approx(
        from_str_path.slant_angle_degrees, abs=1e-6
    )


def test_letter_size_estimate_positive_for_normal_image_and_zero_for_blank() -> None:
    normal_features = analyze(_draw_stroke_grid((400, 300)))
    blank_features = analyze(_blank_image())

    assert normal_features.letter_size_estimate > 0
    assert blank_features.letter_size_estimate == 0.0


# --- rhythm regularity ---------------------------------------------------


def test_rhythm_regularity_field_and_confidence_key_exist() -> None:
    assert "rhythm_regularity" in Features.__dataclass_fields__
    assert "rhythm_regularity" in FEATURE_CONFIDENCE_KEYS

    features = analyze(_draw_stroke_grid((400, 300)))
    assert "rhythm_regularity" in features.confidence


def test_uniform_rhythm_scores_higher_than_varied_with_matched_ink_amount() -> None:
    # Every line uses the same (stroke_width, stroke_height, word_gap)
    # triple: zero spread in any of the three rhythm ingredients.
    uniform_specs = [(6, 16, 30)] * 6

    # Alternating lines use two different triples whose stroke_width *
    # stroke_height product is held at 96 (matching the uniform triple's
    # 6 * 16 = 96), so every line draws the same total ink area -- only
    # the *shape* (thin-and-tall vs thick-and-short letters) and the
    # word-gap length vary from line to line. This isolates rhythm
    # (regularity) from ink density (amount).
    varied_specs = [(4, 24, 15) if i % 2 == 0 else (8, 12, 45) for i in range(6)]

    uniform_image = _draw_rhythm_grid((700, 500), line_specs=uniform_specs)
    varied_image = _draw_rhythm_grid((700, 500), line_specs=varied_specs)

    # Confirm the "matched ink amount" premise: total ink pixel counts
    # should be close (not necessarily identical, since discrete pixel
    # rounding and rendering of different stroke shapes isn't perfectly
    # exact), well within a loose tolerance.
    uniform_ink = (np.asarray(uniform_image.convert("L")) != BACKGROUND).sum()
    varied_ink = (np.asarray(varied_image.convert("L")) != BACKGROUND).sum()
    assert varied_ink == pytest.approx(uniform_ink, rel=0.1)

    uniform_features = analyze(uniform_image)
    varied_features = analyze(varied_image)

    assert uniform_features.rhythm_regularity > varied_features.rhythm_regularity


def test_rhythm_regularity_within_unit_interval_and_finite_across_fixture_dataset(
    tmp_path: Path,
) -> None:
    entries = generate_fixture_dataset(tmp_path / "raw", count=12, seed=1)
    images_dir = tmp_path / "raw" / "images"

    for entry in entries:
        features = analyze(images_dir / f"{entry.sample_id}.png")
        assert math.isfinite(features.rhythm_regularity)
        assert 0.0 <= features.rhythm_regularity <= 1.0
        assert math.isfinite(features.confidence["rhythm_regularity"])
        assert 0.0 <= features.confidence["rhythm_regularity"] <= 1.0


def test_analyze_blank_image_rhythm_regularity_and_confidence_are_zero() -> None:
    features = analyze(_blank_image())
    assert features.rhythm_regularity == 0.0
    assert features.confidence["rhythm_regularity"] == 0.0


def test_rhythm_regularity_is_deterministic_across_repeated_analyze_calls() -> None:
    image = _draw_rhythm_grid((700, 500), line_specs=[(6, 16, 30)] * 6)

    first = analyze(image)
    second = analyze(image)

    assert first.rhythm_regularity == second.rhythm_regularity
    assert first.confidence["rhythm_regularity"] == second.confidence["rhythm_regularity"]


def test_rhythm_regularity_confidence_within_unit_interval() -> None:
    rich_image = _draw_rhythm_grid((700, 500), line_specs=[(6, 16, 30)] * 6)
    sparse_image = Image.new("L", (200, 200), color=BACKGROUND)
    draw = ImageDraw.Draw(sparse_image)
    draw.line([(100, 100), (102, 90)], fill=INK, width=1)

    for image in (rich_image, sparse_image.convert("RGB"), _blank_image()):
        features = analyze(image)
        assert 0.0 <= features.confidence["rhythm_regularity"] <= 1.0
