"""Synthetic fixture dataset generator for handwriting-sample images.

Real client handwriting-sample data does not exist yet, so downstream
pipeline stages (dataset ingest/split/export, the feature analyzer) need
something to run against in the meantime: real image files on disk plus a
real manifest, without any dependency on the client's eventual dataset.

:func:`generate_fixture_dataset` renders ``count`` procedurally generated,
handwriting-like PNG images with Pillow -- reusing the same kind of
line-drawing approach used for the test fixtures in
``tests/test_validation.py``, but varying simulated slant, pressure
(stroke width), letter/word spacing, overall image size, and image-quality
tier across the generated set -- and writes a matching
``manifest.json`` of :class:`~grafology_ai.dataset.schema.ManifestEntry`
records.

The generated samples are unlabeled (``labels={}``): this is synthetic
fixture data, not real ground truth, so fabricating fake graphological
labels for it would be misleading. Everything is deterministic given the
same ``seed``: a single :class:`numpy.random.Generator` seeded from
``seed`` (never numpy's global random state) drives every random choice, in
a fixed order that depends only on ``count``, so two calls with the same
``count`` and ``seed`` produce byte-identical images and manifest content.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from grafology_ai.dataset.schema import ManifestEntry

#: Target proportions for the three quality tiers, matching a realistic
#: intake distribution: most samples are clean, a meaningful minority are
#: merely acceptable, and only a small tail is genuinely poor quality.
QUALITY_PROPORTIONS: dict[str, float] = {"high": 0.70, "medium": 0.20, "low": 0.10}

#: Grayscale background/ink levels and Gaussian-blur radius used to render
#: each quality tier. These are tuned to land clearly on either side of the
#: check_blur / check_contrast thresholds in
#: :mod:`grafology_ai.validation.validators` (``DEFAULT_MIN_BLUR_VARIANCE``
#: = 150, ``DEFAULT_MIN_CONTRAST_STD_DEV`` = 25), so that "low" fixtures
#: genuinely fail those automated checks and "high" fixtures genuinely pass
#: them -- the quality label reflects a real difference in the rendered
#: pixels, not just a name.
_QUALITY_RENDER_PARAMS: dict[str, dict[str, float]] = {
    "high": {"background": 245, "ink": 5, "blur_radius": 0.0},
    "medium": {"background": 232, "ink": 45, "blur_radius": 1.3},
    "low": {"background": 132, "ink": 118, "blur_radius": 8.0},
}

#: Pixel-dimension ranges (inclusive low, exclusive high) that generated
#: images are drawn from, simulating varying overall sample size. The
#: shorter side always exceeds check_resolution's default ~450px minimum,
#: so quality-tier differences are driven by blur/contrast, not resolution.
_WIDTH_RANGE = (700, 1200)
_HEIGHT_RANGE = (550, 950)

#: Simulated slant angle range in degrees from vertical (negative = left
#: lean, positive = right lean), and stroke-width ("pressure") range in
#: pixels.
_SLANT_DEGREES_RANGE = (-25.0, 25.0)
_STROKE_WIDTH_RANGE = (2, 9)

#: Multiplier applied to the base letter/word/line spacing, simulating
#: tighter or looser handwriting spacing across the generated set.
_SPACING_SCALE_RANGE = (0.7, 1.6)


def _quality_tier_counts(count: int) -> dict[str, int]:
    """Split ``count`` samples across quality tiers per QUALITY_PROPORTIONS.

    The "high" and "medium" tier sizes are rounded from their target
    proportions; "low" takes the remainder, so the three counts always sum
    to exactly ``count``.
    """
    n_high = round(count * QUALITY_PROPORTIONS["high"])
    n_medium = round(count * QUALITY_PROPORTIONS["medium"])
    n_low = count - n_high - n_medium
    return {"high": n_high, "medium": n_medium, "low": n_low}


def _assign_quality_tiers(count: int, rng: np.random.Generator) -> list[str]:
    """Return a length-``count`` list mapping sample index -> quality tier.

    Tier counts are fixed by :func:`_quality_tier_counts` (so the target
    proportions are met exactly rather than merely in expectation); which
    index gets which tier is decided by a single deterministic permutation
    draw from ``rng``, so tiers are interleaved rather than grouped.
    """
    counts = _quality_tier_counts(count)
    tiers = (
        ["high"] * counts["high"] + ["medium"] * counts["medium"] + ["low"] * counts["low"]
    )
    order = rng.permutation(count)
    assigned = [""] * count
    for position, index in enumerate(order):
        assigned[index] = tiers[position]
    return assigned


def _draw_handwriting_like_strokes(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    slant_deg: float,
    stroke_width: int,
    spacing_scale: float,
    ink: int,
) -> None:
    """Draw procedurally generated, handwriting-like strokes onto ``draw``.

    Individual "letter" strokes are short line segments leaning
    ``slant_deg`` from vertical (simulating handwriting slant), drawn at
    ``stroke_width`` (simulating pen pressure). Strokes are grouped into
    "words" separated by wider gaps and arranged along baselines
    ("lines"), with every horizontal gap scaled by ``spacing_scale``
    (simulating letter/word spacing).
    """
    width, height = size
    margin = max(15, int(min(width, height) * 0.06))
    slant_rad = math.radians(slant_deg)
    line_height = max(22, int(30 * spacing_scale))

    y = margin + line_height
    while y < height - margin:
        x = margin
        while x < width - margin:
            word_length = int(rng.integers(3, 8))
            for _ in range(word_length):
                if x > width - margin:
                    break
                stroke_height = int(rng.integers(10, 20))
                dx = stroke_height * math.tan(slant_rad)
                draw.line(
                    [(x, y), (x + dx, y - stroke_height)],
                    fill=ink,
                    width=stroke_width,
                )
                x += int(rng.integers(4, 9) * spacing_scale)
            x += int(rng.integers(10, 22) * spacing_scale)
        y += line_height


def _render_sample_image(
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    quality: str,
    slant_deg: float,
    stroke_width: int,
    spacing_scale: float,
) -> Image.Image:
    """Render one synthetic handwriting-like sample image."""
    params = _QUALITY_RENDER_PARAMS[quality]
    image = Image.new("L", size, color=int(params["background"]))
    draw = ImageDraw.Draw(image)
    _draw_handwriting_like_strokes(
        draw,
        size,
        rng,
        slant_deg=slant_deg,
        stroke_width=stroke_width,
        spacing_scale=spacing_scale,
        ink=int(params["ink"]),
    )
    blur_radius = params["blur_radius"]
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image.convert("RGB")


def generate_fixture_dataset(
    output_dir: Path,
    count: int = 20,
    seed: int = 0,
) -> list[ManifestEntry]:
    """Generate a synthetic fixture dataset of handwriting-like samples.

    Renders ``count`` procedurally generated PNG images under
    ``output_dir / "images"`` and writes a matching manifest of
    :class:`ManifestEntry` records to ``output_dir / "manifest.json"``
    (an array of ``entry.to_dict()``, in the same format validated by
    ``tests/fixtures/sample_manifest.json`` and ``manifest.schema.json``).

    Rendered properties vary meaningfully across the set: simulated slant
    angle, simulated pressure (stroke width), letter/word spacing, overall
    image size, and image-quality tier (~70% "high", ~20% "medium", ~10%
    "low" -- see :data:`QUALITY_PROPORTIONS`; "low" samples are rendered
    with genuinely low contrast and heavy blur, not merely labeled that
    way). Samples are unlabeled (``labels={}``): this is synthetic fixture
    data, not real graphological ground truth.

    Every random choice is drawn from a single :class:`numpy.random.Generator`
    seeded from ``seed`` (never numpy's global random state), in a fixed
    order depending only on ``count``. Calling this function twice with the
    same ``count`` and ``seed`` therefore produces byte-identical images
    and manifest content, regardless of ``output_dir``.

    Args:
        output_dir: Directory to write ``images/`` and ``manifest.json``
            into. Created if it does not already exist.
        count: Number of samples to generate.
        seed: Seed for the deterministic random generator driving every
            varied property (quality-tier assignment, slant, stroke width,
            spacing, size, acquisition method).

    Returns:
        The list of generated :class:`ManifestEntry` records, in the same
        order as ``manifest.json``.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    quality_tiers = _assign_quality_tiers(count, rng)

    entries: list[ManifestEntry] = []
    for i in range(count):
        sample_id = f"synthetic-{i:04d}"
        quality = quality_tiers[i]

        width = int(rng.integers(*_WIDTH_RANGE))
        height = int(rng.integers(*_HEIGHT_RANGE))
        slant_deg = float(rng.uniform(*_SLANT_DEGREES_RANGE))
        stroke_width = int(rng.integers(*_STROKE_WIDTH_RANGE))
        spacing_scale = float(rng.uniform(*_SPACING_SCALE_RANGE))
        acquisition_method = "photo" if rng.random() < 0.5 else "scan"

        image = _render_sample_image(
            (width, height),
            rng,
            quality=quality,
            slant_deg=slant_deg,
            stroke_width=stroke_width,
            spacing_scale=spacing_scale,
        )
        image.save(images_dir / f"{sample_id}.png", format="PNG")

        entries.append(
            ManifestEntry(
                sample_id=sample_id,
                acquisition_method=acquisition_method,
                quality=quality,
                language="en",
                labels={},
            )
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump([entry.to_dict() for entry in entries], f, indent=2)
        f.write("\n")

    return entries
