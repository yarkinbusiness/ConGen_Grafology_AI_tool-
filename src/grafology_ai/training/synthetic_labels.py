"""Synthetic "pressure" label generator -- FOR TESTING ONLY, not a real labeling tool.

PLUMBING PROOF ONLY -- see ``grafology_ai/training/__init__.py``'s package
docstring. The client's technical proposal explicitly excludes building a
real labeling tool; nothing here is that. This module exists purely to
manufacture fake-but-plausible-looking labels for the "pressure" rubric
indicator (see ``docs/labeling_rubric.md``) so that
:func:`grafology_ai.training.baseline.train_baseline` and
:func:`grafology_ai.training.baseline.predict` have *something* to run
against in tests, since
:func:`grafology_ai.dataset.fixtures.generate_fixture_dataset` always
produces empty (``{}``) labels for every sample (there is no real labeled
data yet -- see that module's docstring).

The bucketing below is a documented-arbitrary threshold on
``stroke_width_mean``, chosen only to land within the
``_STROKE_WIDTH_RANGE = (2, 9)`` pixel range
:mod:`grafology_ai.dataset.fixtures` renders synthetic samples with, so a
generated fixture set produces a mix of all three labels rather than a
single degenerate bucket. It has no basis in real graphological pressure
assessment and must never be treated as, or confused with, a graphologist's
ground-truth label.
"""

from __future__ import annotations

from collections.abc import Sequence

from grafology_ai.analysis.features import Features

#: Label vocabulary this generator emits, in low-to-high pressure order.
#: Matches the qualitative "light"/"firm" language used by
#: ``docs/labeling_rubric.md``'s Pressure indicator.
PRESSURE_LABELS: tuple[str, str, str] = ("light", "moderate", "firm")

#: Documented-arbitrary thresholds (pixels) on ``stroke_width_mean``, the
#: fixture generator's simulated pressure/stroke-width range being
#: ``(2, 9)`` (see :mod:`grafology_ai.dataset.fixtures`). Chosen only to
#: split that range into three non-trivial buckets for testing -- NOT
#: calibrated against any real measurement of pen pressure.
_LIGHT_MAX_STROKE_WIDTH = 3.5
_FIRM_MIN_STROKE_WIDTH = 6.0


def generate_synthetic_pressure_label(features: Features) -> str:
    """Fabricate a fake "pressure" label from one sample's :class:`Features`.

    FOR TESTING ONLY -- see the module docstring. Deterministically buckets
    ``features.stroke_width_mean`` into ``"light"``, ``"moderate"``, or
    ``"firm"`` (see :data:`PRESSURE_LABELS`) using fixed, documented,
    arbitrary thresholds. This is not a real graphological judgment.

    Returns:
        One of :data:`PRESSURE_LABELS`.
    """
    if features.stroke_width_mean < _LIGHT_MAX_STROKE_WIDTH:
        return "light"
    if features.stroke_width_mean < _FIRM_MIN_STROKE_WIDTH:
        return "moderate"
    return "firm"


def generate_synthetic_pressure_labels(features_list: Sequence[Features]) -> list[str]:
    """Fabricate fake "pressure" labels for a list of :class:`Features`.

    FOR TESTING ONLY -- see the module docstring and
    :func:`generate_synthetic_pressure_label`. Purely a per-element map;
    provided as a convenience for building
    ``list[tuple[Features, str]]`` training data in tests.

    Returns:
        A list of labels, one per element of ``features_list``, in the
        same order, each one of :data:`PRESSURE_LABELS`.
    """
    return [generate_synthetic_pressure_label(features) for features in features_list]
