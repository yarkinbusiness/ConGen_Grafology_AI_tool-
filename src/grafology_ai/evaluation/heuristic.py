"""Extract "the heuristic's pressure reading" from :class:`Features`.

PLUMBING PROOF -- see the package docstring
(``grafology_ai/evaluation/__init__.py``). This module answers one
question: what does the *existing, shipped* heuristic interpretation layer
(:mod:`grafology_ai.interpretation.interpret`) think this sample's
pressure is, expressed in the same ``light``/``moderate``/``firm``
vocabulary the M2.3 plumbing-proof baseline model's labels use (see
:data:`grafology_ai.training.synthetic_labels.PRESSURE_LABELS`)?

Design choice: reuse the real thresholds, don't parse text
-------------------------------------------------------------

The task allowed two approaches: (a) call
:func:`grafology_ai.interpretation.interpret.interpret` and derive a
bucket from its pressure :class:`~grafology_ai.interpretation.interpret.Finding`,
or (b) independently bucket ``Features.stroke_width_mean`` using the same
thresholds :func:`grafology_ai.interpretation.interpret._candidate_pressure`
actually uses.

This module takes **(b)**, implemented by *importing* the interpretation
layer's own threshold constants
(:data:`grafology_ai.interpretation.interpret.PRESSURE_LIGHT_MAX_PX` and
:data:`grafology_ai.interpretation.interpret.PRESSURE_FIRM_MIN_PX`) rather
than copying their numeric values -- so this module can never silently
drift out of sync with the interpretation layer's real behavior. If those
thresholds are ever recalibrated (e.g. once real labeled data exists),
this module's bucketing changes with them automatically.

Approach (a) was rejected because it is fragile in the wrong way for this
purpose: :func:`~grafology_ai.interpretation.interpret._candidate_pressure`'s
``Finding.observation``/``interpretation`` text never actually contains
the literal words "light"/"moderate"/"firm" (its hedged narrative text says
things like "a comparatively light touch" and "a moderate, middle-of-the-
range touch") -- recovering a clean label would mean parsing prose that is
explicitly documented as free-form, hedged narrative rather than a stable
machine-readable contract. Reading the *decision thresholds* the same
function branches on, and reproducing the same three-way branch here, is
strictly more robust and is exactly "the REAL heuristic that ships in this
product" (its actual decision boundary), not a reinvented one.

Note: :func:`_candidate_pressure`'s three branches already line up
one-to-one with :data:`~grafology_ai.training.synthetic_labels.PRESSURE_LABELS`
(``value <= PRESSURE_LIGHT_MAX_PX`` -> light-touch descriptor,
``value >= PRESSURE_FIRM_MIN_PX`` -> firm-touch descriptor, otherwise a
moderate/middle-of-the-range descriptor), so no additional bucket-name
remapping is needed beyond attaching the matching
:data:`~grafology_ai.training.synthetic_labels.PRESSURE_LABELS` string to
each branch.
"""

from __future__ import annotations

from grafology_ai.analysis.features import Features
from grafology_ai.interpretation.interpret import PRESSURE_FIRM_MIN_PX, PRESSURE_LIGHT_MAX_PX
from grafology_ai.training.synthetic_labels import PRESSURE_LABELS


def extract_heuristic_pressure_reading(features: Features) -> str:
    """Return the shipped heuristic interpretation layer's "pressure" bucket.

    Reproduces the exact three-way branch
    :func:`grafology_ai.interpretation.interpret._candidate_pressure` uses
    on ``features.stroke_width_mean``, via the same imported threshold
    constants (see the module docstring's "Design choice" section), and
    reports it using the label vocabulary in :data:`PRESSURE_LABELS`
    (``"light"``, ``"moderate"``, ``"firm"``) so it is directly comparable
    to :func:`grafology_ai.training.baseline.predict`'s output.

    Args:
        features: A measured :class:`~grafology_ai.analysis.features.Features`
            record.

    Returns:
        One of :data:`PRESSURE_LABELS`.
    """
    value = features.stroke_width_mean
    if value <= PRESSURE_LIGHT_MAX_PX:
        return PRESSURE_LABELS[0]  # "light"
    if value >= PRESSURE_FIRM_MIN_PX:
        return PRESSURE_LABELS[2]  # "firm"
    return PRESSURE_LABELS[1]  # "moderate"
