# Labeling Rubric Template

This rubric supports graphologists in producing consistent ground-truth
("gold") labels for handwriting samples in the dataset. It lists the
indicators the analyzer pipeline will eventually estimate from images, so
that human labels and future automated estimates can be compared on the
same terms.

The language below is deliberately cautious and descriptive rather than
diagnostic. Labels describe observable features of a writing sample; they
are not claims about a person's character, health, or psychological state,
and no single indicator should be treated as conclusive on its own. This
tool is a support aid for professional graphologists, not a diagnostic
instrument, and labels recorded here should be understood in that spirit.

For each indicator, record a label/value string (e.g. `"right"`,
`"medium"`, `"regular"`) under the corresponding key in a sample's
`labels` map. Use `"unclear"` or leave the indicator out of `labels` when
a sample does not permit a confident reading (e.g. due to poor scan
quality).

## Pressure

Definition: The apparent force behind the strokes, inferred from stroke
weight and ink density rather than measured directly (pen pressure cannot
be recovered from a static image with certainty).

- Positive example (firm/heavy pressure): strokes appear dark and
  thick with consistent ink saturation throughout, suggesting a firmer
  hand.
- Negative example (light pressure): strokes appear thin and pale with
  uneven ink coverage, suggesting a lighter touch.

## Slant

Definition: The angle of vertical strokes (e.g. ascenders, stems)
relative to the baseline.

- Positive example (right slant): strokes lean noticeably rightward,
  which may be associated with a forward-oriented, outward-facing writing
  style.
- Negative example (left slant / vertical): strokes are upright or lean
  left, which may be associated with a more reserved or self-contained
  writing style.

## Letter Size

Definition: The relative height and width of lowercase letters compared
to the ruled line or to the writer's overall script.

- Positive example (larger letters): letter bodies occupy noticeably
  more vertical space than the surrounding baseline grid would suggest is
  typical for the sample.
- Negative example (smaller letters): letter bodies are compact and
  occupy comparatively little vertical space.

## Spacing

Definition: The horizontal gaps between letters, words, and lines.

- Positive example (generous spacing): words and lines are clearly
  separated with visible white space, giving the text an open, uncrowded
  appearance.
- Negative example (tight spacing): words, letters, or lines sit close
  together with little visible white space, giving the text a dense,
  crowded appearance.

## Baseline Movement

Definition: How consistently a line of writing follows a straight,
horizontal path.

- Positive example (steady baseline): successive words sit close to a
  consistent horizontal line across the page.
- Negative example (wandering baseline): the line of writing drifts
  upward, downward, or wavers noticeably as it crosses the page.

## Margins

Definition: The blank space left at the edges of the page (left, right,
top, bottom) relative to the writing area.

- Positive example (balanced margins): margins on both sides are
  present and roughly consistent in width across the sample.
- Negative example (irregular/absent margins): writing runs close to or
  off the edge of the page on one or more sides, or margin width varies
  noticeably from line to line.

## Rhythm

Definition: The overall regularity and flow of the writing — how evenly
stroke shapes, sizes, and spacing repeat across the sample.

- Positive example (regular rhythm): letterforms, sizes, and spacing
  repeat in a consistent, evenly paced manner across the sample.
- Negative example (irregular rhythm): letterforms, sizes, or spacing
  vary noticeably from word to word or line to line, giving an uneven,
  halting appearance.

## Stroke Continuity

Definition: The degree to which strokes within a word are joined versus
lifted/broken (connectedness of letterforms).

- Positive example (connected strokes): letters within a word are
  mostly joined by continuous strokes with few pen lifts.
- Negative example (broken/disconnected strokes): letters within a word
  are frequently separated by pen lifts, appearing more segmented or
  printed.

## Overall Organization

Definition: How the writing is arranged on the page as a whole —
alignment of lines, use of space, and apparent planning of layout (e.g.
title placement, paragraph structure, date/signature position).

- Positive example (organized layout): lines are aligned, spacing
  between sections is consistent, and the page layout appears planned.
- Negative example (disorganized layout): lines are unevenly aligned,
  spacing between sections is inconsistent, and the page layout appears
  unplanned.
