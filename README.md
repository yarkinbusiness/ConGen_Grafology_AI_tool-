# ConGen Grafology AI Tool

ConGen Grafology AI Tool is a Python pipeline that supports professional
graphologists in analyzing handwriting samples. It validates a graphological
image dataset and runs a classical image-processing feature analyzer —
covering slant, pressure-proxy, spacing, baseline, and margins — as a
stand-in for a not-yet-trainable machine learning model. The pipeline
produces structured markdown reports intended to assist expert graphologists
in their evaluation; it is a support tool, not a diagnostic tool, and does
not replace professional judgment.

## Development

Install the package in editable mode with the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```
