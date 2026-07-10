"""Dataset manifest schema and related utilities.

This subpackage defines the structure of a graphological handwriting-sample
dataset manifest: per-sample metadata (anonymous ID, acquisition method,
image quality, language) and ground-truth labels supplied by professional
graphologists. It does not perform validation of a whole dataset directory
or any image processing — see later tasks for that.
"""

from grafology_ai.dataset.schema import ManifestEntry

__all__ = ["ManifestEntry"]
