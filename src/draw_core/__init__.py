"""DRAW core: pure segmentation policy.

Importing this package has no side effects and does NOT pull in torch/nnU-Net —
those live behind the optional ``gpu`` extra and are imported lazily inside the
functions that actually run inference. This keeps core importable (and testable)
on a machine with no GPU.
"""
