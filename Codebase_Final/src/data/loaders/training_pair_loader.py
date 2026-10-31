"""Canonical training-pair loader entry point.

The implementation lives in :mod:`dataloader` because the model-selection
scripts and the public training API share the same dataset semantics. This
module provides the package-level import path without a second loader.
"""

from dataloader import SiameseDataset, get_dataloader, get_transform

__all__ = ["SiameseDataset", "get_dataloader", "get_transform"]
