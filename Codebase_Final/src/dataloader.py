"""Compatibility entry point for the canonical Siamese dataset loader."""

from data.loaders.image_dataset_loader import SiameseDataset, get_dataloader, get_transform

__all__ = ["SiameseDataset", "get_dataloader", "get_transform"]
