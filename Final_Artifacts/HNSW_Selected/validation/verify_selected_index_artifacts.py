"""Validate shapes and memberships of the four report-selected index bundles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "selected_indexes"
MANIFEST = ROOT / "index_manifest.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    require(manifest["embedding_space"]["dimensions"] == 384, "expected 384-D embeddings")

    for dataset, record in manifest["datasets"].items():
        folder = ROOT / dataset
        configuration = record["configuration"]
        files = record["files"]
        centroid = np.load(folder / files["centroid_layout"], allow_pickle=False)
        graph = np.load(folder / files["hnsw_graph"], allow_pickle=False)
        radii = np.load(folder / files["bucket_radii"], allow_pickle=False)

        count = configuration["centroids"]
        bank = record["bank_embeddings"]
        require(centroid["centroids"].shape == (count, 384), f"{dataset}: centroid shape")
        require(centroid["labels"].shape == (bank,), f"{dataset}: label shape")
        require(graph["adj"].shape == (count, configuration["degree"]), f"{dataset}: graph shape")
        require(radii.shape == (count,), f"{dataset}: radii shape")
        require(np.all((centroid["labels"] >= 0) & (centroid["labels"] < count)), f"{dataset}: labels")
        require(np.all(np.isfinite(centroid["centroids"])), f"{dataset}: finite centroids")
        require(np.all(np.isfinite(radii) & (radii >= 0)), f"{dataset}: finite radii")

        overlap_file = files["overlap_membership"]
        if configuration["overlap"] == 1:
            require(overlap_file.startswith("derived from"), f"{dataset}: single assignment declaration")
            membership_count = bank
        else:
            overlap = np.load(folder / overlap_file, allow_pickle=False)
            require(overlap["overlap"].shape == (1,), f"{dataset}: overlap metadata")
            require(int(overlap["overlap"][0]) == configuration["overlap"], f"{dataset}: overlap value")
            require(overlap["offsets"].shape == (count + 1,), f"{dataset}: offsets")
            membership_count = int(overlap["cluster_indices"].size)
        require(membership_count == bank * configuration["overlap"], f"{dataset}: membership count")
        print(f"{dataset}: OK ({bank} bank vectors, C={count}, overlap={configuration['overlap']})")


if __name__ == "__main__":
    main()
