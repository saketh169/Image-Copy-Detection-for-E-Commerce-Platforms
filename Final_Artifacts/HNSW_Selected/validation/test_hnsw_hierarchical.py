import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnsw_hierarchical import build_hnsw


class HNSWConstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.centroids = rng.normal(size=(40, 12)).astype(np.float32)

    def test_seed_reproduces_structure_and_canonical_entry(self):
        first = build_hnsw(self.centroids, m=4, ef_construction=16, seed=9)
        second = build_hnsw(self.centroids, m=4, ef_construction=16, seed=9)

        np.testing.assert_array_equal(first.levels, second.levels)
        np.testing.assert_array_equal(first.adjacency, second.adjacency)
        np.testing.assert_array_equal(first.degrees, second.degrees)
        self.assertEqual(int(first.entry_points[0]), first.entry_point)
        self.assertEqual(len(first.entry_points), 1)

    def test_multi_entry_extension_has_spatially_diverse_starts(self):
        index = build_hnsw(
            self.centroids,
            m=4,
            ef_construction=16,
            seed=9,
            entry_count=4,
        )

        self.assertEqual(int(index.entry_points[0]), index.entry_point)
        self.assertEqual(len(np.unique(index.entry_points)), 4)
        np.testing.assert_array_equal(index.entry_levels, index.levels[index.entry_points])
        self.assertEqual(int(index.entry_levels[0]), index.max_level)
        selected = index.entry_points
        d2 = np.sum((self.centroids[selected, None] - self.centroids[None, selected]) ** 2, axis=2)
        np.testing.assert_array_equal(np.diag(d2), np.zeros(len(selected)))
        self.assertGreater(float(np.min(d2[np.triu_indices(len(selected), 1)])), 0.0)


if __name__ == "__main__":
    unittest.main()
