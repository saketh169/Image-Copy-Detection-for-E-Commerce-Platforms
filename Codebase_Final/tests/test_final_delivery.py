import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_DATASETS = ("ucid", "flickr", "cifar", "amazon")


def load_contract():
    path = Path(__file__).resolve().parents[1] / "configs" / "final_graph_choices.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class FinalDeliveryContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_core_source_files_are_real_implementations(self):
        loader = (self.ROOT / "src/data/loaders/training_pair_loader.py").read_text(encoding="utf-8")
        augmentor = (self.ROOT / "src/data/augmentation/attack_augmentor.py").read_text(encoding="utf-8")
        materializer = (self.ROOT / "src/data/materialization/build_amazon50k_compact_light3.py").read_text(encoding="utf-8")
        self.assertIn("SiameseDataset", loader)
        self.assertIn("class CopyAttackAugmentor", augmentor)
        self.assertIn("def main", materializer)
        self.assertNotIn("patch point not found", loader)

    def test_all_datasets_are_exact(self):
        contract = load_contract()
        self.assertEqual(tuple(contract["datasets"]), EXPECTED_DATASETS)
        self.assertTrue(all(item["exact_match_percent"] == 100.0 for item in contract["datasets"].values()))

    def test_amazon_uses_full_bank(self):
        contract = load_contract()
        self.assertEqual(contract["datasets"]["amazon"]["embedding_bank_size"], 35000)


if __name__ == "__main__":
    unittest.main()
