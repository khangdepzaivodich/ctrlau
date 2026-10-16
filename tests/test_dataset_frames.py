"""Regression checks for verified DISFA annotation-to-image correspondence."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from config import DISFA_AUS
from dataset import DISFADataset, _resolve_disfa_frame_paths


class DISFAFrameAlignmentTests(unittest.TestCase):
    def test_zero_based_images_keep_one_based_annotation_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "img" / "SN002"
            labels = root / "ActionUnit_Labels" / "SN002"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (4, 4), "red").save(images / "0.png")
            Image.new("RGB", (4, 4), "blue").save(images / "1.png")
            for au in DISFA_AUS:
                (labels / f"SN002_au{au}.txt").write_text("1,2\n2,0\n", encoding="utf-8")
            dataset = DISFADataset(root, subjects=["SN002"], transform=ToTensor())
            self.assertEqual(len(dataset), 2)
            first, second = dataset[0], dataset[1]
            self.assertEqual((first["frame"], second["frame"]), (1, 2))
            self.assertTrue(torch.equal(first["au_labels"], torch.ones(8)))
            self.assertTrue(torch.equal(second["au_labels"], torch.zeros(8)))
            self.assertEqual(first["image"][:, 0, 0].tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(second["image"][:, 0, 0].tolist(), [0.0, 0.0, 1.0])

    def test_one_based_images_keep_exact_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0001.png", "0002.png"):
                Image.new("RGB", (1, 1)).save(root / name)
            paths = _resolve_disfa_frame_paths(root, {1, 2})
            self.assertEqual(Path(paths[1]).name, "0001.png")
            self.assertEqual(Path(paths[2]).name, "0002.png")

    def test_incomplete_images_are_not_silently_shifted_or_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in (0, 2):
                Image.new("RGB", (1, 1)).save(root / f"{frame}.png")
            with self.assertRaisesRegex(FileNotFoundError, "no frames were skipped"):
                _resolve_disfa_frame_paths(root, {1, 2, 3})

    def test_duplicate_numeric_image_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("1.png", "0001.png"):
                Image.new("RGB", (1, 1)).save(root / name)
            with self.assertRaisesRegex(ValueError, "Ambiguous DISFA image frame"):
                _resolve_disfa_frame_paths(root, {1})


if __name__ == "__main__":
    unittest.main()
