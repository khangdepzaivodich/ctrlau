"""Check storage/decode separation using real encoded images, no model."""
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from diagnose_training_io import STAGES, SlowOperation, probe_image


class TrainingIOTests(unittest.TestCase):
    def test_header_wait_is_not_charged_to_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (17, 23), (123, 45, 67)).save(path)
            encoded = path.read_bytes()
            original_open = Path.open

            class DelayedReader:
                def __init__(self, handle):
                    self.handle = handle

                def read(self, size=-1):
                    if size == 16:
                        time.sleep(.08)
                    return self.handle.read(size)

                def close(self):
                    self.handle.close()

            def delayed_open(filename, *args, **kwargs):
                return DelayedReader(original_open(filename, *args, **kwargs))

            seen = []
            with SlowOperation(10) as monitor, patch.object(Path, "open", delayed_open):
                row, actual = probe_image(path, lambda image: seen.append(image.getpixel((0, 0))), monitor)
            self.assertEqual(actual, encoded)
            self.assertEqual(row["sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(row["bytes"], len(encoded))
            self.assertEqual(seen, [(123, 45, 67)])
            self.assertGreaterEqual(row["header_read_s"], .07)
            self.assertLess(row["header_read_cpu_s"], row["header_read_s"])
            for stage in STAGES:
                self.assertGreaterEqual(row[f"{stage}_s"], 0)
            self.assertLess(row["decode_s"], row["header_read_s"])

    def test_invalid_image_raises_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_bytes(b"not an image")
            with SlowOperation(10) as monitor, self.assertRaises(UnidentifiedImageError):
                probe_image(path, lambda image: image, monitor)


if __name__ == "__main__":
    unittest.main()
