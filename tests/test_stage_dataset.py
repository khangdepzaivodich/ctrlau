"""Staging must preserve bytes/layout and stop training on incomplete copies."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage_dataset import stage_dataset


class DatasetStagingTests(unittest.TestCase):
    def test_copies_bytes_and_layout_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "local"
            (source / "img" / "SN002").mkdir(parents=True)
            (source / "ActionUnit_Labels").mkdir()
            (source / "empty").mkdir()
            (source / "img" / "SN002" / "0.png").write_bytes(bytes(range(256)) * 8193)
            (source / "ActionUnit_Labels" / "labels.txt").write_bytes(b"1,2\n2,0\n")
            originals = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                result = stage_dataset(source, target)
            self.assertEqual(result, str(target.resolve()))
            self.assertEqual(originals, {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()})
            self.assertEqual(originals, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})
            self.assertTrue((target / "empty").is_dir())
            with self.assertRaises(FileExistsError):
                stage_dataset(source, target)
            with self.assertRaises(ValueError):
                stage_dataset(source, source / "nested")

    def test_read_failure_does_not_publish_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            image = source / "frame.png"
            image.write_bytes(b"source bytes")
            target = Path(directory) / "local"
            real_open = Path.open

            def failing_open(path, *args, **kwargs):
                if path == image:
                    raise OSError("injected input read failure")
                return real_open(path, *args, **kwargs)

            with patch.object(Path, "open", failing_open), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "training has not started") as failure:
                    stage_dataset(source, target)
            self.assertIsInstance(failure.exception.__cause__, OSError)
            self.assertFalse(target.exists())
            self.assertEqual(image.read_bytes(), b"source bytes")

    def test_checksum_failure_does_not_publish_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "frame.png").write_bytes(b"source bytes")
            target = Path(directory) / "local"
            with patch("stage_dataset.hashlib.file_digest") as check, contextlib.redirect_stdout(io.StringIO()):
                check.return_value.digest.return_value = b"wrong digest"
                with self.assertRaises(RuntimeError):
                    stage_dataset(source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
