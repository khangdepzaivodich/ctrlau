"""Copy a dataset once to an explicitly selected local directory before training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
from pathlib import Path
import tempfile


def stage_dataset(source, destination):
    """Publish a new dataset directory only after all copied bytes are verified.

    Never overwrite an existing destination or fall back to the source on error.
    A failed copy is retained in a temporary sibling directory for inspection.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and staging destination must be separate directory trees")
    if destination.exists():
        raise FileExistsError(
            f"Staging destination already exists: {destination}. "
            "Use --data_root with a previously completed copy, or choose a new staging destination."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.copy-", dir=destination.parent))
    print(f"Copying dataset: {source} -> {destination}; verifying every file before training.", flush=True)

    def entries(folder):
        # Explicit traversal propagates listing failures instead of skipping data.
        for item in folder.iterdir():
            if item.is_symlink():
                raise ValueError(f"Dataset staging requires ordinary files/directories: {item}")
            relative = item.relative_to(source)
            if item.is_dir():
                (temporary / relative).mkdir()
                yield from entries(item)
            elif item.is_file():
                yield item, temporary / relative
            else:
                raise ValueError(f"Unsupported dataset entry: {item}")

    def copy_verified(pair):
        original, copied = pair
        digest = hashlib.sha256()
        with original.open("rb") as reader, copied.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
        with copied.open("rb") as reader:
            actual = hashlib.file_digest(reader, "sha256").digest()
        if actual != digest.digest():
            raise OSError(f"Copied dataset file failed SHA256 verification: {original}")

    count = 0
    try:
        jobs = entries(source)
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Bound pending work; do not queue the entire dataset in memory.
            while batch := list(islice(jobs, 256)):
                for _ in pool.map(copy_verified, batch):
                    count += 1
                if count % 4096 == 0:
                    print(f"Dataset copy: {count} files verified", flush=True)
        if not count:
            raise ValueError(f"Source dataset contains no files: {source}")
        if destination.exists():
            raise FileExistsError(f"Staging destination appeared during copying: {destination}")
        temporary.rename(destination)
    except Exception as error:
        raise RuntimeError(f"Dataset staging failed; training has not started. Partial copy: {temporary}") from error
    print(f"Dataset ready: {destination} ({count} files verified)", flush=True)
    return str(destination)
