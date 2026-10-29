"""Measure real training-image I/O without loading a model or training.

Times storage reads separately from in-memory PIL decoding and the repository's
training transforms. Optional local copies use exactly the bytes read from the
source, verified with SHA256. Cache state is uncontrolled; passes are not cold
cache measurements. CSV is flushed after every image, not printed per image.
"""
import argparse
import csv
import hashlib
import io
import random
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


STAGES = ("open", "header_read", "remaining_read", "decode", "transform")


class SlowOperation:
    """Report an outstanding operation once, only if it actually runs slowly."""

    def __init__(self, seconds=10):
        self.seconds = seconds
        self.lock = threading.Lock()
        self.active = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while not self.stop.wait(min(1, self.seconds)):
            with self.lock:
                if self.active is None:
                    continue
                stage, path, started, reported = self.active
                elapsed = time.perf_counter() - started
                if elapsed >= self.seconds and not reported:
                    print(f"WAITING {elapsed:.1f}s: {stage}: {path}", flush=True)
                    self.active = (stage, path, started, True)

    @contextmanager
    def operation(self, stage, path):
        with self.lock:
            self.active = (stage, str(path), time.perf_counter(), False)
        try:
            yield
        finally:
            with self.lock:
                self.active = None

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)


def probe_image(path, transform, monitor):
    """Return actual measurements and original encoded bytes; never skip errors."""
    row = {"path": str(path)}

    def measured(stage, function):
        with monitor.operation(stage, path):
            wall, cpu = time.perf_counter(), time.process_time()
            value = function()
            row[f"{stage}_s"] = time.perf_counter() - wall
            row[f"{stage}_cpu_s"] = time.process_time() - cpu
            return value

    handle = measured("open", lambda: Path(path).open("rb"))
    try:
        # PIL.Image.open starts with fp.read(16), the operation in the logs.
        prefix = measured("header_read", lambda: handle.read(16))
        remaining = measured("remaining_read", handle.read)
    finally:
        handle.close()
    encoded = prefix + remaining

    def decode():
        # No source filesystem reads here: decode already-read bytes in memory.
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB")

    image = measured("decode", decode)
    try:
        measured("transform", lambda: transform(image))
    finally:
        image.close()
    row["bytes"] = len(encoded)
    row["sha256"] = hashlib.sha256(encoded).hexdigest()
    return row, encoded


def report(rows, name):
    print(f"\n{name}: {len(rows)} images (wall / CPU seconds)")
    for stage in STAGES:
        walls = sorted(row[f"{stage}_s"] for row in rows)
        cpu = sum(row[f"{stage}_cpu_s"] for row in rows)
        p95 = walls[min(len(walls) - 1, int(.95 * (len(walls) - 1)))]
        print(f"  {stage:15s} total={sum(walls):8.3f} / {cpu:8.3f}; "
              f"median={statistics.median(walls):.6f}; p95={p95:.6f}; max={walls[-1]:.3f}")
    slowest = sorted(rows, key=lambda row: sum(row[f"{s}_s"] for s in STAGES[:3]), reverse=True)[:5]
    print("  Slowest source-file operations (open + header + remaining read):")
    for row in slowest:
        seconds = sum(row[f"{stage}_s"] for stage in STAGES[:3])
        print(f"    {seconds:.3f}s  {row['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=("disfa", "bp4d"), default="disfa")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slow-seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=Path("training_io.csv"))
    parser.add_argument("--local-dir", type=Path,
                        help="Optional parent for a NEW temporary directory holding selected image copies")
    args = parser.parse_args()
    if min(args.samples, args.passes) < 1 or not 0 < args.slow_seconds < float("inf"):
        parser.error("samples, passes and slow-seconds must be positive and finite")

    # Import no model, CLIP encoder, checkpoint or optimizer.
    import torch
    from dataset import DISFADataset, BP4DDataset

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("No training; GPU intentionally unused. CSV flushed per image.", flush=True)
    print("Storage and decode are separated; OS/mount cache state is uncontrolled.", flush=True)
    with SlowOperation(args.slow_seconds) as monitor:
        with monitor.operation("dataset initialization (paths/labels)", args.data_root):
            if args.dataset == "disfa":
                # Identical training subject order and fold mapping to train.py.
                groups = [
                    ["SN002", "SN010", "SN001", "SN026", "SN027", "SN032", "SN030", "SN009", "SN016"],
                    ["SN013", "SN018", "SN011", "SN028", "SN012", "SN006", "SN031", "SN021", "SN024"],
                    ["SN003", "SN029", "SN023", "SN025", "SN008", "SN005", "SN007", "SN017", "SN004"],
                ]
                held_out = {1: 2, 2: 1, 3: 0}[args.fold]
                subjects = sum([group for i, group in enumerate(groups) if i != held_out], [])
                data = DISFADataset(args.data_root, subjects=subjects, train=True, fold=args.fold)
            else:
                data = BP4DDataset(args.data_root, train=True, fold=args.fold)
        if not len(data):
            raise ValueError("Training split is empty")
        indices = random.Random(args.seed).sample(range(len(data)), min(args.samples, len(data)))
        # Both list layouts and DISFA raw tuples store the resolved path last.
        paths = [Path(data.samples[index][-1]) for index in indices]
        print(f"Probing {len(paths)} shuffled training images; {args.passes} passes.", flush=True)
        local_root = None
        if args.local_dir is not None:
            args.local_dir.mkdir(parents=True, exist_ok=True)
            local_root = Path(tempfile.mkdtemp(prefix="ctrlau-io-", dir=args.local_dir))
            print(f"Selected-image copies: {local_root} (kept for inspection)", flush=True)
        fields = ["location", "pass", "index", "path", "bytes", "sha256"]
        fields += [f"{stage}{suffix}" for stage in STAGES for suffix in ("_s", "_cpu_s")]
        hashes = {}
        # Refuse to overwrite an existing report.
        with args.output.open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            output.flush()
            for pass_id in range(1, args.passes + 1):
                for location in (["source", "local"] if local_root else ["source"]):
                    rows = []
                    for position, (index, source) in enumerate(zip(indices, paths), 1):
                        path = source if location == "source" else local_root / f"{index}{source.suffix}"
                        row, encoded = probe_image(path, data.transform, monitor)
                        if location == "source" and pass_id == 1:
                            hashes[index] = row["sha256"]
                            if local_root:
                                destination = local_root / f"{index}{source.suffix}"
                                with monitor.operation("copy selected image to local storage", destination):
                                    destination.write_bytes(encoded)
                        if row["sha256"] != hashes[index]:
                            raise ValueError(f"Image bytes changed or local copy differs: {path}")
                        row.update(location=location, **{"pass": pass_id, "index": index})
                        writer.writerow(row)
                        output.flush()
                        rows.append(row)
                        if position % 256 == 0:
                            recent = rows[-256:]
                            storage = sum(r[f"{s}_s"] for r in recent for s in STAGES[:3])
                            prep = sum(r[f"{s}_s"] for r in recent for s in STAGES[3:])
                            print(f"{location} pass {pass_id}: {position}/{len(paths)}; "
                                  f"last 256: storage={storage:.2f}s, decode+transform={prep:.2f}s", flush=True)
                    report(rows, f"{location} pass {pass_id}")
    print(f"\nReport: {args.output.resolve()}")
    print("Large read wall time with little CPU time means waiting on the input/storage path.")
    print("Faster local reads support a source-storage bottleneck; repeat passes also reflect caching.")
    print("This serial image probe does not measure worker scaling, CUDA compute or end-to-end throughput.")


if __name__ == "__main__":
    main()
