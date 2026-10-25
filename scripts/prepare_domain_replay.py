"""Prepare a fixed, native-resolution train replay cohort with two RGB crops per frame."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from sample_video import sha256

FRAME = re.compile(r"img(\d{3})(\d{3})\.jpg")
BOXES = ((192, 128, 704, 640), (1152, 512, 1664, 1024))
MAX_SOURCE = 16 * 2**20
MAX_OUTPUT = 32 * 2**20


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def annotation(path):
    if not 0 < path.stat().st_size <= MAX_SOURCE:
        raise ValueError(f"MAT outside size limit: {path}")
    info = loadmat(path)["image_info"][0, 0][0, 0]
    points = np.asarray(info["location"], dtype=np.float64)
    number = info["number"].item()
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) > 512
            or not np.isfinite(points).all() or number != len(points)
            or np.any(points[:, :2] < 0) or np.any(points[:, 0] >= 1920)
            or np.any(points[:, 1] >= 1080)
            or len(np.unique(points[:, 2])) != len(points)):
        raise ValueError(f"Malformed or out-of-frame points: {path}")
    return points[:, :2].tolist()


def select_sequences(sequences):
    """Choose eight unique, evenly spaced IDs from an already sorted sequence list."""
    if len(sequences) < 8:
        raise ValueError("At least eight train sequences are required")
    indices = [round(i * (len(sequences) - 1) / 7) for i in range(8)]
    selected = [sequences[index] for index in indices]
    if len(set(selected)) != 8:
        raise ValueError("Equidistant selection did not produce eight unique sequences")
    return selected


def prepare(dataset, output):
    dataset = Path(dataset).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output and dataset must be separate")
    splits = {}
    for split in ("train_data", "test_data", "val_data"):
        root = dataset / split / "images"
        sequences = {}
        for path in sorted(root.glob("*.jpg")):
            match = FRAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected image basename: {path.name}")
            sequences.setdefault(match[1], []).append(path)
        splits[split] = sequences
    train_ids = set(splits["train_data"])
    if not train_ids or train_ids & (set(splits["test_data"]) | set(splits["val_data"])):
        raise ValueError("Train sequence IDs must be nonempty and disjoint from test/val")
    selected_ids = select_sequences(sorted(splits["train_data"]))
    selected = [(seq, splits["train_data"][seq][len(splits["train_data"][seq])//2])
                for seq in selected_ids]
    output.mkdir(parents=True, exist_ok=False)
    entries, sources = [], {}
    for sequence, image in selected:
        mat = dataset / "train_data/ground_truth" / f"GT_{image.stem}.mat"
        if not 0 < image.stat().st_size <= MAX_SOURCE:
            raise ValueError(f"Image outside size limit: {image}")
        before = identity(image), identity(mat)
        with Image.open(image) as source:
            if source.size != (1920, 1080):
                raise ValueError(f"Expected native 1920x1080 source: {image}")
            rgb = source.convert("RGB")
            full_points = annotation(mat)
            width, height = rgb.size
            for box_index, box in enumerate(BOXES):
                x, y, right, bottom = box
                crop = rgb.crop(box)
                name = f"{image.stem}-crop{box_index:02d}"
                ppm = output / f"frame-{len(entries):03d}.ppm"
                ppm.write_bytes(f"P6\n512 512\n255\n".encode() + crop.tobytes())
                entries.append({"index": len(entries), "name": name, "origin": "original",
                    "width": 512, "height": 512, "rgb_file": ppm.name, "rgb_sha256": sha256(ppm),
                    "source_image": str(image), "source_sha256": sha256(image),
                    "source_mat": str(mat), "source_mat_sha256": sha256(mat),
                    "source_group": f"original-sequence-{sequence}", "full_points_xy": full_points,
                    "full_width": width, "full_height": height, "box_xyxy": list(box), "split": "train"})
        if before != (identity(image), identity(mat)):
            raise ValueError(f"Dataset source changed during preparation: {image}")
        sources[str(image)] = sha256(image)
        sources[str(mat)] = sha256(mat)
    manifest = {"selection": "eight equidistant IDs including endpoints from sorted train sequences; midpoint available frame; two native crops",
        "entries": entries, "sequences": {"train": sorted(train_ids),
        "test": sorted(splits["test_data"]), "val": sorted(splits["val_data"])},
        "source_hashes": sources, "script_sha256": sha256(__file__), "resize": False}
    target = output / "replay.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p.stat().st_size for p in output.iterdir())
    if total > MAX_OUTPUT:
        raise ValueError(f"Output exceeds 32 MiB: {total}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = prepare(args.dataset_dir, args.output_dir)
    print(json.dumps({"replay": str(path), "sha256": sha256(path), "entries": 16}))


if __name__ == "__main__":
    main()
