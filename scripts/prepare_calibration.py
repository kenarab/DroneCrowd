"""Freeze one central annotated test image per sequence, without HDF5."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.io import loadmat

from sample_video import memory_limits, sha256


def source_identity(path):
    stat = path.stat()
    # Reading may update atime; compare identity and content-change metadata.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def ground_truth(path, width, height):
    if not 0 < path.stat().st_size <= 4 * 2**20:
        raise ValueError("MAT outside bounded size")
    info = loadmat(path)["image_info"][0, 0][0, 0]
    points = np.asarray(info["location"], dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not 0 <= len(points) <= 512
            or not np.isfinite(points).all() or info["number"].item() != len(points)):
        raise ValueError("Malformed point annotations")
    if (np.any(points[:, :2] < 0) or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height) or len(np.unique(points[:, 2])) != len(points)):
        raise ValueError("Out of bounds or duplicate annotation identities")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(strict=True), args.output_dir.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        parser.error("Output must be separate from dataset")
    sequences = {}
    for path in sorted((dataset / "test_data/images").glob("*.jpg")):
        match = re.fullmatch(r"img(\d{3})(\d{3})", path.stem)
        if not match:
            raise ValueError("Unexpected image basename")
        sequences.setdefault(match[1], []).append(path)
    if not 1 <= len(sequences) <= 30:
        raise ValueError("Expected at most thirty test sequences")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, (sequence, frames) in enumerate(sorted(sequences.items())):
        image = frames[len(frames)//2]
        annotation = dataset / "test_data/ground_truth" / ("GT_" + image.stem + ".mat")
        before = source_identity(image), source_identity(annotation)
        if not 0 < image.stat().st_size <= 8 * 2**20:
            raise ValueError("JPEG exceeds bounded size")
        with Image.open(image) as picture:
            width, height = picture.size
            if (width, height) != (1920, 1080):
                raise ValueError("Historical protocol expects native 1920×1080")
            rgb = picture.convert("RGB").tobytes()
        points = ground_truth(annotation, width, height)
        ppm = output / f"frame-{index:03d}.ppm"
        ppm.write_bytes(f"P6\n{width} {height}\n255\n".encode() + rgb)
        entries.append({"index": index, "sequence": sequence, "frame": int(image.stem[-3:]),
                        "source_image": str(image), "source_image_sha256": sha256(image),
                        "annotation": str(annotation), "annotation_sha256": sha256(annotation),
                        "gt_count": len(points), "points_xy_id": points.tolist(),
                        "width": width, "height": height, "rgb_file": ppm.name, "rgb_sha256": sha256(ppm)})
        if before != (source_identity(image), source_identity(annotation)):
            raise ValueError("Dataset sources changed during preparation")
    report = {"selection": "sorted test sequences; central available image at index len//2, before predictions",
              "entries": entries, "decoder": "Pillow RGB, no resize (historical recipe)",
              "dataset": str(dataset), "script_sha256": sha256(__file__), "memory_limits": memory_limits(),
              "interpretation": "Annotated subset, not the full published benchmark"}
    path = output / "cohort.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cohort": str(path), "sha256": sha256(path), "images": len(entries)}))


if __name__ == "__main__":
    main()
