"""Compare historical adaptive and fixed-sigma density targets on original frames only."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import cKDTree

from audit_hdf5 import reconstruct_density as audit_reconstruct_density
from sample_video import sha256

WIDTH, HEIGHT = 960, 540
INPUT_WIDTH, INPUT_HEIGHT = 1920, 1080
SOURCE_LIMIT = 64 * 2**20
MAT_LIMIT = 16 * 2**20


def occupied_cells(points, width=WIDTH, height=HEIGHT):
    values = np.asarray(points, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()
            or (values < 0).any() or (values[:, 0] >= width*2).any()
            or (values[:, 1] >= height*2).any()):
        raise ValueError("Points must be finite and inside the native frame")
    cells = np.zeros((height, width), dtype=np.bool_)
    for px, py in values:
        cells[int(py/2), int(px/2)] = True
    ys, xs = np.nonzero(cells)
    return cells, np.column_stack((xs, ys)), len(values)-len(xs)


def adaptive_sigmas(coordinates, height=HEIGHT, width=WIDTH):
    if len(coordinates) == 0:
        raise ValueError("A complete original frame needs annotated points")
    if len(coordinates) == 1:
        return np.asarray([(height+width)/8.0], dtype=np.float64)
    if len(coordinates) < 4:
        raise ValueError("Historical adaptive sigma requires at least four occupied cells")
    distances, _ = cKDTree(coordinates).query(coordinates, k=4)
    return distances[:, 1:4].sum(axis=1) * 0.1


def gaussian_1d(sigma):
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("Sigma must be finite and positive")
    radius = int(4.0*sigma + 0.5)
    positions = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5/(sigma*sigma) * positions**2)
    return kernel/kernel.sum(), radius


def add_impulse(density, x, y, sigma):
    kernel, radius = gaussian_1d(sigma)
    height, width = density.shape
    left, right = max(0, x-radius), min(width, x+radius+1)
    top, bottom = max(0, y-radius), min(height, y+radius+1)
    vertical = kernel[top-y+radius:bottom-y+radius].astype(np.float32)
    horizontal = kernel[left-x+radius:right-x+radius]
    contribution = (vertical[:, None].astype(np.float64)*horizontal[None, :]).astype(np.float32)
    density[top:bottom, left:right] += contribution
    return float(contribution.sum(dtype=np.float64))


def density_map(coordinates, sigmas):
    density = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    retained = sum(add_impulse(density, int(x), int(y), float(sigma))
                   for (x, y), sigma in zip(coordinates, sigmas))
    return density, retained


def reconstruct_density(points):
    """Pure NumPy equivalent of the historical reconstruct_density path."""
    occupied, coordinates, collisions = occupied_cells(points)
    sigmas = adaptive_sigmas(coordinates)
    density, retained = density_map(coordinates, sigmas)
    return occupied, coordinates, collisions, sigmas, density, retained


def quantiles(values):
    return {key: float(np.quantile(values, q)) for key, q in
            (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))}


def map_metrics(density):
    mass = float(density.sum(dtype=np.float64))
    peak = float(density.max(initial=0))
    if mass <= 0:
        entropy = 0.0
    else:
        probabilities = density[density > 0].astype(np.float64)/mass
        entropy = float(-np.sum(probabilities*np.log(probabilities)))
    return {"sum": mass, "peak": peak, "mass_entropy_nats": entropy}


def cosine(left, right):
    a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a, b)/denominator) if denominator else None


def isolated_peak_ratio(sigmas):
    fixed, radius = gaussian_1d(2.0)
    fixed_peak = float(fixed[radius]**2)
    ratios = []
    for sigma in sigmas:
        adaptive, radius = gaussian_1d(float(sigma))
        ratios.append(float(adaptive[radius]**2/fixed_peak))
    return ratios


def validate_mat(path, expected_points):
    if not 0 < path.stat().st_size <= MAT_LIMIT:
        raise ValueError(f"MAT exceeds 16 MiB: {path}")
    info = loadmat(path, variable_names=["image_info"])["image_info"][0, 0][0, 0]
    locations = np.asarray(info["location"], dtype=np.float64)
    number = float(info["number"].item())
    if (locations.ndim != 2 or locations.shape[1] != 3 or not math.isfinite(number)
            or number != len(locations)
            or not np.isfinite(locations).all() or len(locations) > 512
            or (locations[:, 0] < 0).any() or (locations[:, 0] >= INPUT_WIDTH).any()
            or (locations[:, 1] < 0).any() or (locations[:, 1] >= INPUT_HEIGHT).any()
            or len(np.unique(locations[:, 2])) != len(locations)):
        raise ValueError(f"Malformed original full-frame annotations: {path}")
    expected = np.asarray(expected_points, dtype=np.float64).reshape(-1, 2)
    if expected.shape != locations[:, :2].shape or not np.array_equal(expected, locations[:, :2]):
        raise ValueError(f"Replay points are not the complete MAT coordinates: {path}")
    return locations[:, :2]


def source_identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def diagnose_frame(entries, group):
    first = entries[0]
    for entry in entries:
        for key in ("source_image", "source_sha256", "source_mat", "source_mat_sha256", "full_points_xy"):
            if entry[key] != first[key]:
                raise ValueError(f"Inconsistent frame metadata in source group {group}")
    image, mat = Path(first["source_image"]).resolve(strict=True), Path(first["source_mat"]).resolve(strict=True)
    before = source_identity(image), source_identity(mat)
    if not 0 < image.stat().st_size <= SOURCE_LIMIT or sha256(image) != first["source_sha256"]:
        raise ValueError(f"Image size or hash mismatch: {image}")
    if not 0 < mat.stat().st_size <= MAT_LIMIT:
        raise ValueError(f"MAT exceeds 16 MiB: {mat}")
    if sha256(mat) != first["source_mat_sha256"]:
        raise ValueError(f"MAT hash mismatch: {mat}")
    with Image.open(image) as picture:
        if picture.size != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError(f"Expected native 1920x1080 image: {image}")
    points = validate_mat(mat, first["full_points_xy"])
    occupied, coordinates, collisions, sigmas, adaptive, retained_adaptive = reconstruct_density(points)
    oracle = audit_reconstruct_density(points, height=HEIGHT, width=WIDTH)
    oracle_equal = bool(np.array_equal(adaptive, oracle))
    if not oracle_equal:
        raise ValueError(f"Adaptive map differs bitwise from audit_hdf5 oracle: {group}")
    fixed, retained_fixed = density_map(coordinates, np.full(len(coordinates), 2.0))
    quantile_keys = ("min", "p25", "median", "p75", "max")
    ratio_values = isolated_peak_ratio(sigmas)
    borders = {"top": int(occupied[0].sum()), "bottom": int(occupied[-1].sum()),
               "left": int(occupied[:, 0].sum()), "right": int(occupied[:, -1].sum())}
    frame = {"source_group": group, "source_image": str(image), "source_image_sha256": first["source_sha256"],
        "source_mat": str(mat), "source_mat_sha256": first["source_mat_sha256"],
        "native_dimensions": [INPUT_WIDTH, INPUT_HEIGHT], "raw_point_count": len(points),
        "occupied_cells": len(coordinates), "duplicate_cell_collisions": collisions,
        "occupied_border_cells": borders, "adaptive_sigma": quantiles(sigmas),
        "adaptive_to_fixed_isolated_peak_ratio": quantiles(ratio_values),
        "normalization": "1D discrete Gaussian normalized over full support radius round(4*sigma); cropped edge support is not renormalized",
        "adaptive_retained_impulse_mass": retained_adaptive, "fixed_retained_impulse_mass": retained_fixed,
        "adaptive_full_frame": map_metrics(adaptive), "fixed2_full_frame": map_metrics(fixed),
        "full_frame_spatial_cosine": cosine(adaptive, fixed),
        "audit_hdf5_bitwise_equal": oracle_equal}
    crops = []
    seen_boxes = set()
    for entry in entries:
        box = entry["box_xyxy"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Invalid crop box")
        x, y, right, bottom = box
        if (x % 4 or y % 4 or right % 4 or bottom % 4 or right <= x or bottom <= y
                or not 0 <= x < right <= INPUT_WIDTH or not 0 <= y < bottom <= INPUT_HEIGHT
                or (x, y, right, bottom) in seen_boxes):
            raise ValueError("Invalid or duplicate replay crop")
        seen_boxes.add((x, y, right, bottom))
        grid_box = (x//2, y//2, right//2, bottom//2)
        gx, gy, gr, gb = grid_box
        adaptive_crop, fixed_crop = adaptive[gy:gb, gx:gr], fixed[gy:gb, gx:gr]
        in_roi = ((points[:, 0] >= x) & (points[:, 0] < right)
                  & (points[:, 1] >= y) & (points[:, 1] < bottom))
        crops.append({"name": entry["name"], "box_xyxy": box,
            "raw_points_inside_box": int(in_roi.sum()),
            "adaptive": map_metrics(adaptive_crop), "fixed2": map_metrics(fixed_crop),
            "spatial_cosine": cosine(adaptive_crop, fixed_crop)})
    if len(crops) != 2:
        raise ValueError(f"Expected two replay crops for source group {group}")
    after = source_identity(image), source_identity(mat)
    if before != after:
        raise ValueError(f"Source changed during kernel diagnostic: {group}")
    frame["crops"] = crops
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    replay = args.replay.resolve(strict=True)
    if not 0 < replay.stat().st_size <= 1024*1024 or sha256(replay) != args.replay_sha256:
        parser.error("Replay manifest size or hash mismatch")
    data = json.loads(replay.read_text())
    entries = data.get("entries", [])
    if len(entries) != 16 or any(e.get("origin") != "original" or e.get("split") != "train" for e in entries):
        parser.error("Kernel comparison requires exactly sixteen original train crops")
    groups = {}
    for entry in entries:
        groups.setdefault(entry["source_group"], []).append(entry)
    if len(groups) != 8 or any(len(group) != 2 for group in groups.values()):
        parser.error("Expected eight unique full-frame sources with two replay crops each")
    frames = [diagnose_frame(group, name) for name, group in sorted(groups.items())]
    if sum(len(frame["crops"]) for frame in frames) != 16:
        raise ValueError("Expected exactly sixteen crop diagnostics")
    output = args.output_dir.resolve()
    source_paths = {Path(entry[key]).resolve() for entry in entries
                    for key in ("source_image", "source_mat")}
    dataset_root = Path(os.path.commonpath([str(path) for path in source_paths]))
    if (output.is_relative_to(replay.parent) or replay.is_relative_to(output)
            or output.is_relative_to(dataset_root) or dataset_root.is_relative_to(output)
            or any(output.is_relative_to(path.parent) or path.is_relative_to(output)
                   for path in source_paths)):
        parser.error("Output must be separate from replay and original dataset sources")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "complete", "replay": str(replay), "replay_sha256": args.replay_sha256,
        "script_sha256": sha256(__file__), "method": "CPU original full annotations; no model inference",
        "audit_hdf5_script": str(Path(__file__).with_name("audit_hdf5.py").resolve()),
        "audit_hdf5_sha256": sha256(Path(__file__).with_name("audit_hdf5.py")),
        "adaptive_oracle": "np.array_equal with audit_hdf5.reconstruct_density; all eight source frames must match",
        "grid": [WIDTH, HEIGHT], "coordinate_rule": "occupied cell = (int(x/2), int(y/2)); duplicate cells are boolean",
        "adaptive_sigma_rule": "0.1 * sum of distances to the 3 nearest other occupied cells; historical one-cell fallback only",
        "fixed_sigma": 2.0, "kernel_support": "radius round(4*sigma)",
        "edge_rule": "normalize 1D kernel on full support, clip at full-frame border without renormalization",
        "interpretation": "Spatial implementation diagnostic only; not a quality or accuracy measure.",
        "frames": frames, "seconds": time.monotonic()-started}
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
