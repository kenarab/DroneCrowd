"""Bounded STNNet density inference on a verified RGB video sample.

Requires a complete trained checkpoint with explicit provenance and SHA-256.
Never downloads weights, trains, or imports the historical benchmark loaders.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import resource
import re
import stat
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "STNNet"))
from stnmodel import STNNet
from sample_video import memory_limits, sha256


# Inspected archive/data.pkl: best_prec1 is a NumPy float64 scalar. Only these
# exact author-supplied checkpoints need the historical NumPy module alias.
NUMPY_METADATA_CHECKPOINTS = {
    "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d",
    "88168a85079b4931b86ea1145d501ad5e73d3d27e903fedf4c049d62514a8d5d",
}


def restricted_checkpoint(path, digest):
    if digest not in NUMPY_METADATA_CHECKPOINTS:
        return torch.load(path, map_location="cpu", weights_only=True)
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
               np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def normalized_rgb(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB uint8 H×W×3")
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


def quadrants(rgb):
    height, width = rgb.shape[:2]
    if height % 4 or width % 4 or height < 32 or width < 32:
        raise ValueError("Frame dimensions must be multiples of four, at least 32")
    if width > 960 or height > 540:
        raise ValueError("Initial inference is limited to 960×540")
    half_height, half_width = height // 2, width // 2
    return [rgb[:half_height, :half_width], rgb[:half_height, half_width:],
            rgb[half_height:, :half_width], rgb[half_height:, half_width:]]


def count_from_maps(maps):
    if len(maps) != 4 or any(not np.isfinite(array).all() for array in maps):
        raise ValueError("Four finite density maps are required")
    # Match mytest.py: clamp the SUM of each crop, not each pixel.
    with np.errstate(over="ignore", invalid="ignore"):
        sums = [float(array.sum(dtype=np.float32)) for array in maps]
    if not all(math.isfinite(value) for value in sums):
        raise ValueError("Non-finite density sums")
    return sum(max(0.0, value) for value in sums) / 100.0, sums


def load_density_checkpoint(path, expected_hash, *, use_loc=False):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 512 * 2**20:
        raise ValueError("Checkpoint missing or exceeds 512 MiB")
    if len(expected_hash) != 64 or sha256(path) != expected_hash.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")
    checkpoint = restricted_checkpoint(path, expected_hash.lower())
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Expected a complete checkpoint with state_dict")
    state = {}
    for name, value in checkpoint["state_dict"].items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("State dict must contain named tensors")
        normalized = name.removeprefix("module.")
        if normalized in state:
            raise ValueError("Checkpoint prefix normalization collision")
        state[normalized] = value
    model = STNNet(use_loc=use_loc, use_trk=False, pretrained_backbone=False)
    wanted = model.state_dict()
    if set(state) != set(wanted):
        raise ValueError(f"Density checkpoint keys differ: missing={sorted(set(wanted)-set(state))}; extra={sorted(set(state)-set(wanted))}")
    for name, tensor in state.items():
        if tensor.shape != wanted[name].shape or tensor.dtype != wanted[name].dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint shape/dtype/finiteness mismatch: {name}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def read_rgb(path, width, height, *, max_width=960, max_height=540):
    if (type(width) is not int or type(height) is not int or not 32 <= width <= max_width
            or not 32 <= height <= max_height or width % 4 or height % 4):
        raise ValueError("RGB dimensions exceed the initial inference budget")
    with Path(path).open("rb") as handle:
        if handle.readline() != b"P6\n" or handle.readline() != f"{width} {height}\n".encode() or handle.readline() != b"255\n":
            raise ValueError("Unexpected sampled RGB format")
        data = handle.read(width * height * 3 + 1)
    if len(data) != width * height * 3:
        raise ValueError("RGB payload size differs from its manifest")
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def validate_frames(frames, directory, *, max_width=960, max_height=540, max_frames=30, require_timestamps=True):
    if not isinstance(frames, list) or not 1 <= len(frames) <= max_frames:
        raise ValueError("Input must have one to thirty frames")
    indices = set()
    for frame in frames:
        width, height, index = frame["width"], frame["height"], frame["index"]
        if (type(width) is not int or type(height) is not int or not 32 <= width <= max_width
                or not 32 <= height <= max_height or width % 4 or height % 4):
            raise ValueError("RGB dimensions exceed the initial inference budget")
        if type(index) is not int or not 0 <= index < 30 or index in indices:
            raise ValueError("Frame indices must be distinct integers in [0, 30)")
        indices.add(index)
        if require_timestamps and (type(frame["pts_seconds"]) not in (float, int) or not math.isfinite(frame["pts_seconds"])):
            raise ValueError("Invalid frame timestamp")
        if not re.fullmatch(r"frame-\d{3}\.ppm", frame["rgb_file"]):
            raise ValueError("Unsafe RGB basename")
        if not re.fullmatch(r"[0-9a-f]{64}", frame["rgb_sha256"]):
            raise ValueError("Invalid RGB hash")
        path = directory / frame["rgb_file"]
        info = path.lstat()
        expected_size = len(f"P6\n{width} {height}\n255\n".encode()) + width * height * 3
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or path.resolve(strict=True) != path or info.st_size != expected_size):
            raise ValueError("Expected an unchanged bounded regular RGB file without links")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input.json from sample_video.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 120:
        parser.error("Maximum duration is 120 seconds")
    started = time.monotonic()
    torch.set_num_threads(1)
    output, input_path = args.output_dir.resolve(), args.input.resolve(strict=True)
    if input_path.stat().st_size > 256 * 1024:
        parser.error("Input manifest exceeds 256 KiB")
    inputs = json.loads(input_path.read_text())
    if output.is_relative_to(Path(inputs["source"]).parent) or input_path.is_relative_to(output):
        parser.error("Choose an output separate from originals and sampled inputs")
    validate_frames(inputs["frames"], input_path.parent)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "input_manifest": str(input_path), "input_sha256": sha256(input_path),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
              "checkpoint_source": args.checkpoint_source, "device": args.device, "density_factor": 100,
              "script_sha256": sha256(__file__), "model_code_sha256": sha256(Path(__file__).resolve().parents[1] / "STNNet/stnmodel.py"),
              "torch_version": torch.__version__, "numpy_version": np.__version__, "frames": [],
              "memory_limits": memory_limits(),
              "interpretation": "Per-frame exploratory estimates; no accuracy measurement or unique-person total."}
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        model = load_density_checkpoint(args.checkpoint, args.checkpoint_sha256).to(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for frame in inputs["frames"]:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Inference deadline exhausted")
                relative = Path(frame["rgb_file"])
                if relative.name != str(relative):
                    raise ValueError("Sample frame path must be a basename")
                path = input_path.parent / relative
                if sha256(path) != frame["rgb_sha256"]:
                    raise ValueError("Sampled frame SHA-256 mismatch")
                rgb = read_rgb(path, frame["width"], frame["height"])
                maps = []
                for crop in quadrants(rgb):
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("Inference deadline exhausted")
                    tensor = normalized_rgb(crop).to(args.device)
                    density = model([tensor])[0][0][0, 0].cpu().numpy().copy()
                    if density.shape != (crop.shape[0] // 2, crop.shape[1] // 2):
                        raise ValueError("Unexpected density output geometry")
                    maps.append(density)
                count, sums = count_from_maps(maps)
                assembled = np.block([[maps[0], maps[1]], [maps[2], maps[3]]])
                name = f"density-{frame['index']:03d}.npy"
                np.save(output / name, assembled, allow_pickle=False)
                report["frames"].append({"index": frame["index"], "pts_seconds": frame["pts_seconds"],
                                         "count": count, "crop_density_sums": sums, "density_file": name})
        with (output / "counts.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "pts_seconds", "count"])
            writer.writeheader()
            writer.writerows({k: frame[k] for k in writer.fieldnames} for frame in report["frames"])
        report.update(status="complete", seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
