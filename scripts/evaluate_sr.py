"""Bounded native SR inference on frozen local controls or one video frame.

Per-image counts, not unique participants. Uses official DroneCrowd weights;
no training, automatic downloads, upscaling, or arbitrary pickle loading.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from sample_video import sha256, memory_limits
from sr_model import SRDensity

CHECKPOINT_SHA256 = "22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb"
UPSTREAM_COMMIT = "6c5f66c823dd86dcb1329f1b007ba369e622036b"


def windows(width, height):
    if (type(width) is not int or type(height) is not int or width % 16 or height % 16
            or not 32 <= width <= 3840 or not 32 <= height <= 2160):
        raise ValueError("Expected multiples of 16 within 3840 by 2160")
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            right, bottom = min(width, x + 256), min(height, y + 256)
            yield ((x, y, right, bottom),
                   (max(0, x - 160), max(0, y - 160), min(width, right + 160), min(height, bottom + 160)))


def assemble(rgb, predict, deadline):
    height, width, channels = rgb.shape
    if rgb.dtype != np.uint8 or channels != 3:
        raise ValueError("Expected RGB uint8")
    tile_windows = list(windows(width, height))
    result = np.empty((height // 8, width // 8), dtype=np.float32)
    for core, context in tile_windows:
        if time.monotonic() >= deadline:
            raise TimeoutError("Inference deadline exceeded")
        x, y, right, bottom = core
        left, top, end_x, end_y = context
        density = predict(rgb[top:end_y, left:end_x])
        if (density.dtype != np.float32 or density.shape != ((end_y - top) // 8, (end_x - left) // 8)
                or not np.isfinite(density).all() or np.any(density < 0)):
            raise ValueError("Unexpected SR density")
        result[y // 8:bottom // 8, x // 8:right // 8] = density[
            (y - top) // 8:(bottom - top) // 8, (x - left) // 8:(right - left) // 8]
    return result, len(tile_windows)


def regional_sum(density, box):
    x, y, right, bottom = box
    if (any(type(v) is not int for v in box) or any(v % 8 for v in box)
            or not 0 <= x < right <= density.shape[1] * 8
            or not 0 <= y < bottom <= density.shape[0] * 8):
        raise ValueError("ROI must align to stride eight and lie inside density")
    return float(density[y // 8:bottom // 8, x // 8:right // 8].sum(dtype=np.float64))


def load_model(path):
    if path.stat().st_size != 86123541 or sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Official SR checkpoint size/hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = SRDensity()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint keys differ from SR VGG19-BN architecture")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid SR tensor: {name}")
    model.load_state_dict(state, strict=True)
    return model.eval(), {"tensor_count": len(state), "epoch": checkpoint.get("epoch")}


def read_rgb(path, width, height, digest):
    if sha256(path) != digest:
        raise ValueError("RGB hash mismatch")
    with path.open("rb") as handle:
        if (handle.readline() != b"P6\n" or handle.readline() != f"{width} {height}\n".encode()
                or handle.readline() != b"255\n"):
            raise ValueError("Unexpected PPM header")
        payload = handle.read(width * height * 3 + 1)
    if len(payload) != width * height * 3:
        raise ValueError("PPM size mismatch")
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)


def inputs(path, frame_index):
    if path.stat().st_size > 256 * 1024:
        raise ValueError("Manifest exceeds 256 KiB")
    manifest = json.loads(path.read_text())
    if "entries" in manifest:
        entries = manifest["entries"]
    else:
        entries = [dict(f, name=f"video-frame-{f['index']}") for f in manifest["frames"]
                   if f["index"] == frame_index]
    if not 1 <= len(entries) <= 20:
        raise ValueError("Expected one to twenty inputs")
    for entry in entries:
        width, height = entry["width"], entry["height"]
        if not 32 <= width <= 3840 or not 32 <= height <= 2160:
            raise ValueError("Input exceeds bounded native geometry")
        if Path(entry["rgb_file"]).name != entry["rgb_file"]:
            raise ValueError("RGB must be in manifest directory")
        effective = (width // 16 * 16, height // 16 * 16)
        list(windows(*effective))
        box = entry.get("roi_context_xyxy", [0, 0, *effective])
        regional_sum(np.zeros((effective[1] // 8, effective[0] // 8), np.float32), box)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 115
    torch.set_num_threads(1)
    manifest_path = args.input.resolve(strict=True)
    selected = inputs(manifest_path, args.frame_index)
    output = args.output_dir.resolve()
    if output == manifest_path.parent or manifest_path.is_relative_to(output):
        raise ValueError("Output must be separate from inputs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "model": "SR VGG19-BN DroneCrowd", "device": args.device,
              "upstream_commit": UPSTREAM_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA256,
              "input_manifest": str(manifest_path), "input_sha256": sha256(manifest_path),
              "script_sha256": sha256(__file__), "architecture_sha256": sha256(Path(__file__).with_name("sr_model.py")),
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "memory_limits": memory_limits(), "tile_core": [256, 256], "halo": 160,
              "input_transform": "native RGB; crop right/bottom to multiples of 16; ImageNet normalization",
              "integration": "sum density cells directly; no STNNet factor100",
              "interpretation": "Exploratory per-image counts. Tiling approximates full forward; no accuracy or unique-person claim.",
              "results": []}
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        model, metadata = load_model(args.checkpoint)
        model = model.to(args.device)
        report["checkpoint_metadata"] = metadata
        mean = torch.tensor([.485, .456, .406], device=args.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=args.device)[None, :, None, None]

        def sync():
            if args.device == "cuda":
                torch.cuda.synchronize()

        def predict(crop):
            tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).unsqueeze(0).to(args.device, torch.float32) / 255
            return model((tensor - mean) / std)[0, 0].cpu().numpy().copy()

        with torch.inference_mode():
            synthetic = predict(np.zeros((64, 64, 3), np.uint8))
            if synthetic.shape != (8, 8) or not np.isfinite(synthetic).all():
                raise ValueError("Synthetic execution failed")
            report["synthetic_check_passed"] = True
            sync()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            report["load_and_warmup_seconds"] = time.monotonic() - started
            for index, entry in enumerate(selected):
                width, height = entry["width"], entry["height"]
                rgb = read_rgb(manifest_path.parent / entry["rgb_file"], width, height, entry["rgb_sha256"])
                effective_width, effective_height = width // 16 * 16, height // 16 * 16
                sync()
                inference_started = time.monotonic()
                density, tile_count = assemble(rgb[:effective_height, :effective_width], predict, deadline)
                sync()
                seconds = time.monotonic() - inference_started
                box = entry.get("roi_context_xyxy", [0, 0, effective_width, effective_height])
                density_path = output / f"density-{index:03d}.npy"
                np.save(density_path, density, allow_pickle=False)
                report["results"].append({"name": entry["name"], "rgb_file": entry["rgb_file"],
                    "rgb_sha256": entry["rgb_sha256"], "input_size": [width, height],
                    "effective_size": [effective_width, effective_height], "roi_context_xyxy": box,
                    "count": regional_sum(density, box), "context_count": float(density.sum(dtype=np.float64)),
                    "tile_count": tile_count, "inference_seconds": seconds,
                    "pts_seconds": entry.get("pts_seconds"), "density_file": density_path.name,
                    "density_sha256": sha256(density_path)})
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(total_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" and torch.cuda.is_available() else 0)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
