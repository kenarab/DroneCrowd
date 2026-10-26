"""Compare frozen crowd counters on identical native frames with bounded streaming."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import re
import time
import tempfile

import numpy as np
import torch

from evaluate_sr import assemble as sr_assemble, read_rgb, CHECKPOINT_SHA256
from sample_video import memory_limits, sha256

FROZEN_INPUT_SHA256 = "f9737bb74cce72913959475808957f059ea62e09ca61e6f31651c01ef0e47745"
FROZEN_CHECKPOINTS = {
    "sr": CHECKPOINT_SHA256,
    "clip-ebc": "da40b664a0a5a57bc33235e7334403e5000271e0741134e301ea706d754bf7df",
    "mpcount": "4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0",
}


def validated_inputs(path, expected_hash, full_video_block=False):
    if path.stat().st_size > 256 * 1024:
        raise ValueError("Manifest too large")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or sha256(path) != expected_hash:
        raise ValueError("Input manifest hash mismatch")
    manifest = json.loads(path.read_text())
    frames = manifest["frames"]
    maximum = 16 if full_video_block else 3
    if not isinstance(frames, list) or not 1 <= len(frames) <= maximum:
        raise ValueError(f"This process allows one to {maximum} frames")
    indices = set()
    for frame in frames:
        if (type(frame["width"]) is not int or type(frame["height"]) is not int
                or not 224 <= frame["width"] <= 3840 or not 224 <= frame["height"] <= 2160
                or frame["width"] % 16 or frame["height"] % 16
                or not re.fullmatch(r"frame-\d{3}\.ppm", frame["rgb_file"])
                or not re.fullmatch(r"[0-9a-f]{64}", frame["rgb_sha256"])
                or type(frame["index"]) is not int or frame["index"] in indices
                or type(frame["pts_seconds"]) not in (int, float) or not math.isfinite(frame["pts_seconds"])):
            raise ValueError("Invalid bounded frame geometry or identity")
        indices.add(frame["index"])
    if full_video_block:
        for index, frame in enumerate(frames):
            if (frame["index"] != index or type(frame.get("source_ordinal")) is not int
                    or frame["source_ordinal"] < 0 or type(frame.get("pts")) is not int
                    or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", str(frame.get("time_base", "")))
                    or frame["pts_seconds"] != float(frame["pts"] * Fraction(frame["time_base"]))
                    or (index and frame["source_ordinal"] != frames[index-1]["source_ordinal"]+1)):
                raise ValueError("Full video block requires consecutive ordinals and original PTS")
    return manifest


def write_report_atomic(path, report):
    """Keep staging in the authorized temporary, preserving the last valid report."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix="dronecrowd-report-", suffix=".json",
                                         dir=os.environ["TMPDIR"], delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clip_windows(width, height):
    if min(width, height) < 224 or width % 8 or height % 8:
        raise ValueError("CLIP streaming requires dimensions >=224 aligned to eight")
    rows = (height - 224 + 223) // 224 + 1
    columns = (width - 224 + 223) // 224 + 1
    for row in range(rows):
        for column in range(columns):
            top, left = min(row * 224, height - 224), min(column * 224, width - 224)
            yield left, top, left + 224, top + 224


def streaming_clip(rgb, predict, deadline):
    height, width = rgb.shape[:2]
    density = np.zeros((height // 8, width // 8), np.float32)
    coverage = np.zeros_like(density)
    count = 0
    for left, top, right, bottom in clip_windows(width, height):
        if time.monotonic() >= deadline:
            raise TimeoutError("CLIP frame deadline exceeded")
        tile = predict(rgb[top:bottom, left:right])
        if tile.shape != (28, 28) or not np.isfinite(tile).all() or np.any(tile < 0):
            raise ValueError("Invalid CLIP density")
        density[top // 8:bottom // 8, left // 8:right // 8] += tile
        coverage[top // 8:bottom // 8, left // 8:right // 8] += 1
        count += 1
    if not np.all(coverage > 0):
        raise ValueError("Uncovered CLIP output cells")
    return density / coverage, count


def streaming_mpcount(rgb, predict, deadline):
    height, width = rgb.shape[:2]
    if width % 16 or height % 16:
        raise ValueError("MPCount input must align to sixteen")
    density = np.empty((height, width), np.float32)
    count = 0
    # Upstream inference explicitly permits reducing patch_size from 3584.
    # Preserve its disjoint patch assembly, with a bounded patch_size of 512.
    for top in range(0, height, 512):
        for left in range(0, width, 512):
            if time.monotonic() >= deadline:
                raise TimeoutError("MPCount frame deadline exceeded")
            right, bottom = min(width, left + 512), min(height, top + 512)
            tile = predict(rgb[top:bottom, left:right])
            if tile.shape != (bottom - top, right - left) or not np.isfinite(tile).all():
                raise ValueError("Invalid MPCount density")
            density[top:bottom, left:right] = tile / np.float32(1000)
            count += 1
    return density, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sr", "clip-ebc", "mpcount"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", default=FROZEN_INPUT_SHA256)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--clip-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-video-block", action="store_true")
    parser.add_argument("--rgb-root", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 115
    torch.set_num_threads(1)
    if args.checkpoint_sha256 != FROZEN_CHECKPOINTS[args.model]:
        raise ValueError("This experiment requires the frozen model checkpoint hashes")
    manifest = validated_inputs(args.input, args.expected_input_sha256, args.full_video_block)
    if args.rgb_root is not None and not args.full_video_block:
        raise ValueError("Separate RGB cache requires an explicit full video block")
    rgb_root = args.rgb_root if args.rgb_root is not None else args.input.parent
    frames = manifest["frames"]
    output = args.output_dir.resolve()
    if args.input.resolve().is_relative_to(output):
        raise ValueError("Output must be separate from inputs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "model": args.model, "input_sha256": sha256(args.input),
              "source_sha256": manifest["source_sha256"], "checkpoint_sha256": args.checkpoint_sha256,
              "script_sha256": sha256(__file__), "memory_limits": memory_limits(),
              "device": "cuda", "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "interpretation": "Same native images; model-specific contexts. No GT, accuracy or attendance claim. Never sum frames.",
              "results": []}

    def save_report():
        write_report_atomic(output / "report.json", report)

    save_report()
    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        if args.model == "sr":
            from evaluate_sr import load_model
            if args.checkpoint_sha256 != CHECKPOINT_SHA256:
                raise ValueError("Expected frozen SR checkpoint")
            model, metadata = load_model(args.checkpoint)
            assembler = sr_assemble
            mean_values, std_values = [.485, .456, .406], [.229, .224, .225]
            geometry = {"stride": 8, "tile_core": [256, 256], "halo": 160, "aggregation": "disjoint cores; sum directly"}
        elif args.model == "clip-ebc":
            from clip_ebc_model import load_model, prepare_text_cache
            if args.clip_source is None:
                raise ValueError("CLIP source required")
            model, metadata = load_model(args.checkpoint, args.checkpoint_sha256, args.clip_source, "resnet50")
            assembler = streaming_clip
            mean_values, std_values = [.485, .456, .406], [.229, .224, .225]
            geometry = {"stride": 8, "window": 224, "window_step": 224, "aggregation": "average overlapping cells; sum directly"}
        else:
            from mpcount_model import load_model
            model, metadata = load_model(args.checkpoint, args.checkpoint_sha256)
            assembler = streaming_mpcount
            mean_values, std_values = [.5] * 3, [.5] * 3
            geometry = {"stride": 1, "patch_size": 512, "aggregation": "disjoint patches; density divided by1000"}
        model.to("cuda")
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        mean = torch.tensor(mean_values, device="cuda")[None, :, None, None]
        std = torch.tensor(std_values, device="cuda")[None, :, None, None]

        def predict(crop):
            tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32) / 255
            result = model((tensor - mean) / std)
            if args.model == "mpcount":
                result = result[0]
            return result[0, 0].cpu().numpy().copy()

        with torch.inference_mode():
            if args.model == "clip-ebc":
                metadata["trained_text_features_sha256"] = prepare_text_cache(model, "cuda")
            size = 224 if args.model == "clip-ebc" else 64
            synthetic = predict(np.zeros((size, size, 3), np.uint8))
            if not np.isfinite(synthetic).all():
                raise ValueError("Synthetic check failed")
            report.update(checkpoint_metadata=metadata, geometry=geometry, synthetic_check_passed=True,
                          model_code_sha256=sha256(Path(__file__).with_name({"sr": "sr_model.py", "clip-ebc": "clip_ebc_model.py", "mpcount": "mpcount_model.py"}[args.model])))
            torch.cuda.synchronize()
            report["load_and_warmup_seconds"] = time.monotonic() - started
            torch.cuda.reset_peak_memory_stats()
            for index, frame in enumerate(frames):
                rgb = read_rgb(rgb_root / frame["rgb_file"], frame["width"], frame["height"], frame["rgb_sha256"])
                torch.cuda.synchronize()
                tick = time.monotonic()
                density, tile_count = assembler(rgb, predict, deadline)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                file = output / f"density-{index:03d}.npy"
                np.save(file, density, allow_pickle=False)
                report["results"].append({"index": index, "pts_seconds": frame["pts_seconds"],
                    "rgb_sha256": frame["rgb_sha256"], "width": frame["width"], "height": frame["height"],
                    "count": float(density.sum(dtype=np.float64)), "negative_mass": float(np.minimum(density, 0).sum(dtype=np.float64)),
                    "tile_count": tile_count, "inference_seconds": elapsed,
                    "density_file": file.name, "density_sha256": sha256(file)})
                if args.full_video_block:
                    report["results"][-1].update({key: frame[key] for key in
                        ("source_ordinal", "pts", "time_base")})
                save_report()
                # Release the previous frame/map before reading the next RGB.
                del rgb, density
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(total_seconds=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        save_report()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
