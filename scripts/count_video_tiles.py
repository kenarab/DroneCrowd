"""One verified frame at native resolution or Lanczos scale, with bounded tiles."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import torch

from count_video import count_from_maps, load_density_checkpoint, normalized_rgb, read_rgb, validate_frames
from sample_video import memory_limits, sha256


def tile_windows(width, height):
    if (type(width) is not int or type(height) is not int or not 32 <= width <= 3840
            or not 32 <= height <= 2160 or width % 4 or height % 4):
        raise ValueError("Tiled geometry requires multiples of four up to 3840×2160")
    for y in range(0, height, 128):
        for x in range(0, width, 320):
            right, bottom = min(x + 320, width), min(y + 128, height)
            yield ((x, y, right, bottom),
                   (max(0, x - 96), max(0, y - 96), min(width, right + 96), min(height, bottom + 96)))


def assemble_density(rgb, predict, deadline):
    height, width = rgb.shape[:2]
    windows = list(tile_windows(width, height))
    if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
        raise ValueError("Expected RGB uint8")
    assembled = np.empty((height // 2, width // 2), dtype=np.float32)
    for core, context in windows:
        if time.monotonic() > deadline:
            raise TimeoutError("Tiled inference deadline exhausted")
        x, y, right, bottom = core
        left_context, top_context, right_context, bottom_context = context
        density = predict(rgb[top_context:bottom_context, left_context:right_context])
        if (density.dtype != np.float32
                or density.shape != ((bottom_context - top_context) // 2, (right_context - left_context) // 2)
                or not np.isfinite(density).all()):
            raise ValueError("Unexpected or non-finite tile density")
        assembled[y // 2:bottom // 2, x // 2:right // 2] = density[
            (y - top_context) // 2:(bottom - top_context) // 2,
            (x - left_context) // 2:(right - left_context) // 2]
    return assembled, len(windows)


def quadrant_count(density):
    half_height, half_width = density.shape[0] // 2, density.shape[1] // 2
    return count_from_maps([density[:half_height, :half_width], density[:half_height, half_width:],
                            density[half_height:, :half_width], density[half_height:, half_width:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 120
    torch.set_num_threads(1)
    path = args.input.resolve(strict=True)
    if path.stat().st_size > 256 * 1024:
        parser.error("Input manifest exceeds 256 KiB")
    inputs = json.loads(path.read_text())
    frames = inputs["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 30:
        parser.error("Input must contain one to thirty frames")
    selected = [f for f in frames if f["index"] == args.frame_index]
    validate_frames(selected, path.parent, max_width=3840, max_height=2160, max_frames=1)
    frame = selected[0]
    width, height = frame["width"] * args.scale, frame["height"] * args.scale
    # Validate geometry and output bounds before hashing/reading/upscaling pixels.
    tiles = list(tile_windows(width, height))
    output = args.output_dir.resolve()
    if output.is_relative_to(Path(inputs["source"]).parent) or path.is_relative_to(output):
        parser.error("Output must be separate from source videos and sampled inputs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "input_manifest": str(path), "input_sha256": sha256(path),
              "source_sha256": inputs["source_sha256"], "frame_index": args.frame_index,
              "pts_seconds": frame["pts_seconds"], "input_width": frame["width"], "input_height": frame["height"],
              "width": width, "height": height, "scale": args.scale,
              "transform": "identity" if args.scale == 1 else "FFmpeg Lanczos interpolation",
              "tile_core": [320, 128], "halo": 96, "tile_count": len(tiles),
              "aggregation": "assemble disjoint cores; clamp four global quadrant sums; divide by 100",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
              "checkpoint_source": args.checkpoint_source, "script_sha256": sha256(__file__),
              "checkpoint_loader_sha256": sha256(Path(__file__).with_name("count_video.py")),
              "model_code_sha256": sha256(Path(__file__).resolve().parents[1] / "STNNet/stnmodel.py"),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "memory_limits": memory_limits(), "device": "cuda",
              "interpretation": "Per-frame exploratory output; interpolation adds no observed detail; accuracy unmeasured."}
    try:
        rgb_path = path.parent / frame["rgb_file"]
        if sha256(rgb_path) != frame["rgb_sha256"]:
            raise ValueError("Input RGB SHA-256 mismatch")
        rgb = read_rgb(rgb_path, frame["width"], frame["height"], max_width=3840, max_height=2160)
        if args.scale > 1:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise ValueError("FFmpeg is required for Lanczos")
            result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
                                     "-filter_threads", "1", "-i", str(rgb_path),
                                     "-vf", f"scale={width}:{height}:flags=lanczos", "-frames:v", "1",
                                     "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                                    check=True, capture_output=True, timeout=min(30, max(.01, deadline-time.monotonic())))
            if len(result.stdout) != width * height * 3:
                raise ValueError("Unexpected upscaled RGB size")
            rgb = np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)
            report["ffmpeg"] = ffmpeg
        report["transformed_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        model = load_density_checkpoint(args.checkpoint, args.checkpoint_sha256).to("cuda")
        torch.cuda.reset_peak_memory_stats()

        def predict(crop):
            return model([normalized_rgb(crop).to("cuda")])[0][0][0, 0].cpu().numpy().copy()

        with torch.inference_mode():
            density, tile_count = assemble_density(rgb, predict, deadline)
        count, sums = quadrant_count(density)
        np.save(output / "density.npy", density, allow_pickle=False)
        report.update(status="complete", count=count, quadrant_density_sums=sums, tile_count=tile_count,
                      density_file="density.npy", density_sha256=sha256(output / "density.npy"),
                      seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
