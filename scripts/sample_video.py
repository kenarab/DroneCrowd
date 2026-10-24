"""Sample local video into RGB frames and a private reproducible manifest.

Uses FFmpeg/FFprobe without installing packages. Never modifies the source.
This command does not run a crowd model or produce estimated counts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def memory_limits():
    """Record the actual kernel limit of this process, not just a launch flag."""
    entry = next(line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    directory = Path("/sys/fs/cgroup") / entry.lstrip("/")
    return {"cgroup": entry, **{name: (directory / name).read_text().strip()
                                for name in ("memory.high", "memory.max", "memory.swap.max")}}


def geometry(width, height, target_width):
    if target_width < 32 or target_width > 1920 or target_width % 4:
        raise ValueError("Target width must be a multiple of four between 32 and 1920")
    target_height = max(32, round(height * target_width / width / 4) * 4)
    if target_height > 1080:
        raise ValueError("Target geometry exceeds the bounded pixel budget")
    return target_width, target_height


def output_budget(width, height, frames):
    # PNG can be nearly as large as incompressible RGB, plus container overhead.
    planned = frames * (2 * width * height * 3 + 65536)
    if planned > 120 * 2**20:
        raise ValueError("Planned RGB/preview output exceeds 120 MiB")
    return planned


def sample(video, output, width=960, fractions=(0.1, 0.5, 0.9), max_seconds=120):
    if not 1 <= len(fractions) <= 30 or not all(math.isfinite(f) and 0 <= f < 1 for f in fractions):
        raise ValueError("Choose one to thirty fractions in [0, 1)")
    if not 0 < max_seconds <= 120:
        raise ValueError("Wall time must be bounded to at most 120 seconds")
    video = Path(video).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(video.parent) or video.is_relative_to(output):
        raise ValueError("Output must be separate from the original video directory")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("FFmpeg and FFprobe are required")
    started = time.monotonic()

    def run(command, **kwargs):
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Sampling wall time exhausted")
        return subprocess.run(command, check=True, capture_output=True, timeout=min(30, remaining), **kwargs)

    before = video.stat()
    probe = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,width,height,r_frame_rate,time_base,start_time,nb_frames:stream_side_data=rotation:format=duration,start_time,size",
                 "-of", "json", str(video)], text=True)
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    duration = float(metadata["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    native_width, native_height = int(stream["width"]), int(stream["height"])
    rotation = next((float(s["rotation"]) for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    if round(rotation) % 180:
        native_width, native_height = native_height, native_width
    target_width, target_height = geometry(native_width, native_height, width)
    source_hash = sha256(video)
    output_budget(target_width, target_height, len(fractions))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for index, fraction in enumerate(fractions):
            requested = duration * fraction
            decoded = run([ffmpeg, "-hide_banner", "-loglevel", "info", "-threads", "1",
                           "-filter_threads", "1", "-copyts", "-ss", f"{requested:.9f}",
                           "-i", str(video), "-map", "0:v:0", "-an", "-sn",
                           "-vf", f"scale={target_width}:{target_height},showinfo",
                           "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
            rgb = decoded.stdout
            if len(rgb) != target_width * target_height * 3:
                raise ValueError("FFmpeg did not return exactly one RGB frame")
            timestamps = re.findall(r"\bn:\s*0\s+pts:.*?pts_time:([-+0-9.eE]+)", decoded.stderr.decode("utf-8", "replace"))
            if not timestamps:
                raise ValueError("Decoded frame lacks an actual PTS timestamp")
            pts = float(timestamps[0])
            if not math.isfinite(pts):
                raise ValueError("Invalid decoded timestamp")
            ppm = output / f"frame-{index:03d}.ppm"
            ppm.write_bytes(f"P6\n{target_width} {target_height}\n255\n".encode() + rgb)
            png = output / f"frame-{index:03d}.png"
            run([ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
                 "-i", str(ppm), "-frames:v", "1", "-threads", "1", "-n", str(png)])
            records.append({"index": index, "requested_seconds": requested, "pts_seconds": pts,
                            "rgb_file": ppm.name, "preview_file": png.name,
                            "rgb_sha256": sha256(ppm), "width": target_width, "height": target_height})
        after = video.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Original video changed during sampling")
        report = {"source": str(video), "source_sha256": source_hash, "source_bytes": before.st_size,
                  "source_mtime_ns": before.st_mtime_ns, "metadata": metadata, "frames": records,
                  "sampling_seconds": time.monotonic() - started, "model_inference": False,
                  "ffmpeg": ffmpeg, "ffprobe": ffprobe, "script_sha256": sha256(__file__),
                  "memory_limits": memory_limits()}
        (output / "input.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as error:
        (output / "sampling-failure.json").write_text(json.dumps({"error": str(error), "model_inference": False}) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    args = parser.parse_args()
    report = sample(args.video, args.output_dir, args.width, args.fractions)
    print(json.dumps({"source": report["source"], "frames": len(report["frames"]),
                      "actual_timestamps": [f["pts_seconds"] for f in report["frames"]],
                      "model_inference": False, "manifest": str(args.output_dir / "input.json")}))


if __name__ == "__main__":
    main()
