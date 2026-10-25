"""Bounded annotated-subset validation with historical crops or halo tiles."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from count_video import count_from_maps, load_density_checkpoint, normalized_rgb, read_rgb, validate_frames
from count_video_tiles import assemble_density, quadrant_count
from sample_video import memory_limits, sha256


def error_metrics(rows):
    errors = np.array([row["count"] - row["gt_count"] for row in rows], dtype=np.float64)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Finite nonempty evaluation required")
    return {"images": len(errors), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())), "bias": float(errors.mean()),
            "gt_total": sum(row["gt_count"] for row in rows), "prediction_total": sum(row["count"] for row in rows)}


def historical_density(rgb, model, device="cuda"):
    height, width = rgb.shape[:2]
    halves = [rgb[:height//2, :width//2], rgb[:height//2, width//2:],
              rgb[height//2:, :width//2], rgb[height//2:, width//2:]]
    maps = [[], [], []]
    for crop in halves:
        outputs = model([normalized_rgb(crop).to(device)])
        for head in range(3):
            maps[head].append(outputs[head][0][0, 0].cpu().numpy().copy())
    counts = [count_from_maps(group) for group in maps]
    return np.block([[maps[0][0], maps[0][1]], [maps[0][2], maps[0][3]]]), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=["quadrants", "tiles"], required=True)
    parser.add_argument("--variant", choices=["density", "localization"], default="density")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 120
    torch.set_num_threads(1)
    cohort_path = args.cohort.resolve(strict=True)
    if cohort_path.stat().st_size > 2**20 or sha256(cohort_path) != args.cohort_sha256:
        parser.error("Cohort size or SHA-256 mismatch")
    cohort = json.loads(cohort_path.read_text())
    if (not 1 <= len(args.indices) <= 30 or len(set(args.indices)) != len(args.indices)
            or any(not 0 <= i < len(cohort["entries"]) for i in args.indices)):
        parser.error("Choose distinct cohort indices")
    entries = [cohort["entries"][i] for i in args.indices]
    validate_frames(entries, cohort_path.parent, max_width=1920, max_height=1080, require_timestamps=False)
    if any(type(e["gt_count"]) is not int or not 0 <= e["gt_count"] <= 512 for e in entries):
        parser.error("Invalid GT counts")
    output = args.output_dir.resolve()
    if output.is_relative_to(Path(cohort["dataset"])) or cohort_path.is_relative_to(output):
        parser.error("Output must be separate from original dataset and RGB inputs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "mode": args.mode, "variant":args.variant, "cohort": str(cohort_path),
              "prediction_head":"den1 primary; density heads separate; localization peaks not decoded",
              "cohort_sha256": args.cohort_sha256, "indices": args.indices,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
              "script_sha256": sha256(__file__), "loader_sha256": sha256(Path(__file__).with_name("count_video.py")),
              "model_code_sha256": sha256(Path(__file__).resolve().parents[1]/"STNNet/stnmodel.py"),
              "memory_limits": memory_limits(), "torch_version": torch.__version__, "numpy_version": np.__version__,
              "device": "cuda", "density_factor": 100, "rows": [],
              "interpretation": "Metrics on a frozen annotated subset; not full benchmark or march-video accuracy."}
    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        model = load_density_checkpoint(args.checkpoint, args.checkpoint_sha256, use_loc=args.variant=="localization").to("cuda")
        report["loaded_state_tensors"]=len(model.state_dict())
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for entry in entries:
                if time.monotonic() > deadline:
                    raise TimeoutError("Evaluation deadline exhausted")
                path = cohort_path.parent / entry["rgb_file"]
                if sha256(path) != entry["rgb_sha256"]:
                    raise ValueError("RGB SHA-256 mismatch")
                rgb = read_rgb(path, entry["width"], entry["height"], max_width=1920, max_height=1080)
                if args.mode == "quadrants":
                    density, head_counts = historical_density(rgb, model)
                    count, sums = head_counts[0]
                else:
                    def predict(crop):
                        return model([normalized_rgb(crop).to("cuda")])[0][0][0, 0].cpu().numpy().copy()
                    density, _ = assemble_density(rgb, predict, deadline)
                    count, sums = quadrant_count(density)
                    head_counts = [(count, sums)]
                density_path = output / f"density-{entry['index']:03d}.npy"
                np.save(density_path, density, allow_pickle=False)
                report["rows"].append({"index": entry["index"], "sequence": entry["sequence"],
                    "gt_count": entry["gt_count"], "count": count, "signed_error": count-entry["gt_count"],
                    "head_counts": [item[0] for item in head_counts], "quadrant_density_sums": sums,
                    "density_file": density_path.name, "density_sha256": sha256(density_path)})
        report.update(status="complete", metrics=error_metrics(report["rows"]), seconds=time.monotonic()-started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "mode", "metrics", "seconds", "max_rss_kib", "peak_cuda_bytes")}))


if __name__ == "__main__":
    main()
