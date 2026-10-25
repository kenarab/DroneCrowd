"""Measure initial 129-parameter den1 gradients without updating model weights."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from count_video import load_density_checkpoint, normalized_rgb, read_rgb
from domain_targets import density_loss
from sample_video import memory_limits, sha256
from train_domain_pilot import (OriginalTargets, configure_trainable, new_target,
                                ORIGINAL_DENSITY_SHA256, state_hashes, verified_manifest)


def schedule_frequencies(original_count, new_count):
    if original_count != 16 or new_count != 4:
        raise ValueError("Gradient diagnostic requires 16 original and 4 new train crops")
    original = [150 // original_count + (i < 150 % original_count) for i in range(original_count)]
    new = [50 // new_count + (i < 50 % new_count) for i in range(new_count)]
    return original, new


def pairwise_source_products(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(np.asarray(row["gradient"], dtype=np.float64))
    result = []
    names = sorted(groups)
    for i, left_name in enumerate(names):
        for right_name in names[i:]:
            products, cosines = [], []
            for left in groups[left_name]:
                for right in groups[right_name]:
                    product = float(np.dot(left, right))
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    products.append(product)
                    cosines.append(product / denominator if denominator else 0.0)
            result.append({"source_a": left_name, "source_b": right_name,
                           "pairs": len(products), "mean_dot_product": float(np.mean(products)),
                           "mean_cosine": float(np.mean(cosines))})
    return result


def weighted_aggregates(rows):
    result = {}
    for supervision in ("historical_full_crop", "partial_positive", "negative"):
        selected = [row for row in rows if row["supervision"] == supervision]
        if not selected:
            raise ValueError(f"Missing gradients for {supervision}")
        vectors = [np.asarray(row["gradient"], dtype=np.float64) * row["frequency"] for row in selected]
        total = np.sum(vectors, axis=0)
        result[supervision] = {"steps": sum(row["frequency"] for row in selected),
            "weighted_gradient_sum": total.tolist(), "weighted_sum_norm": float(np.linalg.norm(total)),
            "interpretation": "Initial weighted gradients only; not an Adam update estimate"}
    expected = {"historical_full_crop": 150, "partial_positive": 26, "negative": 24}
    if {k: v["steps"] for k, v in result.items()} != expected:
        raise ValueError("Training cohort does not match 150/26/24 mixture frequencies")
    return result


def gradient_cohorts(old_entries, new_entries, old_path, new_path, old_frequency, new_frequency):
    """Pair parsed cohort entries with the filesystem paths of their RGB manifests."""
    return ((old_entries, Path(old_path), old_frequency, False),
            (new_entries, Path(new_path), new_frequency, True))


def measure_gradient(model, entry, manifest, target, mask, partial, device):
    parameters = [model.den_output_layer1.weight, model.den_output_layer1.bias]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    model.zero_grad(set_to_none=True)
    try:
        rgb = read_rgb(manifest.parent/entry["rgb_file"], entry["width"], entry["height"],
                       max_width=512, max_height=512)
        prediction = model([normalized_rgb(rgb).to(device)])[0][0][0, 0]
        loss = density_loss(prediction, torch.from_numpy(target).to(device),
                            torch.from_numpy(mask).to(device), partial=partial)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite initial gradient loss")
        loss.backward()
        vector = torch.cat([parameters[0].grad.detach().reshape(-1),
                            parameters[1].grad.detach().reshape(-1)])
        raw = vector.cpu().numpy().astype(np.float64)
        raw_norm = float(np.linalg.norm(raw))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        clipped = torch.cat([parameters[0].grad.detach().reshape(-1),
                             parameters[1].grad.detach().reshape(-1)]).cpu().numpy().astype(np.float64)
        return {"loss": float(loss.detach()), "gradient": raw.tolist(),
                "clipped_gradient": clipped.tolist(), "norm_before_clip": raw_norm,
                "norm_after_clip": float(np.linalg.norm(clipped)), "bias_gradient": float(raw[-1]),
                "signs": {"positive": int((raw > 0).sum()), "negative": int((raw < 0).sum()),
                          "zero": int((raw == 0).sum())}}
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "controls", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    replay = verified_manifest(args.replay, args.replay_sha256)
    controls = verified_manifest(args.controls, args.controls_sha256)
    old = replay["entries"]
    new = [entry for entry in controls["entries"] if entry.get("split") == "train"]
    old_freq, new_freq = schedule_frequencies(len(old), len(new))
    for path, entries in ((args.replay, old), (args.controls, new)):
        for entry in entries:
            if sha256(path.parent/entry["rgb_file"]) != entry["rgb_sha256"]:
                parser.error("Training RGB hash mismatch")
    if any(not isinstance(entry.get("source_group"), str) or not entry["source_group"]
           for entry in old + new):
        parser.error("Every diagnostic example needs a source_group")
    if not torch.cuda.is_available():
        parser.error("CUDA required for bounded gradient diagnostic")
    if args.checkpoint_sha256 != ORIGINAL_DENSITY_SHA256:
        parser.error("Gradient diagnostic requires the verified original density checkpoint")
    if not 0 < args.checkpoint.stat().st_size <= 2 * 2**30 or sha256(args.checkpoint) != args.checkpoint_sha256:
        parser.error("Checkpoint size or hash mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    model = load_density_checkpoint(args.checkpoint, args.checkpoint_sha256).cuda().train()
    torch.cuda.reset_peak_memory_stats()
    configure_trainable(model, "head")
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 129:
        raise ValueError("Expected exactly 129 trainable head parameters")
    state_before = state_hashes(model)
    cache, rows = OriginalTargets(), []
    for entries, manifest_path, frequencies, is_new in gradient_cohorts(
            old, new, args.replay, args.controls, old_freq, new_freq):
        for entry, frequency in zip(entries, frequencies):
            if time.monotonic() - started > 110:
                raise TimeoutError("Gradient diagnostic exceeded 110 seconds")
            if is_new:
                target, mask = new_target(entry)
                supervision = entry["supervision"]
                partial = supervision == "partial_positive"
            else:
                target, mask = cache.get(entry)
                supervision, partial = "historical_full_crop", False
            measured = measure_gradient(model, entry, manifest_path, target, mask, partial, "cuda")
            rows.append({"name": entry["name"], "source_group": entry["source_group"],
                         "origin": "new" if is_new else "original", "supervision": supervision,
                         "frequency": frequency, **measured})
    report = {"status": "complete", "model_weights_updated": False, "parameter_count": 129,
        "seed": 20261025,
        "parameter_order": ["den_output_layer1.weight.flatten()", "den_output_layer1.bias"],
        "inputs": {name: {"path": str(getattr(args, name).resolve()),
                          "sha256": getattr(args, name+"_sha256")}
                   for name in ("replay", "controls", "checkpoint")},
        "script_sha256": sha256(__file__), "code_sha256": {name: sha256(Path(__file__).with_name(name))
            for name in ("train_domain_pilot.py", "domain_targets.py", "audit_hdf5.py", "count_video.py", "sample_video.py")},
        "torch_version": torch.__version__, "memory_limits": memory_limits(), "device": "cuda",
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "initial_model_state_sha256": state_before,
        "final_model_state_sha256": state_hashes(model), "examples": rows,
        "pairwise_source_products": pairwise_source_products(rows),
        "frequency_weighted_aggregates": weighted_aggregates(rows),
        "kernel_comparison": "Pending; only valid against complete original annotations.",
        "adam_note": "Gradient norms and weighted sums do not estimate Adam updates or its step clock.",
        "seconds": time.monotonic() - started}
    if report["final_model_state_sha256"] != state_before:
        raise ValueError("Diagnostic unexpectedly changed model state")
    output = args.output_dir.resolve()
    inputs = (args.replay.resolve(), args.controls.resolve(), args.checkpoint.resolve())
    if (any(path.is_relative_to(output) for path in inputs)
            or any(output.is_relative_to(path.parent) for path in inputs)):
        raise ValueError("Output overlaps diagnostic inputs")
    output.mkdir(parents=True, exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
