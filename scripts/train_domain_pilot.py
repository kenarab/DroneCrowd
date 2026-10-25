"""Finite den1-only adaptation experiment; original weights and dataset stay intact."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from audit_hdf5 import reconstruct_density
from count_video import load_density_checkpoint, normalized_rgb, read_rgb, validate_frames
from domain_targets import density_loss, supervision_mask
from sample_video import memory_limits, sha256

ORIGINAL_DENSITY_SHA256 = "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d"


def configure_trainable(model, adaptation):
    if adaptation not in ("head", "conv4-head"):
        raise ValueError("Unknown adaptation scope")
    prefixes = ("den_output_layer1.",) if adaptation == "head" else ("den_output_layer1.", "conv4.")
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            names.append(name)
    expected = 129 if adaptation == "head" else 147713
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != expected:
        raise ValueError("Unexpected trainable parameter count")
    return names


def state_hashes(model):
    return {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.state_dict().items()}


def verify_updates(before, after, allowed):
    if set(before) != set(after):
        raise ValueError("Model state keys changed")
    changed = sorted(name for name in before if before[name] != after[name])
    if not changed or set(changed)-set(allowed):
        raise ValueError("No real update or frozen state changed")
    return changed


class OriginalTargets:
    """Two full-frame maps in RAM; crop AFTER recreating historical Gaussians."""
    def __init__(self):
        self.cache = OrderedDict()

    def get(self, entry):
        if (entry.get("origin") != "original" or entry.get("split") != "train"
                or (entry["full_width"], entry["full_height"]) != (1920, 1080)):
            raise ValueError("Original targets must come from native training frames")
        points = np.asarray(entry["full_points_xy"], dtype=np.float64)
        if points.size == 0:
            points = points.reshape(0, 2)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) > 512
                or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= 1920).any() or (points[:, 1] >= 1080).any()):
            raise ValueError("Invalid full-frame training points")
        key = (entry["source_sha256"], entry["source_mat_sha256"])
        if key not in self.cache:
            self.cache[key] = reconstruct_density(points)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        x, y, right, bottom = entry["box_xyxy"]
        if (any(type(v) is not int or v % 4 for v in (x, y, right, bottom))
                or not 0 <= x < right <= 1920 or not 0 <= y < bottom <= 1080
                or (right-x, bottom-y) != (entry["width"], entry["height"])):
            raise ValueError("Invalid historical crop geometry")
        result = self.cache[key][y//2:bottom//2, x//2:right//2].copy()
        return result, np.ones(result.shape, dtype=bool)


def new_target(entry):
    x, y, right, bottom = entry["roi_context_xyxy"]
    if (any(type(v) is not int or v % 4 for v in (x, y, right, bottom))
            or not 0 <= x < right <= entry["width"] or not 0 <= y < bottom <= entry["height"]):
        raise ValueError("Invalid new-label ROI geometry")
    region, valid = supervision_mask(entry["points_confirmed_xy"], entry["points_uncertain_xy"],
                                       right-x, bottom-y, entry["supervision"])
    target = np.zeros((entry["height"]//2, entry["width"]//2), dtype=np.float32)
    mask = np.zeros(target.shape, dtype=bool)
    target[y//2:bottom//2, x//2:right//2] = region
    mask[y//2:bottom//2, x//2:right//2] = valid
    return target, mask


def verified_manifest(path, digest):
    path = path.resolve(strict=True)
    if not 0 < path.stat().st_size <= 1024*1024 or sha256(path) != digest:
        raise ValueError("Manifest size/hash mismatch")
    data = json.loads(path.read_text())
    validate_frames(data["entries"], path.parent, max_width=768, max_height=768,
                    max_frames=20, require_timestamps=False)
    return data


def collapse_gate(model, entries, manifest, cache, deadline, device="cuda"):
    """Detect catastrophic all-zero counts only on three positive original crops.

    This is an internal guard, separate from the full historical retention test.
    Negative controls are never used to decide collapse.
    """
    rows = []
    model.eval()
    with torch.inference_mode():
        for entry in entries:
            target, _ = cache.get(entry)
            mass = float(target.sum(dtype=np.float64))
            if mass < 1:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError("Collapse guard deadline exhausted")
            rgb = read_rgb(manifest.parent/entry["rgb_file"], entry["width"], entry["height"], max_width=512, max_height=512)
            density = model([normalized_rgb(rgb).to(device)])[0][0][0, 0]
            count = float(density.sum(dtype=torch.float64)/100)
            if not np.isfinite(count):
                raise ValueError("Non-finite collapse guard count")
            rows.append({"name": entry["name"], "target_mass": mass, "raw_count": count, "count": max(0., count)})
            if len(rows) == 3:
                break
    if len(rows) != 3 or not any(row["count"] > 0 for row in rows):
        raise ValueError("Missing positive guard examples or catastrophic collapse to zero")
    return rows


def training_count_gate(before_counts, after_counts):
    """Bound total train16 count inflation/collapse against the starting model."""
    before_total = float(np.sum(before_counts, dtype=np.float64))
    after_total = float(np.sum(after_counts, dtype=np.float64))
    if (len(before_counts) != 16 or len(after_counts) != 16
            or not np.isfinite([before_total, after_total]).all() or before_total <= 0):
        raise ValueError("Train16 count gate requires sixteen finite positive-baseline counts")
    ratio = after_total / before_total
    return {"train_examples": 16, "original_counts": [float(x) for x in before_counts],
            "updated_counts": [float(x) for x in after_counts], "original_count_sum": before_total,
            "updated_count_sum": after_total, "updated_to_original_ratio": ratio,
            "accepted_ratio": [0.5, 1.5], "accepted": 0.5 <= ratio <= 1.5,
            "generalization_evidence": False}


def require_training_count_gate(result):
    if not result["accepted"]:
        raise ValueError("Train16 count inflation/collapse gate failed: "
                         f"ratio={result['updated_to_original_ratio']:.6g}")
    return result


def train_counts(model, entries, manifest, deadline, device="cuda"):
    model.eval()
    counts = []
    with torch.inference_mode():
        for entry in entries:
            if time.monotonic() > deadline:
                raise TimeoutError("Train16 count gate deadline exhausted")
            rgb = read_rgb(manifest.parent/entry["rgb_file"], entry["width"], entry["height"],
                           max_width=512, max_height=512)
            density = model([normalized_rgb(rgb).to(device)])[0][0][0, 0]
            count = float(density.sum(dtype=torch.float64)/100)
            if not np.isfinite(count):
                raise ValueError("Non-finite train16 count")
            counts.append(count)
    if len(counts) != 16:
        raise ValueError("Train16 count gate requires exactly sixteen training crops")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--controls-sha256")
    parser.add_argument("--adaptation", choices=["head", "conv4-head"], default="head")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--original-only150", action="store_true",
                        help="Run the 150-step original replay control; controls manifest is not needed")
    parser.add_argument("--allow-provisional-labels", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200 or (args.original_only and args.steps > 5):
        parser.error("One to 200 steps; --original-only smoke is limited to five")
    if args.original_only and args.original_only150:
        parser.error("Choose only one original-only mode")
    original_only = args.original_only or args.original_only150
    if not args.original_only150 and (args.controls is None or not args.controls_sha256):
        parser.error("--controls and --controls-sha256 are required outside --original-only150")
    if args.original_only150 and args.steps != 150:
        parser.error("--original-only150 requires exactly 150 steps")
    if args.checkpoint_sha256 != ORIGINAL_DENSITY_SHA256:
        parser.error("Each experiment must start from the verified original density checkpoint")
    started, cache = time.monotonic(), OriginalTargets()
    replay = verified_manifest(args.replay, args.replay_sha256)
    controls = None if args.original_only150 else verified_manifest(args.controls, args.controls_sha256)
    old = replay["entries"]
    new = [] if controls is None else [e for e in controls["entries"] if e.get("split") == "train"]
    if not old or (not original_only and not new) or any(e.get("split") != "train" for e in old):
        parser.error("Nonempty, correctly split training cohorts required")
    if args.original_only150 and len(old) != 16:
        parser.error("The 150-step control requires exactly sixteen original replay crops")
    partition = replay["sequences"]
    if set(partition["train"]) & (set(partition["test"]) | set(partition["val"])):
        parser.error("Historical train/test sequence overlap")
    if controls is not None:
        for field in ("source_group", "source_sha256"):
            train_values = {e[field] for e in new}
            test_values = {e[field] for e in controls["entries"] if e.get("split") == "test"}
            if train_values & test_values:
                parser.error("New source group/hash crosses train/test")
    for entry in old+new:
        if max(entry["width"], entry["height"]) > 512:
            parser.error("Training context exceeds 512 pixels")
    if not original_only and not args.allow_provisional_labels and any(e.get("review_status") != "user_reviewed" for e in new):
        parser.error("Provisional labels require an explicitly exploratory run")
    # Verify all training RGB before allocating CUDA; test RGB never enter gradients.
    training_manifests = [(args.replay, old)]
    if controls is not None:
        training_manifests.append((args.controls, new))
    for path, entries in training_manifests:
        for entry in entries:
            if sha256(path.parent/entry["rgb_file"]) != entry["rgb_sha256"]:
                parser.error("Training RGB hash mismatch")
    output = args.output_dir.resolve()
    inputs = [args.replay, args.checkpoint] + ([] if args.controls is None else [args.controls])
    if any(p.resolve().is_relative_to(output) for p in inputs):
        parser.error("Output overlaps experiment inputs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "adaptation": args.adaptation, "steps_requested": args.steps,
        "original_only": original_only, "seed": 20261025, "learning_rate": 1e-5,
        "optimizer": "fresh Adam; grad norm clipped to 1", "batch_size": 1,
        "mixture": "150-step original-only control" if args.original_only150 else
            ("original-only smoke" if args.original_only else "one new then three original; equal per-example mean loss"),
        "density_factor": 100, "prediction_head": "den1", "history": [],
        "inputs": {name: {"path": str(getattr(args, name).resolve()), "sha256": getattr(args, name+"_sha256")}
                   for name in ("replay", "checkpoint")} |
                  ({} if args.controls is None else {"controls": {"path": str(args.controls.resolve()),
                                                                   "sha256": args.controls_sha256}}),
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("train_domain_pilot.py", "domain_targets.py", "audit_hdf5.py", "count_video.py")},
        "model_code_sha256": sha256(Path(__file__).resolve().parents[1]/"STNNet/stnmodel.py"),
        "memory_limits": memory_limits(), "torch_version": torch.__version__,
        "interpretation": "Exploratory adaptation using provisional agent labels; no certified march accuracy."}
    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        torch.set_num_threads(1)
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        model = load_density_checkpoint(args.checkpoint, args.checkpoint_sha256).to("cuda").train()
        allowed = configure_trainable(model, args.adaptation)
        before = state_hashes(model)
        original_counts = train_counts(model, old, args.replay, started+110)
        report["train16_original_counts"] = original_counts
        model.train()
        report["trainable_names"] = allowed
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        torch.cuda.reset_peak_memory_stats()
        old_index = new_index = 0
        for step in range(args.steps):
            if time.monotonic()-started > 110:
                raise TimeoutError("Training deadline exhausted before checkpoint save")
            is_new = not original_only and step % 4 == 0
            if is_new:
                entry, manifest = new[new_index % len(new)], args.controls
                new_index += 1
                target, mask = new_target(entry)
            else:
                entry, manifest = old[old_index % len(old)], args.replay
                old_index += 1
                target, mask = cache.get(entry)
            rgb = read_rgb(manifest.parent/entry["rgb_file"], entry["width"], entry["height"], max_width=512, max_height=512)
            optimizer.zero_grad(set_to_none=True)
            prediction = model([normalized_rgb(rgb).to("cuda")])[0][0][0, 0]
            loss = density_loss(prediction, torch.from_numpy(target).to("cuda"), torch.from_numpy(mask).to("cuda"),
                                partial=is_new and entry["supervision"] == "partial_positive")
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()  # unconditional finite-gradient step; unlike historical pure-den branch
            row = {"step": step+1, "origin": "new" if is_new else "original", "name": entry["name"],
                "loss": float(loss.detach()), "valid_pixels": int(mask.sum()), "gradient_norm": float(norm),
                "target_count_entire_context": float(target.sum(dtype=np.float64)),
                "target_mass_on_supervised_pixels": float(target[mask].sum(dtype=np.float64)),
                "supervision": entry["supervision"] if is_new else "historical_full_crop",
                "raw_prediction_count_entire_context": float(prediction.detach().sum(dtype=torch.float64)/100)}
            if is_new:
                rx, ry = entry["roi_context_xyxy"][:2]
                row["confirmed_heads"] = len(entry["points_confirmed_xy"])
                row["confirmed_heads_with_supervised_center"] = sum(
                    bool(mask[int((py+ry)/2), int((px+rx)/2)]) for px, py in entry["points_confirmed_xy"])
            if not np.isfinite(row["raw_prediction_count_entire_context"]):
                raise ValueError("Non-finite predicted count")
            report["history"].append(row)
            if step % 20 == 0 or step == args.steps-1:
                print(json.dumps(row), flush=True)
        updated_counts = train_counts(model, old, args.replay, started+115)
        report["train16_updated_counts"] = updated_counts
        report["train16_count_gate"] = training_count_gate(original_counts, updated_counts)
        report["historical_retention"] = {
            "status": "pending; no historical test30 evaluation used for model selection",
            "acceptance_mae": 24.3621808, "original_reference_mae": 20.3018173}
        require_training_count_gate(report["train16_count_gate"])
        report["collapse_guard"] = collapse_gate(model, old, args.replay, cache, started+115)
        after = state_hashes(model)
        report["changed_parameters"] = verify_updates(before, after, allowed)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Non-finite updated model")
        checkpoint = output/"den-domain-pilot.pth.tar"
        if time.monotonic() > started+115:
            raise TimeoutError("Training deadline exhausted before checkpoint save")
        torch.save({"state_dict": {n: t.detach().cpu() for n, t in model.state_dict().items()},
                    "pilot_metadata": {"adaptation": args.adaptation, "steps": args.steps, "density_factor": 100,
                                       "parent_sha256": args.checkpoint_sha256}}, checkpoint)
        report.update(status="complete", steps_completed=len(report["history"]),
            original_steps=old_index, new_steps=new_index, checkpoint_file=checkpoint.name,
            checkpoint_sha256=sha256(checkpoint), frozen_state_sha256_before=before, state_sha256_after=after,
            seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output/"training.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
