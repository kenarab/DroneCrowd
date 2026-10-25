"""Compare the two preregistered domain pilots with unchanged baseline controls."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sample_video import sha256


def read_complete(path):
    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise ValueError(f"Incomplete experiment: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {str(args.historical_summary): sha256(args.historical_summary)}
    historical = json.loads(args.historical_summary.read_text())
    baseline_mae = historical["metrics"]["quadrants"]["mae"]
    threshold = 1.2*baseline_mae
    cases = [("Original", "baseline-rois"), ("Cabeza", "mixto-head-rois"),
             ("Conv4 + cabeza", "mixto-conv4-rois")]
    runs = []
    for label, name in cases:
        path = args.root/name/"evaluation.json"
        report = read_complete(path)
        sources[str(path)] = sha256(path)
        runs.append((label, path.parent, report))
    controls_path = Path(runs[0][2]["controls"])
    digest = sha256(controls_path)
    if any(r["controls_sha256"] != digest or r["scale"] != 1 or len(r["rows"]) != 7 for _, _, r in runs):
        raise ValueError("Expected identical seven native-resolution ROI controls")
    controls = json.loads(controls_path.read_text())
    sources[str(controls_path)] = digest
    comparisons = [{"name": "original", "metrics": historical["metrics"]["quadrants"], "accepted": True}]
    for name, retention, report_index in (("mixto-head", "mixto-head-retencion", 1),
                                           ("mixto-conv4-head", "mixto-conv4-retencion", 2)):
        train_path, test_path = args.root/name/"training.json", args.root/retention/"evaluation.json"
        train, test = read_complete(train_path), read_complete(test_path)
        sources.update({str(train_path): sha256(train_path), str(test_path): sha256(test_path)})
        if (test["cohort_sha256"] != historical["cohort_sha256"] or test["mode"] != "quadrants"
                or test["indices"] != list(range(30))
                or train["checkpoint_sha256"] != test["checkpoint_sha256"]
                or train["checkpoint_sha256"] != runs[report_index][2]["checkpoint_sha256"]):
            raise ValueError("Checkpoint or historical cohort mismatch")
        losses = {}
        for policy in sorted({r["supervision"] for r in train["history"]}):
            values = [r["loss"] for r in train["history"] if r["supervision"] == policy]
            losses[policy] = {"steps": len(values), "mean": float(np.mean(values)), "first": values[0], "last": values[-1]}
        comparisons.append({"name": name, "metrics": test["metrics"],
            "retention_pass": test["metrics"]["mae"] <= threshold, "accepted": False,
            "checkpoint_sha256": train["checkpoint_sha256"], "trainable_parameters": train["trainable_parameters"],
            "original_steps": train["original_steps"], "new_steps": train["new_steps"],
            "training_seconds": train["seconds"], "max_rss_kib": train["max_rss_kib"],
            "peak_cuda_bytes": train["peak_cuda_bytes"], "changed_parameters": train["changed_parameters"],
            "loss_by_supervision": losses})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    table = []
    for entry in controls["entries"]:
        row = {"name": entry["name"], "split": entry["split"], "reference_interval": entry["reference_interval"]}
        for label, directory, report in runs:
            result = next(r for r in report["rows"] if r["index"] == entry["index"])
            row[label] = result["count"]
        table.append(row)
    with (output/"counts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    reserved = [entry for entry in controls["entries"] if entry["split"] == "test"]
    fig, axes = plt.subplots(len(reserved), 4, figsize=(16, 4*len(reserved)), squeeze=False, layout="constrained")
    fig.suptitle("Regiones reservadas · marcas provisionales de agentes\nDensidad den1: cada fila comparte escala de color")
    for entry, row_axes in zip(reserved, axes):
        x, y, right, bottom = entry["roi_context_xyxy"]
        source = controls_path.parent/entry["rgb_file"]
        if sha256(source) != entry["rgb_sha256"]:
            raise ValueError("RGB changed")
        with Image.open(source) as image:
            row_axes[0].imshow(image.crop((x, y, right, bottom)), interpolation="nearest")
        for field, color in (("points_confirmed_xy", "lime"), ("points_uncertain_xy", "orange")):
            for px, py in entry[field]:
                row_axes[0].scatter(px, py, s=30, facecolors="none", edgecolors=color, linewidths=.8)
        row_axes[0].set_title(entry["name"]+"\nReferencia provisional: "+str(entry["reference_interval"]))
        maps = []
        for label, directory, report in runs:
            result = next(r for r in report["rows"] if r["index"] == entry["index"])
            path = directory/result["density_file"]
            if sha256(path) != result["density_sha256"]:
                raise ValueError("Density map changed")
            raw = np.load(path, allow_pickle=False)
            region = raw[y//2:bottom//2, x//2:right//2]/100
            if not np.isfinite(region).all() or abs(float(region.sum(dtype=np.float64))-result["raw_roi_count"]) > 1e-4:
                raise ValueError("Density integral does not match report")
            sources[str(path)] = sha256(path)
            maps.append((label, result, region))
        vmax = max(1e-6, max(float(np.quantile(array, .995)) for _, _, array in maps))
        for axis, (label, result, array) in zip(row_axes[1:], maps):
            axis.imshow(array, cmap="magma", vmin=0, vmax=vmax)
            axis.set_title(f"{label}\nMasa/personas: {result['count']:.2f}")
        for axis in row_axes:
            axis.axis("off")
    fig.savefig(output/"controles-reservados.png", dpi=130)
    plt.close(fig)
    summary = {"baseline_mae": baseline_mae, "retention_threshold": threshold, "comparisons": comparisons,
        "roi_counts": table, "source_sha256": sources, "script_sha256": sha256(__file__),
        "decision": "Reject both pilots; retain original checkpoint. No evidence of improved march counting.",
        "limits": "Thirty previously observed historical controls; march labels provisional, same-event sources; no independent march accuracy."}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"decision": summary["decision"], "retention_threshold": threshold, "output": str(output)}))


if __name__ == "__main__":
    main()
