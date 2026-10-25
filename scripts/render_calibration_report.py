"""Aggregate the disjoint pilot/remainder and plot frozen annotated controls."""
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


def metrics(rows):
    errors = np.array([r["count"]-r["gt_count"] for r in rows], dtype=np.float64)
    return {"images": len(rows), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())), "bias": float(errors.mean()),
            "gt_total": sum(r["gt_count"] for r in rows), "prediction_total": sum(r["count"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources, groups, paths = {}, {"quadrants": [], "tiles": []}, {}
    for name in ("cuadrantes-piloto", "cuadrantes-restantes", "mosaico-historico"):
        path = args.root/name/"evaluation.json"
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError("Incomplete evaluation")
        sources[str(path)] = sha256(path)
        for row in report["rows"]:
            groups[report["mode"]].append(row)
            key = report["mode"], row["index"]
            if key in paths:
                raise ValueError("Duplicate cohort record")
            paths[key] = path.parent/row["density_file"]
    if any({r["index"] for r in rows} != set(range(30)) for rows in groups.values()):
        raise ValueError("Expected exactly thirty distinct records per mode")
    cohort_path = Path(report["cohort"])
    cohort = json.loads(cohort_path.read_text())
    summary = {"quadrants": metrics(groups["quadrants"]), "tiles": metrics(groups["tiles"])}
    for head in (1, 2):
        summary[f"quadrants_head{head+1}"] = metrics([{**r, "count": r["head_counts"][head]} for r in groups["quadrants"]])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"summary.json").write_text(json.dumps({"metrics": summary, "source_reports": sources,
        "cohort_sha256": sha256(cohort_path), "script_sha256": sha256(__file__),
        "interpretation": "30 selected images; den1 is primary; not the full benchmark or march accuracy"}, indent=2)+"\n")
    with (args.output_dir/"counts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "index", "sequence", "gt_count", "count", "signed_error"])
        writer.writeheader()
        for mode, rows in groups.items():
            writer.writerows({"mode": mode, **{k: r[k] for k in writer.fieldnames if k != "mode"}} for r in sorted(rows, key=lambda r:r["index"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for mode, rows in groups.items():
        axes[0].scatter([r["gt_count"] for r in rows], [r["count"] for r in rows], label=mode, alpha=.7)
    axes[0].plot([0, 300], [0, 300], color="gray", linestyle="--")
    axes[0].set(xlabel="Personas anotadas", ylabel="Salida den1", title="30 escenas históricas anotadas")
    axes[0].legend()
    axes[1].bar(["4 crops den1", "mosaico den1", "4 crops den2", "4 crops den3"], [m["mae"] for m in summary.values()])
    axes[1].set(ylabel="MAE (personas / imagen)", title="Cabezas evaluadas por separado")
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig(args.output_dir/"metricas.png", dpi=130)
    plt.close(fig)
    selected = [next(e for e in cohort["entries"] if e["sequence"] == sequence) for sequence in ("011", "017", "023", "070")]
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), layout="constrained")
    for axes_row, entry in zip(axes, selected):
        axes_row[0].imshow(Image.open(entry["source_image"]))
        points = np.asarray(entry["points_xy_id"])
        axes_row[0].scatter(points[:, 0], points[:, 1], s=8, facecolors="none", edgecolors="yellow", linewidths=.5)
        axes_row[0].set_title(f"Secuencia {entry['sequence']} — {entry['gt_count']} anotados")
        maps = [np.load(paths[mode, entry["index"]], allow_pickle=False)/100 for mode in ("quadrants", "tiles")]
        maximum = max(float(np.quantile(d, .995)) for d in maps)
        for axis, mode, density in zip(axes_row[1:], ("quadrants", "tiles"), maps):
            row = next(r for r in groups[mode] if r["index"] == entry["index"])
            axis.imshow(density, cmap="magma", vmin=0, vmax=maximum)
            axis.set_title(f"{mode} — {row['count']:.1f}")
        for axis in axes_row:
            axis.axis("off")
    fig.suptitle("Controles históricos: puntos originales y mapas den1; escala compartida dentro de cada fila")
    fig.savefig(args.output_dir/"controles-historicos.png", dpi=120)
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
