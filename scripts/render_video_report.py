"""Render local inference evidence with the existing base Python plotting stack."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.root.glob("*/inferencia-validada/inference.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete inference: {path}")
        inputs = json.loads(Path(report["input_manifest"]).read_text())
        samples = {frame["index"]: frame for frame in inputs["frames"]}
        records = []
        for frame in report["frames"]:
            sample = samples[frame["index"]]
            image = np.asarray(Image.open(Path(report["input_manifest"]).parent / sample["rgb_file"]))
            density = np.load(path.parent / frame["density_file"], allow_pickle=False) / 100
            records.append((frame, image, density))
        reports.append((path.parent.parent.name, report, records))
    if not reports:
        raise ValueError("No completed inference manifests")
    maximum = max(float(np.quantile(density, .997)) for _, _, records in reports for _, _, density in records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, report, records in reports:
        figure, axes = plt.subplots(len(records), 2, figsize=(14, 4 * len(records)), squeeze=False, layout="constrained")
        figure.suptitle(f"STNNet — video {name}, 960×540\nSalidas exploratorias; sin validación de precisión")
        for axes_row, (frame, image, density) in zip(axes, records):
            axes_row[0].imshow(image)
            axes_row[0].set_title(f"Fotograma original muestreado — {frame['pts_seconds']:.3f} s")
            plotted = axes_row[1].imshow(density, cmap="magma", vmin=0, vmax=maximum)
            axes_row[1].set_title(f"Mapa de densidad — salida: {frame['count']:.1f} personas")
            for axis in axes_row:
                axis.axis("off")
            rows.append({"video": name, "index": frame["index"], "pts_seconds": frame["pts_seconds"], "count": frame["count"]})
        figure.colorbar(plotted, ax=axes[:, 1], shrink=.7, label="Salida / 100 por píxel del mapa; escala común (saturada al percentil99,7)")
        figure.savefig(args.output_dir / f"{name}.png", dpi=120)
        plt.close(figure)
    with (args.output_dir / "counts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "index", "pts_seconds", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"reports": len(reports), "frames": len(rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
