"""Compare local native/interpolated density maps on a common spatial grid."""
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
    groups = {}
    for path in sorted(args.root.glob("*/*/inference.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete inference: {path}")
        raw = np.load(path.parent / report["density_file"], allow_pickle=False) / 100
        # BOX is an exact spatial block average here (scale factors 1, 2 or 4).
        # Multiply by the area ratio so display values conserve density mass.
        display = np.asarray(Image.fromarray(raw).resize((480, 270), Image.Resampling.BOX)) * (raw.size / (480 * 270))
        groups.setdefault(path.parent.parent.name, []).append((path.parent.name, report, display))
    if not groups:
        raise ValueError("No completed runs")
    maximum = max(float(np.quantile(display, .997)) for records in groups.values() for _, _, display in records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for video, records in groups.items():
        records.sort(key=lambda record: (record[0] != "native", record[1]["scale"]))
        first = records[0][1]
        inputs = json.loads(Path(first["input_manifest"]).read_text())
        frame = next(f for f in inputs["frames"] if f["index"] == first["frame_index"])
        preview = Image.open(Path(first["input_manifest"]).parent / frame["preview_file"])
        figure, axes = plt.subplots(1, 1+len(records), figsize=(6*(1+len(records)), 5), layout="constrained")
        figure.suptitle(f"Video {video} — {first['pts_seconds']:.3f} s\nComparación de resolución; precisión sin medir")
        axes[0].imshow(preview)
        axes[0].set_title("Vista de referencia (preview)")
        for axis, (variant, report, display) in zip(axes[1:], records):
            label = "Resolución nativa" if variant == "native" else f"Lanczos {report['scale']}× desde 960×540"
            plotted = axis.imshow(display, cmap="magma", vmin=0, vmax=maximum)
            axis.set_title(f"{label}\n{report['width']}×{report['height']}: {report['count']:.1f}")
            rows.append({"video": video, "variant": variant, "pts_seconds": report["pts_seconds"],
                         "width": report["width"], "height": report["height"], "count": report["count"],
                         "seconds": report["seconds"]})
        for axis in axes:
            axis.axis("off")
        figure.colorbar(plotted, ax=axes[1:], shrink=.65, label="Densidad por celda de la misma grilla espacial 480×270")
        figure.savefig(args.output_dir / f"{video}.png", dpi=110)
        plt.close(figure)
    with (args.output_dir / "counts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "display.json").write_text(json.dumps({"grid": [480, 270], "density_mass_preserved": True,
        "shared_maximum": maximum, "maximum_rule": "maximum of per-map 99.7th percentiles; negatives displayed black"}, indent=2)+"\n")
    print(json.dumps({"videos": len(groups), "runs": len(rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
