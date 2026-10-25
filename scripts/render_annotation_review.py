"""Render private, offline visual sheets for reviewing head annotations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from sample_video import sha256

LIMIT = 64 * 2**20
POSITIVE = ("foto-dispersas", "foto-grupo", "test-peaton-fuente")


def file_hash(path, limit):
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f"File exceeds size limit: {path}")
    return sha256(path)


def crop_image(entry):
    source = Path(entry["source_image"]).resolve(strict=True)
    digest = file_hash(source, LIMIT)
    if digest != entry["source_sha256"]:
        raise ValueError(f"Source hash mismatch: {source}")
    left, top, right, bottom = entry["box_xyxy"]
    with Image.open(source) as image:
        if right > image.width or bottom > image.height or min(left, top) < 0:
            raise ValueError(f"ROI outside source image: {source}")
        roi = image.convert("RGB").crop((left, top, right, bottom))
    return roi, source, digest


def render(annotations, output, manifest_hash):
    names = {entry["name"]: entry for entry in annotations["entries"]}
    if len(names) != 7 or set(names) != set(POSITIVE) | {
            "edificio-b", "foto-calle-vacia", "edificio-a", "test-suelo-fuente"}:
        raise ValueError("Expected the seven current annotation regions")
    output = Path(output).resolve()
    if not output.is_relative_to(Path("outputs").resolve()):
        raise ValueError("Review sheets must stay under outputs/")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    fig.suptitle("Anotaciones provisionales: verde = cabeza marcada; naranja D = dudosa\n"
                 "Izquierda sin marcas; derecha puntos usados como referencia parcial")
    for row, name in enumerate(POSITIVE):
        entry = names[name]
        roi, source, digest = crop_image(entry)
        sources[str(source)] = digest
        axes[row, 0].imshow(roi, interpolation="nearest")
        axes[row, 0].set_title(f"{name} · original ROI")
        axes[row, 1].imshow(roi, interpolation="nearest")
        for field, color, prefix in (("points_confirmed_xy", "#00ff7f", ""),
                                     ("points_uncertain_xy", "#f97316", "D")):
            for index, (x, y) in enumerate(entry[field], 1):
                # Annotation coordinates are local to the cropped ROI.
                px, py = x, y
                axes[row, 1].add_patch(Circle((px, py), radius=5, fill=False,
                                             edgecolor=color, linewidth=1.0))
                axes[row, 1].text(px + 6, py - 5, f"{prefix}{index}", color=color,
                                  fontsize=8, weight="bold")
        axes[row, 1].set_title(f"{name} · {entry['split']} · "
            f"marcadas {len(entry['points_confirmed_xy'])} · dudosas {len(entry['points_uncertain_xy'])}")
        for ax in axes[row]:
            ax.set_xlim(-.5, roi.width-.5)
            ax.set_ylim(roi.height-.5, -.5)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "personas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    negatives = [name for name in names if name not in POSITIVE]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, name in zip(axes.flat, negatives):
        entry = names[name]
        roi, source, digest = crop_image(entry)
        sources[str(source)] = digest
        ax.imshow(roi, interpolation="nearest")
        ax.set_title(f"{name} · {entry['split']} · sin personas discernibles (revisión provisional)")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / "negativos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    index = {"manifest_sha256": manifest_hash, "script_sha256": sha256(__file__),
             "source_hashes": sources,
             "png_sha256": {name: sha256(output / name) for name in ("personas.png", "negativos.png")}}
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.annotations.resolve(strict=True)
    manifest_hash = file_hash(manifest, 256 * 1024)
    if manifest_hash != args.annotations_sha256:
        parser.error("Manifest size or SHA-256 mismatch")
    annotations = json.loads(manifest.read_text())
    if annotations.get("schema") != "dronecrowd-head-points-v1":
        parser.error("Unexpected annotation schema")
    render(annotations, args.output_dir, manifest_hash)


if __name__ == "__main__":
    main()
