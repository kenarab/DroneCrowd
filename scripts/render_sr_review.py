"""Create a private static comparison of the frozen eight STNNet/SR controls."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_map(directory, row):
    path = directory / row["density_file"]
    if digest(path) != row["density_sha256"]:
        raise ValueError("Density hash mismatch")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite density map")
    return array


def heatmap(values, maximum):
    unit = np.clip(values / maximum, 0, 1)
    colors = np.stack([np.minimum(1, 2 * unit), np.maximum(0, 2 * unit - 1),
                       np.zeros_like(unit)], axis=-1)
    return Image.fromarray((colors * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--stn", type=Path, required=True)
    parser.add_argument("--sr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    controls = json.loads(args.controls.read_text())
    stn = json.loads(args.stn.read_text())
    sr = json.loads(args.sr.read_text())
    if (stn["status"] != "complete" or sr["status"] != "complete" or stn["scale"] != 1
            or stn["controls_sha256"] != digest(args.controls) or sr["input_sha256"] != digest(args.controls)):
        raise ValueError("Reports must refer to the same frozen native controls")
    if not len(controls["entries"]) == len(stn["rows"]) == len(sr["results"]) == 8:
        raise ValueError("Expected eight matching controls")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index, cards = [], []
    for i, (control, old, new) in enumerate(zip(controls["entries"], stn["rows"], sr["results"])):
        if (not control["name"] == old["name"] == new["name"] or new["rgb_sha256"] != control["rgb_sha256"]
                or new["roi_context_xyxy"] != control["roi_context_xyxy"]
                or new["effective_size"] != [control["width"], control["height"]]):
            raise ValueError("Control identity or geometry differs")
        source = args.controls.parent / control["rgb_file"]
        if digest(source) != control["rgb_sha256"]:
            raise ValueError("RGB hash mismatch")
        box = control["roi_context_xyxy"]
        with Image.open(source) as original:
            rgb = original.convert("RGB").crop(tuple(box))
        maps = []
        for row, report, stride, factor in [(old, args.stn, 2, 100), (new, args.sr, 8, 1)]:
            density = checked_map(report.parent, row)
            if density.shape != (control["height"] // stride, control["width"] // stride):
                raise ValueError("Density grid mismatch")
            # Common visualization unit: persons per native input pixel.
            full = Image.fromarray((np.maximum(0, density) / (factor * stride ** 2)).astype(np.float32))
            maps.append(np.asarray(full.resize((control["width"], control["height"]), Image.Resampling.NEAREST)
                                   .crop(tuple(box))))
        maximum = max(1e-8, float(np.quantile(np.concatenate([m.ravel() for m in maps]), .99)))
        panel = Image.new("RGB", (1536, max(120, rgb.height) + 58), "#202020")
        draw = ImageDraw.Draw(panel)
        titles = [f"{control['name']} | RGB nativo", f"STNNet: {old['count']:.2f}", f"SR: {new['count']:.2f}"]
        for column, (title, image) in enumerate(zip(titles, [rgb, *[heatmap(m, maximum) for m in maps]])):
            draw.text((column * 512 + 8, 8), title, fill="white")
            panel.paste(image, (column * 512, 38))
        name = f"control-{i:02d}.png"
        panel.save(args.output_dir / name)
        index.append({"name": control["name"], "stn_count": old["count"], "sr_count": new["count"],
                      "file": name, "sha256": digest(args.output_dir / name),
                      "positive_density_per_input_pixel_color_max": maximum})
        cards.append(f"<article><h2>{html.escape(control['name'])}</h2><p>STNNet {old['count']:.2f}; SR {new['count']:.2f}</p>"
                     f"<a href='{name}'><img src='{name}' alt='{html.escape(control['name'], quote=True)}'></a></article>")
    page = ("<!doctype html><html lang='es'><meta charset='utf-8'><title>SR y STNNet: controles</title>"
            "<style>body{font:18px sans-serif;background:#171717;color:#eee;margin:24px}img{max-width:100%}"
            "a{color:#9cf}article{border-top:1px solid #777;margin-top:24px}</style>"
            "<h1>SR (2026) y STNNet: ocho controles nativos</h1>"
            "<p>Integrales de densidad en la misma región; no detecciones verificadas ni precisión medida. "
            "Las etiquetas siguen siendo parciales. No sumar regiones como asistencia.</p>"
            "<p>Izquierda: RGB. Centro: STNNet. Derecha: SR. Escala de color común entre modelos dentro de cada fila, "
            "en personas por píxel de entrada; cambia entre filas. Más brillante indica más masa. "
            "Negativos de STNNet se ocultan sólo para visualizar; el número conserva su cálculo original. "
            "SR usa núcleo 256/halo160; STNNet 320×128/halo96. Contexto efectivo diferente; mosaicos aproximados.</p>"
            + "".join(cards) + "</html>")
    (args.output_dir / "index.html").write_text(page)
    (args.output_dir / "index.json").write_text(json.dumps({"controls_sha256": digest(args.controls),
        "stn_report_sha256": digest(args.stn), "sr_report_sha256": digest(args.sr),
        "script_sha256": digest(Path(__file__)),
        "index_html_sha256": digest(args.output_dir / "index.html"), "entries": index}, indent=2) + "\n")


if __name__ == "__main__":
    main()
