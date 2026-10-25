"""Private static views of three frozen counters on identical native video frames."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sample_video import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    names = ["sr", "clip-ebc", "mpcount"]
    reports, previews = {}, {}
    for name in names:
        report_path = args.reports_dir / name / "report.json"
        report = json.loads(report_path.read_text())
        if (report["status"] != "complete" or report["input_sha256"] != sha256(args.input)
                or len(report["results"]) != len(manifest["frames"])):
            raise ValueError("Expected complete reports for the same inputs")
        reports[name] = report
        stride = report["geometry"]["stride"]
        for index, (row, frame) in enumerate(zip(report["results"], manifest["frames"])):
            if row["rgb_sha256"] != frame["rgb_sha256"] or row["pts_seconds"] != frame["pts_seconds"]:
                raise ValueError("Frame identity differs")
            path = report_path.parent / row["density_file"]
            if sha256(path) != row["density_sha256"]:
                raise ValueError("Density hash mismatch")
            density = np.load(path, allow_pickle=False)
            if (density.shape != (frame["height"] // stride, frame["width"] // stride)
                    or not np.isfinite(density).all()
                    or not np.isclose(density.sum(dtype=np.float64), row["count"], rtol=1e-12, atol=1e-10)):
                raise ValueError("Invalid density geometry/integral")
            # Standardize visualization to persons per native input pixel.
            display = Image.fromarray((np.maximum(0, density) / stride ** 2).astype(np.float32))
            previews[name, index] = np.asarray(display.resize((960, 540), Image.Resampling.BOX))
    maximum = max(1e-8, float(np.quantile(np.concatenate([p.ravel() for p in previews.values()]), .995)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cards, rows = [], []
    for index, frame in enumerate(manifest["frames"]):
        path = args.input.parent / frame["rgb_file"]
        if sha256(path) != frame["rgb_sha256"]:
            raise ValueError("RGB hash mismatch")
        with Image.open(path) as source:
            rgb = source.convert("RGB").resize((960, 540), Image.Resampling.BOX)
        canvas = Image.new("RGB", (1920, 1160), "#202020")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"Cuadro {index + 1} | PTS {frame['pts_seconds']:.6f}s | RGB fuente 3840x2160, vista reducida", fill="white")
        canvas.paste(rgb, (0, 40))
        counts = {}
        for name, (left, top) in zip(names, [(960, 0), (0, 580), (960, 580)]):
            value = reports[name]["results"][index]["count"]
            counts[name] = value
            draw.text((left + 12, top + 10), f"{name} | Integral {value:.2f} | Sin GT ni mascara de suelo", fill="white")
            intensity = np.clip(previews[name, index] / maximum, 0, 1)
            colors = np.stack([np.minimum(1, 2 * intensity), np.maximum(0, 2 * intensity - 1),
                               np.zeros_like(intensity)], axis=-1)
            canvas.paste(Image.fromarray((colors * 255).astype(np.uint8)), (left, top + 40))
        filename = f"cuadro-{index:02d}.jpg"
        canvas.save(args.output_dir / filename, quality=92)
        rows.append({"index": index, "pts_seconds": frame["pts_seconds"], "counts": counts,
                     "file": filename, "sha256": sha256(args.output_dir / filename)})
        label = "; ".join(f"{name}: {counts[name]:.2f}" for name in names)
        cards.append(f"<article><h2>Cuadro {index + 1}: {frame['pts_seconds']:.6f} s</h2><p>{html.escape(label)}</p>"
                     f"<a href='{filename}'><img src='{filename}' alt='Comparacion cuadro {index + 1}'></a></article>")
    page = ("<!doctype html><html lang='es'><meta charset='utf-8'><title>Tres modelos: video breve</title>"
        "<style>body{font:18px sans-serif;background:#171717;color:#eee;margin:24px}a{color:#9cf}img{max-width:100%}"
        "article{border-top:1px solid #777;padding-top:20px}</style><h1>Tres modelos sobre el mismo tramo de video</h1>"
        "<p>SR, CLIP-EBC ResNet50 y MPCount. Tres cuadros nativos de un tramo de aproximadamente dos segundos. "
        "No sumar sus conteos. Sin etiquetas exhaustivas ni mascara validada de calle/vereda/plaza: incluye arquitectura.</p>"
        "<p>Arriba izquierda: RGB; arriba derecha: SR; abajo izquierda: CLIP-EBC; abajo derecha: MPCount. "
        "Color comun para los tres modelos y todos los cuadros, en masa positiva por pixel nativo; "
        "saturado al percentil99,5 global. Los numeros conservan la integral completa. "
        "Las vistas estan reducidas; los modelos recibieron pixeles nativos con sus ventanas y contextos propios.</p>"
        + "".join(cards) + "</html>")
    (args.output_dir / "index.html").write_text(page)
    (args.output_dir / "index.json").write_text(json.dumps({"input_sha256": sha256(args.input),
        "report_sha256": {name: sha256(args.reports_dir / name / "report.json") for name in names},
        "script_sha256": sha256(__file__), "index_html_sha256": sha256(args.output_dir / "index.html"),
        "positive_density_per_input_pixel_color_max": maximum, "frames": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
