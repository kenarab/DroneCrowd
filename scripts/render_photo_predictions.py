"""Render private full-photo predictions after verified panel aggregation."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from count_photo_panels import ORIGINAL_CHECKPOINT_SHA256, PRIVATE_ROOT, photo_count_summary
from render_automatic_review import calc_denpt
from sample_video import sha256


def render(manifest_path, manifest_hash, reports, output):
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.stat().st_size > 2**20 or sha256(manifest_path) != manifest_hash:
        raise ValueError("Prepared photo manifest hash/size mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or len(manifest["sources"]) != 8:
        raise ValueError("Expected eight prepared sources")
    if len(reports) != 8 or output.resolve() != Path("revision/fotos-originales").resolve():
        raise ValueError("Expected eight reports and new revision/fotos-originales output")
    output.mkdir(parents=True, exist_ok=False)
    rows, records, seen = [], [], set()
    for path in reports:
        path = path.resolve(strict=True)
        if not path.is_relative_to(PRIVATE_ROOT.resolve()) or path.stat().st_size > 2**20:
            raise ValueError("Invalid aggregate report path/size")
        report = json.loads(path.read_text())
        if (report.get("status") != "complete" or report.get("input_sha256") != manifest_hash
                or report.get("script_sha256") != sha256(Path(__file__).with_name("count_photo_panels.py"))):
            raise ValueError("Aggregate input/code identity mismatch")
        source_index = report["source_index"]
        if source_index in seen or not 0 <= source_index < 8:
            raise ValueError("Duplicate/invalid source index")
        seen.add(source_index)
        source = manifest["sources"][source_index]
        image_path = Path(source["source_image"]).resolve(strict=True)
        if (report["source_sha256"] != source["source_sha256"]
                or Path(report["source_image"]).resolve(strict=True) != image_path
                or sha256(image_path) != source["source_sha256"]):
            raise ValueError("Photo source identity mismatch")
        for panel in report["panel_reports"]:
            panel_path = Path(panel["path"]).resolve(strict=True)
            if sha256(panel_path) != panel["sha256"]:
                raise ValueError("Panel report hash mismatch")
            metadata = json.loads(panel_path.read_text())
            if metadata["checkpoint_sha256"] != ORIGINAL_CHECKPOINT_SHA256:
                raise ValueError("Only the original checkpoint is allowed")
        density_path = (path.parent / report["density_file"]).resolve(strict=True)
        if (density_path.parent != path.parent or density_path.stat().st_size > 128*2**20
                or sha256(density_path) != report["density_sha256"]):
            raise ValueError("Full density hash/path/size mismatch")
        density = np.load(density_path, allow_pickle=False)
        w, h = source["full_effective_dimensions"]
        if density.dtype != np.float32 or density.shape != (h//2, w//2):
            raise ValueError("Full density geometry mismatch")
        counts = photo_count_summary(density)
        if counts["count"] != report["count"]:
            raise ValueError("Count does not match full density")
        points = calc_denpt(density)
        with Image.open(image_path) as image:
            if list(image.size) != source["source_dimensions"] or image.width*image.height > 70_000_000:
                raise ValueError("Unexpected source dimensions")
            view = image.crop((0, 0, w, h)).convert("RGB")
            view.thumbnail((850, 1500), Image.Resampling.LANCZOS)
        width, height = view.size
        overlay = view.copy()
        draw = ImageDraw.Draw(overlay)
        for x, y, _ in points:
            px, py = x*width/w, y*height/h
            draw.ellipse((px-1, py-1, px+1, py+1), fill=(255, 0, 210))
        positive = np.maximum(density, 0)
        maximum = float(positive.max())
        heat = np.zeros(density.shape, np.uint8) if maximum == 0 else np.rint(positive/maximum*255).astype(np.uint8)
        heat_view = Image.fromarray(heat).resize(view.size, Image.Resampling.BOX).convert("RGB")
        canvas = Image.new("RGB", (3*width, height+54), "#17212b")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), f"{source['source_name']} | {w}x{h} | densidad {counts['count']:.2f} | propuestas {len(points)}", fill="white")
        for i, (title, panel) in enumerate(zip(("Original: vista reducida", "Puntos automáticos", "Densidad: escala por foto"), (view, overlay, heat_view))):
            draw.text((i*width+5, 24), title, fill="white")
            canvas.paste(panel, (i*width, 43))
        png = f"foto-{source_index:02d}.jpg"
        canvas.save(output/png, quality=88, subsampling=0)
        if (output/png).stat().st_size > 4*2**20:
            raise ValueError("Photo preview exceeds file browser size limit")
        record = {"source_index": source_index, "source_name": source["source_name"],
            "source_sha256": source["source_sha256"], "source_dimensions": source["source_dimensions"],
            "effective_dimensions": [w, h], "count": counts["count"],
            "automatic_candidate_count": len(points), "cropped_right_bottom_px": source["cropped_right_bottom_px"],
            "aggregate_report": str(path), "aggregate_sha256": sha256(path),
            "preview": png, "preview_sha256": sha256(output/png)}
        records.append(record)
        rows.append(f"<section><h2>{html.escape(source['source_name'])}</h2><p>"
            f"Conteo por densidad: {counts['count']:.2f}. Puntos propuestos: {len(points)}. "
            f"Fuente {source['source_dimensions']}; área efectiva {[w,h]}; recorte de borde "
            f"{source['cropped_right_bottom_px']} píxeles derecha/abajo. Sin máscara de calle.</p>"
            f"<a href='{png}'>Ampliar vista</a><img src='{png}' alt='Original, propuestas y densidad'></section>")
        del density, positive, heat, points, canvas, view, overlay, heat_view
    page = ("<!doctype html><html lang='es'><meta charset='utf-8'><title>Fotos completas: modelo original</title>"
        "<style>body{font:16px system-ui;max-width:1500px;margin:2rem auto;background:#18232d;color:#eee}"
        "section{margin:2rem 0;border-top:1px solid #789}img{width:100%;height:auto}a{color:#9df}</style>"
        "<h1>Modelo original sobre las ocho fotos completas</h1><p>Inferencia nativa por bloques, "
        "sin entrenamiento ni etiquetas necesarias. Estas vistas están reducidas para mostrarlas; "
        "las entradas del modelo conservaron la resolución. Magenta: candidatos automáticos, no "
        "personas verificadas. La integral de densidad y el número de máximos son distintos.</p>"
        "<p>Incluye edificios y otras áreas: todavía no hay máscara de calle/vereda/plaza validada. "
        "Los valores no son cifras de asistencia. Las fotos se solapan: sus conteos no se suman. "
        "Cada mapa de brillo usa su propia escala. "
        "Los tiles/bandas con halo96 aproximan el contexto, sin garantizar igualdad con inferencia "
        "de toda la foto en una sola pasada.</p>"+"".join(rows)+"</html>")
    (output/"index.html").write_text(page)
    (output/"index.json").write_text(json.dumps({"input_sha256": manifest_hash,
        "script_sha256": sha256(__file__),
        "aggregation_code_sha256": sha256(Path(__file__).with_name("count_photo_panels.py")),
        "postprocessing_code_sha256": sha256(Path(__file__).with_name("render_automatic_review.py")),
        "photos": records,
        "index_html_sha256": sha256(output/"index.html")}, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--input-sha256", required=True)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    render(a.input, a.input_sha256, a.reports, a.output_dir)


if __name__ == "__main__":
    main()
