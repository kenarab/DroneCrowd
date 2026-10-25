"""Render a private, offline review of original-model density peaks."""
import argparse
import html
import io
import json
import math
from pathlib import Path
from hashlib import sha256 as hash_bytes

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from render_extended_review import (MANIFEST_LIMIT, ORIGINAL_CHECKPOINT_SHA256,
    PNG_LIMIT, SOURCE_LIMIT, pending, same_roi, summary, verified_json)
from sample_video import sha256

EVAL_LIMIT = 4 *  2**20
ARRAY_LIMIT = 16 *  2**20
TEXT_LIMIT = 512 * 2**10
MAX_DIMENSION = 768


def calc_denpt(density):
    """STNNet/mytest.py calc_denpt postprocess, ratio=2, preserving its thresholds."""
    image = np.asarray(density, dtype=np.float32) / 100.0
    image = image.copy()
    image[image <= 0] = 0
    data_max = ndimage.maximum_filter(image, size=8)
    data_min = ndimage.minimum_filter(image, size=8)
    maxima = image == data_max
    maxima[(data_max - data_min) <= 0.001] = 0
    labels, _ = ndimage.label(maxima)
    points = []
    for ys, xs in ndimage.find_objects(labels):
        if ys is None or xs is None:
            continue
        x = (xs.start + xs.stop - 1) / 2 * 2
        y = (ys.start + ys.stop - 1) / 2 * 2
        points.append((float(x), float(y), float(image[int(y / 2), int(x / 2)])))
    return points


def candidates_in_roi(density, context_box, roi_box):
    """Find peaks on the full context, then apply the source ROI's half-open box."""
    cx, cy, _, _ = context_box
    x0, y0, x1, y1 = roi_box
    return [{"x": gx-x0, "y": gy-y0, "peak_height": peak}
            for px, py, peak in calc_denpt(density)
            for gx, gy in [(cx+px, cy+py)]
            if x0 <= gx < x1 and y0 <= gy < y1]


def verified_prediction(path, targets):
    path = Path(path).resolve(strict=True)
    if not 0 < path.stat().st_size <= EVAL_LIMIT:
        raise ValueError(f"Evaluation size invalid: {path}")
    digest = sha256(path)
    evaluation = verified_json(path, digest, EVAL_LIMIT)
    if (evaluation.get("status") != "complete" or evaluation.get("scale") != 1
            or evaluation.get("variant") != "density"
            or evaluation.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256):
        raise ValueError(f"Only complete scale-1 original density evaluation is allowed: {path}")
    controls_path = Path(evaluation["controls"]).resolve(strict=True)
    controls = verified_json(controls_path, evaluation["controls_sha256"], MANIFEST_LIMIT)
    rows = {row["name"]: row for row in evaluation.get("rows", [])}
    entries = {entry["name"]: entry for entry in controls.get("entries", [])}
    if len(rows) != len(evaluation.get("rows", [])) or len(entries) != len(controls.get("entries", [])):
        raise ValueError("Duplicate evaluation/control ROI names")
    matched, mismatches = {}, []
    for target in targets:
        control, row = entries.get(target["name"]), rows.get(target["name"])
        if control is None or row is None:
            continue
        if (not same_roi(target, control) or row.get("index") != control.get("index")
                or target.get("split") != control.get("split")
                or target.get("source_group") != control.get("source_group")):
            mismatches.append(target["name"])
            continue
        context_box = control.get("context_source_xyxy")
        roi_context = control.get("roi_context_xyxy")
        if (not isinstance(context_box, list) or len(context_box) != 4
                or not isinstance(roi_context, list) or len(roi_context) != 4):
            raise ValueError(f"Missing context geometry for {target['name']}")
        x0, y0, x1, y1 = target["box_xyxy"]
        cx, cy, cr, cb = context_box
        rx, ry, rr, rb = roi_context
        if (any(type(v) is not int for v in context_box + roi_context + target["box_xyxy"])
                or type(control.get("width")) is not int or type(control.get("height")) is not int
                or rx < 0 or ry < 0 or rr <= rx or rb <= ry or rr > control.get("width")
                or rb > control.get("height") or cr-cx != control.get("width")
                or cb-cy != control.get("height") or cx+rx != x0 or cy+ry != y0
                or cx+rr != x1 or cy+rb != y1 or rr-rx != x1-x0 or rb-ry != y1-y0):
            raise ValueError(f"ROI/context geometry mismatch for {target['name']}")
        arr = (Path(path.parent / row["density_file"])).resolve(strict=True)
        if not arr.is_relative_to(path.parent.resolve()) or arr.stat().st_size > ARRAY_LIMIT:
            raise ValueError(f"Unsafe or oversized density array: {arr}")
        raw = arr.read_bytes()
        if hash_bytes(raw).hexdigest() != row.get("density_sha256"):
            raise ValueError(f"Density SHA-256 mismatch: {arr}")
        density = np.load(io.BytesIO(raw), allow_pickle=False)
        del raw
        width, height = control["width"], control["height"]
        if (type(width) is not int or type(height) is not int or min(width, height) <= 0
                or max(width, height) > MAX_DIMENSION or density.shape != (height // 2, width // 2)
                or density.dtype.kind not in "fiu" or not np.isfinite(density).all()):
            raise ValueError(f"Invalid density dimensions or values: {arr}")
        candidates = candidates_in_roi(density, context_box, target["box_xyxy"])
        roi_heat = np.maximum(density.astype(np.float32) / 100.0, 0)
        roi_heat = roi_heat[ry//2:rb//2, rx//2:rr//2]
        scale = float(roi_heat.max()) if roi_heat.size else 0.0
        heatmap = (np.rint(255*roi_heat/scale).astype(np.uint8) if scale > 0
                   else np.zeros(roi_heat.shape, np.uint8))
        count = row.get("count")
        if not isinstance(count, (int, float)) or not math.isfinite(count):
            raise ValueError(f"Invalid density sum for {target['name']}")
        matched[target["name"]] = {"density_count": float(count),
            "candidates": candidates, "density_file": str(arr),
            "density_sha256": row["density_sha256"], "evaluation_sha256": digest,
            "controls_sha256": evaluation["controls_sha256"], "context_source_xyxy": context_box,
            "roi_context_xyxy": roi_context, "density_shape": list(density.shape),
            "heatmap": heatmap}
    meta = {"path": str(path), "sha256": digest, "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "controls": str(controls_path), "controls_sha256": evaluation["controls_sha256"]}
    return matched, mismatches, meta


def validate_source(entry, cache, hashes):
    source = Path(entry["source_image"]).resolve(strict=True)
    if source not in cache:
        before = source.stat()
        if not 0 < before.st_size <= SOURCE_LIMIT or sha256(source) != entry["source_sha256"]:
            raise ValueError(f"Source size/hash invalid: {source}")
        with Image.open(source) as im:
            cache[source] = im.convert("RGB")
        hashes[source] = entry["source_sha256"]
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while opening: {source}")
    elif hashes[source] != entry["source_sha256"]:
        raise ValueError("Conflicting hashes for same source path")
    x0, y0, x1, y1 = entry["box_xyxy"]
    im = cache[source]
    if any(type(x) is not int for x in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
        raise ValueError(f"Invalid ROI geometry: {entry['name']}")
    roi = im.crop((x0, y0, x1, y1))
    for key in ("points_confirmed_xy", "points_uncertain_xy"):
        for px, py in entry.get(key, []):
            if not (math.isfinite(px) and math.isfinite(py) and 0 <= px < roi.width and 0 <= py < roi.height):
                raise ValueError(f"Invalid human mark in {entry['name']}")
    return source, roi


def render(entry, roi, prediction, output):
    w, h = roi.size
    human, auto = roi.copy(), roi.copy()
    for canvas, key, color, prefix in ((human, "points_confirmed_xy", (35, 210, 90), "C"),
            (human, "points_uncertain_xy", (255, 145, 35), "D")):
        draw = ImageDraw.Draw(canvas)
        for i, (x, y) in enumerate(entry.get(key, []), 1):
            x, y = round(x), round(y); draw.ellipse((x-5,y-5,x+5,y+5), outline=color, width=1)
            draw.text((x+6,y-6), f"{prefix}{i}", fill=color, stroke_width=1, stroke_fill="black")
    draw = ImageDraw.Draw(human)
    for i, presence in enumerate(entry.get("presence_only", []), 1):
        x, y = presence["approximate_xy"]
        x, y = round(x), round(y)
        draw.rectangle((x-5,y-5,x+5,y+5), outline=(70,210,255), width=1)
        draw.text((x+6,y-6), f"P{i}", fill=(70,210,255), stroke_width=1, stroke_fill="black")
    draw = ImageDraw.Draw(auto)
    for i, point in enumerate(prediction["candidates"], 1):
        x, y = round(point["x"]), round(point["y"])
        draw.ellipse((x-5,y-5,x+5,y+5), outline=(255,0,210), width=1)
        draw.text((x+6,y-6), f"A{i}", fill=(255,0,210), stroke_width=1, stroke_fill="black")
    heat = Image.fromarray(np.asarray(prediction["heatmap"], dtype=np.uint8), "L").convert("RGB")
    heat = heat.resize((w,h), Image.Resampling.NEAREST)
    panels = (roi, human, auto, heat)
    canvas = Image.new("RGB", (4*w, h+64), "#17212b")
    draw = ImageDraw.Draw(canvas)
    x0,y0,x1,y1 = entry["box_xyxy"]
    title = f"{entry['name']} | {Path(entry['source_image']).name} | {entry['split']} | source ROI ({x0},{y0},{x1},{y1})"
    draw.text((4,4), title[:180], fill="white")
    marked = summary(entry)
    human_text = "marcas pendientes" if pending(entry) else f"marcas nominales {marked['nominal']} (dudosas {marked['uncertain']})"
    if entry.get("presence_only"):
        human_text += "; P1 presencia confirmada por autor, cabeza sin localizar (no es centro/GT)"
    if entry.get("review_dispute"):
        human_text += "; presencia cuestionada por autor, pendiente de confirmación conjunta"
    draw.text((4,22), f"{human_text}; conteo densidad {prediction['density_count']:.3f}; puntos propuestos {len(prediction['candidates'])}", fill="white")
    for i,(name,panel) in enumerate(zip(("original","marcas de referencia","propuestas automáticas","densidad normalizada/ROI"),panels)):
        canvas.paste(panel, (i*w, 48)); draw.text((i*w+3,32), name, fill="white")
    draw.text((4,h+49), "Auto: máximos locales postproceso STNNet; escala de calor normalizada por ROI; picos no son probabilidades/personas confirmadas.", fill="white")
    canvas.save(output, format="PNG", optimize=True)
    if output.stat().st_size > PNG_LIMIT: raise ValueError("Per-ROI PNG exceeds 4 MiB")


def build(annotations, selection, eval_paths, out):
    old, new = annotations.get("entries", []), selection.get("entries", [])
    if len(old) != 7 or len(new) != 12: raise ValueError("Expected 7 annotated + 12 pending regions")
    for e in new:
        if not pending(e): raise ValueError("All 12 new selections must remain pending")
    entries = old + new
    if len({e["name"] for e in entries}) != 19: raise ValueError("Expected 19 unique regions")
    preds, meta, mismatches = {}, [], []
    for path in eval_paths:
        found, skipped, info = verified_prediction(path, entries)
        duplicates = set(preds).intersection(found)
        if duplicates: raise ValueError(f"ROI has multiple matched original evaluations: {sorted(duplicates)}")
        preds.update(found); mismatches += skipped; meta.append(info)
    missing = [e["name"] for e in entries if e["name"] not in preds]
    if missing: raise ValueError(f"No identity-verified original evaluation for: {missing}")
    out = Path(out).resolve()
    if out != Path("revision/automatico-v4").resolve():
        raise ValueError("Output directory must be revision/automatico-v4/")
    out.mkdir(parents=True, exist_ok=False)
    cache, hashes, records, rows = {}, {}, [], []
    try:
        for i,e in enumerate(entries,1):
            source, roi = validate_source(e, cache, hashes)
            pred = preds[e["name"]]
            png = f"roi-{i:02d}.png"; render(e,roi,pred,out/png)
            record = {k:v for k,v in pred.items() if k not in ("heatmap", "density_file")}
            record.update({"name":e["name"],"source_image":str(source),"source_sha256":e["source_sha256"],
                "box_xyxy":e["box_xyxy"],"split":e["split"],"source_group":e.get("source_group"),
                "supervision":e.get("supervision"),
                "presence_only":e.get("presence_only",[]),"review_dispute":e.get("review_dispute"),
                "human_marks":summary(e),"automatic_candidates":pred["candidates"],
                "automatic_candidate_count":len(pred["candidates"]),"png":png,"png_sha256":sha256(out/png)})
            records.append(record)
            status = "pendiente" if pending(e) else f"marcas {record['human_marks']['nominal']} (dudosas {record['human_marks']['uncertain']})"
            if e.get("presence_only"): status += "; presencia autor, centro no localizado"
            if e.get("review_dispute"): status += "; presencia en disputa, revisión pendiente"
            rows.append(f"<tr><td>{html.escape(e['name'])}<br>{html.escape(e['split'])}<br>{status}<br>Conteo por densidad: {pred['density_count']:.3f}; puntos propuestos: {len(pred['candidates'])}</td><td><a href='{png}'>PNG ampliable</a><img src='{png}' alt='{html.escape(e['name'])}'></td></tr>")
            del pred["heatmap"]
        if any(sha256(p)!=h for p,h in hashes.items()): raise ValueError("Source changed during render")
        page = ("<!doctype html><html lang='es'><meta charset='utf-8'><title>Revisión automática</title>"
            "<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;background:#18232d;color:#eee}"
            "table{border-collapse:collapse;width:100%}td{border:1px solid #667;padding:.5rem;vertical-align:top}"
            "img{display:block;width:100%;height:auto;image-rendering:pixelated}</style>"
            "<h1>Propuestas automáticas del modelo original</h1>"
            "<p>Magenta: puntos propuestos automáticamente. Verde: marcas de referencia todavía parciales. "
            "Celeste P1: presencia indicada por el autor, con cabeza sin localizar. Los puntos automáticos "
            "pueden ayudar a revisar; todavía no son personas confirmadas. Su cantidad y el conteo por "
            "densidad son medidas distintas. Las regiones pendientes no equivalen a cero.</p>"
            "<p>Cada imagen tiene cuatro paneles: original, referencias, propuestas automáticas y mapa "
            "de densidad. El brillo del mapa se normaliza por región: no se compara entre imágenes.</p>"
            "<table>"+"".join(rows)+"</table><details><summary>Procedimiento reproducible</summary>"
            "<p>Se aplica calc_denpt de STNNet/mytest.py sin calibración nueva: mapa/100, filtros max/min8, "
            "contraste mayor que0,001 y centros de componentes. Los máximos no son probabilidades "
            "calibradas ni etiquetas para entrenar.</p></details></html>")
        b=page.encode();
        if len(b)>TEXT_LIMIT: raise ValueError("HTML exceeds 512 KiB")
        (out/"index.html").write_bytes(b)
        index={"annotations_sha256":annotations["_sha256"],"selection_sha256":selection["_sha256"],
            "script_sha256":sha256(__file__),
            "historical_postprocess_source_sha256":sha256(Path(__file__).resolve().parents[1]/"STNNet/mytest.py"),
            "identity_helpers_sha256":sha256(Path(__file__).with_name("render_extended_review.py")),
            "evaluations":meta,"geometry_mismatches":mismatches,
            "sources":{str(k):v for k,v in hashes.items()},"regions":records,"index_html_sha256":sha256(out/"index.html")}
        b=(json.dumps(index,indent=2,ensure_ascii=False)+"\n").encode()
        if len(b)>TEXT_LIMIT: raise ValueError("Index exceeds 512 KiB")
        (out/"index.json").write_bytes(b)
    finally:
        for im in cache.values(): im.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",type=Path,required=True); p.add_argument("--annotations-sha256",required=True)
    p.add_argument("--selection12",type=Path,required=True); p.add_argument("--selection-sha256",required=True)
    p.add_argument("--evaluations",type=Path,nargs="+",required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    ann=verified_json(a.annotations,a.annotations_sha256,256*1024); sel=verified_json(a.selection12,a.selection_sha256,256*1024)
    if ann.get("schema")!="dronecrowd-head-points-v1": p.error("Unexpected annotation schema")
    ann["_sha256"]=a.annotations_sha256; sel["_sha256"]=a.selection_sha256
    build(ann,sel,a.evaluations,a.output_dir)


if __name__ == "__main__": main()
