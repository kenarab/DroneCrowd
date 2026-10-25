"""Render an offline, provenance-checked review of annotated and pending ROIs."""
import argparse
import html
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw

from sample_video import sha256

MANIFEST_LIMIT = 1024 * 1024
SOURCE_LIMIT = 64 * 2**20
EVALUATION_LIMIT = 4 * 2**20
PNG_LIMIT = 4 * 2**20
HTML_LIMIT = 512 * 2**10
ORIGINAL_CHECKPOINT_SHA256 = "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d"


def verified_json(path, digest, limit):
    path = Path(path).resolve(strict=True)
    if not 0 < path.stat().st_size <= limit or sha256(path) != digest:
        raise ValueError(f"Size or SHA-256 mismatch: {path}")
    return json.loads(path.read_text())


def pending(entry):
    return (entry.get("points_empty") is True
            or entry.get("review_status") == "pending_joint_review"
            or entry.get("supervision") == "unannotated")


def summary(entry):
    if pending(entry):
        return {"status": "pending", "nominal": None, "uncertain": None}
    confirmed, uncertain = entry.get("points_confirmed_xy", []), entry.get("points_uncertain_xy", [])
    return {"status": "marked", "nominal": len(confirmed) + len(uncertain),
            "uncertain": len(uncertain)}


def same_roi(target, control):
    try:
        return (target["name"] == control["name"]
                and Path(target["source_image"]).resolve(strict=True)
                    == Path(control["source_image"]).resolve(strict=True)
                and target["source_sha256"] == control["source_sha256"]
                and list(target["box_xyxy"]) == list(control["box_xyxy"]))
    except (KeyError, OSError, TypeError):
        return False


def evaluation_label(evaluation, path):
    checkpoint = str(evaluation.get("checkpoint", "")).lower()
    context = (checkpoint + " " + str(path)).lower()
    digest = evaluation.get("checkpoint_sha256")
    if digest == ORIGINAL_CHECKPOINT_SHA256:
        return "original"
    if "conv4" in context:
        return "ajuste conv4 rechazado · solo referencia"
    if "mixto-head" in context or "den-domain-pilot" in context:
        return "ajuste head rechazado · solo referencia"
    return "checkpoint no promovido · identidad no clasificada"


def predictions_for(path, targets):
    path = Path(path).resolve(strict=True)
    if not 0 < path.stat().st_size <= EVALUATION_LIMIT:
        raise ValueError(f"Evaluation exceeds 4 MiB: {path}")
    evaluation_hash = sha256(path)
    evaluation = verified_json(path, evaluation_hash, EVALUATION_LIMIT)
    if evaluation.get("status") != "complete":
        raise ValueError(f"Evaluation is not complete: {path}")
    if evaluation.get("scale") != 1 or evaluation.get("variant") != "density":
        raise ValueError(f"Review requires native scale 1 density evaluations: {path}")
    checkpoint_hash = evaluation.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash):
        raise ValueError(f"Evaluation lacks a valid checkpoint hash: {path}")
    controls_path = Path(evaluation["controls"]).resolve(strict=True)
    controls = verified_json(controls_path, evaluation["controls_sha256"], MANIFEST_LIMIT)
    rows_list, control_list = evaluation.get("rows", []), controls.get("entries", [])
    rows = {row["name"]: row for row in rows_list}
    control_entries = {entry["name"]: entry for entry in control_list}
    if len(rows) != len(rows_list) or len(control_entries) != len(control_list):
        raise ValueError(f"Duplicate evaluation or control names: {path}")
    label = evaluation_label(evaluation, path)
    result, mismatches = {}, []
    for target in targets:
        control, row = control_entries.get(target["name"]), rows.get(target["name"])
        if control is None or row is None:
            continue
        if (not same_roi(target, control) or row.get("index") != control.get("index")):
            mismatches.append(target["name"])
            continue
        count = row.get("count")
        if isinstance(count, (int, float)) and math.isfinite(count):
            result[target["name"]] = {"label": label, "count": float(count),
                "evaluation": str(path), "checkpoint": evaluation.get("checkpoint"),
                "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
                "control_sha256": evaluation["controls_sha256"]}
    return result, mismatches, {"path": str(path), "sha256": evaluation_hash, "label": label,
        "checkpoint": evaluation.get("checkpoint"), "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "status": "complete", "control_manifest": str(controls_path),
        "control_manifest_sha256": evaluation["controls_sha256"]}


def validate_entry(entry, cache, source_hashes):
    source = Path(entry["source_image"]).resolve(strict=True)
    if source not in cache:
        before = source.stat()
        if not 0 < before.st_size <= SOURCE_LIMIT or sha256(source) != entry["source_sha256"]:
            raise ValueError(f"Source size or SHA-256 mismatch: {source}")
        with Image.open(source) as image:
            cache[source] = image.convert("RGB")
        source_hashes[source] = entry["source_sha256"]
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while opening: {source}")
    elif source_hashes[source] != entry["source_sha256"]:
        raise ValueError(f"One source path has conflicting hashes: {source}")
    x, y, right, bottom = entry["box_xyxy"]
    if (any(type(v) is not int for v in (x, y, right, bottom)) or x < 0 or y < 0
            or right <= x or bottom <= y or right > cache[source].width or bottom > cache[source].height):
        raise ValueError(f"ROI outside source image: {entry['name']}")
    roi = cache[source].crop((x, y, right, bottom))
    for field in ("points_confirmed_xy", "points_uncertain_xy"):
        for px, py in entry.get(field, []):
            if (not math.isfinite(px) or not math.isfinite(py)
                    or not 0 <= px < roi.width or not 0 <= py < roi.height):
                raise ValueError(f"Invalid ROI-local point in {entry['name']}")
    return source, roi, cache[source].size, roi.size


def render_roi(roi, entry, destination):
    canvas = Image.new("RGB", (roi.width*2, roi.height), "#17212b")
    canvas.paste(roi, (0, 0))
    marked = roi.copy()
    draw = ImageDraw.Draw(marked)
    for field, color, prefix in (("points_confirmed_xy", (35, 205, 90), "N"),
                                 ("points_uncertain_xy", (255, 145, 35), "D")):
        for index, (x, y) in enumerate(entry.get(field, []), 1):
            px, py = round(x), round(y)
            radius = 5
            draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline=color, width=1)
            draw.text((px+radius+1, py-radius), f"{prefix}{index}", fill=color, stroke_width=1,
                      stroke_fill=(0, 0, 0))
    canvas.paste(marked, (roi.width, 0))
    canvas.save(destination, format="PNG", optimize=True)
    if destination.stat().st_size > PNG_LIMIT:
        raise ValueError(f"Review PNG exceeds 4 MiB: {destination.name}")
    return 1.0


def build_review(annotations, selection, evaluation_paths, output):
    old_entries, new_entries = annotations["entries"], selection["entries"]
    if len(old_entries) != 7 or len(new_entries) != 12:
        raise ValueError("Expected seven annotated regions and twelve selected pending regions")
    for entry in new_entries:
        if not pending(entry) or entry.get("reference_interval") is not None:
            raise ValueError("New selection entries must remain pending without reference intervals")
    all_entries = old_entries + new_entries
    names = [entry["name"] for entry in all_entries]
    if len(set(names)) != len(names):
        raise ValueError("ROI names must be unique across both manifests")
    predictions, sources, mismatches, eval_meta = {name: [] for name in names}, {}, [], []
    for eval_path in evaluation_paths:
        found, skipped, metadata = predictions_for(eval_path, all_entries)
        mismatches.extend({"evaluation": str(eval_path), "name": name} for name in skipped)
        eval_meta.append(metadata)
        for name, value in found.items():
            predictions[name].append(value)
    output = Path(output).resolve()
    if not output.is_relative_to(Path("revision").resolve()):
        raise ValueError("Versioned review output must be under revision/")
    output.mkdir(parents=True, exist_ok=False)
    cache, source_hashes, records, html_rows = {}, {}, [], []
    try:
        for index, entry in enumerate(all_entries, 1):
            source, roi, source_size, roi_size = validate_entry(entry, cache, source_hashes)
            png = f"roi-{index:02d}.png"
            render_roi(roi, entry, output/png)
            marked = summary(entry)
            pred_rows = predictions[entry["name"]]
            records.append({"name": entry["name"], "source": str(source),
                "source_sha256": entry["source_sha256"], "box_xyxy": entry["box_xyxy"],
                "source_dimensions": list(source_size), "roi_dimensions": list(roi_size),
                "split": entry["split"], "supervision": entry.get("supervision", "pending"),
                "marks": marked, "reference_interval": entry.get("reference_interval"),
                "predictions": pred_rows, "png": png, "png_sha256": sha256(output/png),
                "display_pixels": "native ROI pixels"})
            mark_text = ("marcas: pendiente de anotación conjunta" if marked["status"] == "pending" else
                f"marcas nominales: {marked['nominal']}; dudosas: {marked['uncertain']}")
            if entry.get("supervision") == "partial_positive":
                mark_text += " · etiqueta parcial, no representa total de personas"
            elif entry.get("supervision") == "negative":
                mark_text += " · negativa provisional; las marcas no son un conteo total"
            interval = entry.get("reference_interval")
            interval_text = "pendiente" if interval is None else f"referencia provisional {html.escape(str(interval))}"
            source_note = []
            if entry.get("user_confirmation_scope"):
                source_note.append("Alcance confirmado por usuario: " + str(entry["user_confirmation_scope"]))
            for addition in entry.get("agent_additions", []) or []:
                source_note.append(f"Marca añadida por agente N{addition.get('point_index_1based')}: "
                    f"{addition.get('xy')} · {addition.get('evidence', '')}")
            source_note_text = "<br>".join(html.escape(note) for note in source_note)
            prediction_text = "<br>".join(f"{html.escape(p['label'])}: {p['count']:.3f} personas por suma de densidad"
                for p in pred_rows) or "sin predicción validada para esta geometría"
            html_rows.append(f"<tr><th>{html.escape(entry['name'])}</th><td>{html.escape(entry['split'])}</td>"
                f"<td>{html.escape(mark_text)}<br>referencia: {interval_text}<br>fuente: {source_note_text}</td>"
                f"<td>fuente {source_size[0]}×{source_size[1]} px; ROI {roi_size[0]}×{roi_size[1]} px</td>"
                f"<td>{prediction_text}</td></tr><tr><td colspan=\"5\"><a href=\"{png}\">Abrir PNG nativo</a> · "
                f"<img class=\"review\" src=\"{png}\" alt=\"ROI original izquierda; marcas derecha\"></td></tr>")
        eval_list = "".join(f"<li>{html.escape(e['label'])}: {html.escape(str(e['checkpoint']))} · "
            f"{html.escape(e['checkpoint_sha256'] or '')}</li>" for e in eval_meta)
        page = ("<!doctype html><html lang=\"es\"><meta charset=\"utf-8\"><title>Revisión ampliada</title>"
            "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#18232d;color:#eee}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #667;padding:.5rem;vertical-align:top}"
            ".review{display:block;width:100%;height:auto;image-rendering:pixelated}"
            "small{color:#ccd}</style><h1>Revisión ampliada de regiones</h1>"
            "<p>El contexto de inferencia incorpora hasta 128 px alrededor del ROI. La cifra es suma de densidad, "
            "no detección individual ni ground truth. Los ajustes rechazados se muestran solo como referencia y no se promueven. "
            "Las imágenes muestran el ROI original a la izquierda y marcas a la derecha. El PNG conserva todos los píxeles "
            "nativos; el navegador puede ampliar la vista con nearest. Las fuentes permanecen intactas.</p>"
            f"<h2>Proveniencia de evaluaciones</h2><ul>{eval_list}</ul><table><thead><tr>"
            "<th>ROI</th><th>Split</th><th>Marcas y referencia</th><th>Dimensiones</th><th>Predicciones</th>"
            f"</tr></thead><tbody>{''.join(html_rows)}</tbody></table></html>")
        payload = page.encode("utf-8")
        if len(payload) > HTML_LIMIT:
            raise ValueError("Static review HTML exceeds 512 KiB")
        (output/"index.html").write_bytes(payload)
        if any(sha256(path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Source changed during rendering")
        index = {"annotations_sha256": annotations["_sha256"], "selection_sha256": selection["_sha256"],
            "script_sha256": sha256(__file__), "evaluations": eval_meta,
            "geometry_mismatches": mismatches, "sources": {str(path): digest for path, digest in source_hashes.items()},
            "regions": records, "index_html_sha256": sha256(output/"index.html")}
        index_bytes = (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if len(index_bytes) > HTML_LIMIT:
            raise ValueError("Review index JSON exceeds 512 KiB")
        (output/"index.json").write_bytes(index_bytes)
    finally:
        for image in cache.values():
            image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--selection", "--selection12", dest="selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--evaluations", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.evaluations) > 4:
        parser.error("At most four evaluation manifests are supported")
    annotations = verified_json(args.annotations, args.annotations_sha256, MANIFEST_LIMIT)
    selection = verified_json(args.selection, args.selection_sha256, MANIFEST_LIMIT)
    if annotations.get("schema") != "dronecrowd-head-points-v1":
        parser.error("Unexpected annotations schema")
    annotations["_sha256"], selection["_sha256"] = args.annotations_sha256, args.selection_sha256
    build_review(annotations, selection, args.evaluations, args.output_dir)


if __name__ == "__main__":
    main()
