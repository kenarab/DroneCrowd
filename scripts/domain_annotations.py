"""Validate provisional head labels; export offline review and compatible MATs.

    python scripts/domain_annotations.py --manifest labels.json --sha256 HASH \
        --output-dir outputs/ajuste-dominio/revision-v1

Outputs contain private crops; no server or network is used. MAT files are new
artifacts with explicit supervision metadata, never replacements for dataset GT.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from domain_targets import supervision_mask, validate_points
from sample_video import sha256


def validate_manifest(data):
    if data.get("schema") != "dronecrowd-head-points-v1" or not data.get("annotation_provenance"):
        raise ValueError("Explicit annotation schema and provenance required")
    entries = data.get("entries", [])
    if not 1 <= len(entries) <= 20:
        raise ValueError("Select one to twenty annotated regions")
    groups, hashes, names = {}, {}, set()
    for entry in entries:
        name = entry["name"]
        if not name or name in names or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("Unique safe region names required")
        names.add(name)
        if entry["split"] not in ("train", "test"):
            raise ValueError("Train or test split required")
        group = entry["source_group"]
        digest = entry["source_sha256"]
        if not group or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Source group and SHA-256 required")
        for table, key in ((groups, group), (hashes, digest)):
            if key in table and table[key] != entry["split"]:
                raise ValueError("A source/video group or hash crossed train/test")
            table[key] = entry["split"]
        x, y, right, bottom = entry["box_xyxy"]
        if (any(type(v) is not int or v % 4 for v in (x, y, right, bottom))
                or x < 0 or y < 0 or not 96 <= min(right-x, bottom-y)
                or max(right-x, bottom-y) > 512):
            raise ValueError("Bounded ROI aligned to four required")
        width, height = right-x, bottom-y
        points = validate_points(entry["points_confirmed_xy"], width, height)
        uncertain = validate_points(entry["points_uncertain_xy"], width, height)
        # Exact duplicate clicks are mistakes; close but distinct heads are valid.
        combined = np.concatenate((points, uncertain))
        if len(set(map(tuple, combined))) != len(combined):
            raise ValueError("Duplicate annotation coordinates")
        supervision_mask(points, uncertain, width, height, entry["supervision"])
        if entry.get("review_status") not in ("agent_provisional", "user_reviewed"):
            raise ValueError("Explicit provisional/reviewed label status required")
    return entries


def mat_payload(entry):
    points = np.asarray(entry["points_confirmed_xy"], dtype=np.float64).reshape(-1, 2)
    locations = np.column_stack((points, np.arange(1, len(points)+1, dtype=np.float64)))
    info = np.empty((1, 1), dtype=[("location", object), ("number", object)])
    info[0, 0]["location"] = locations
    info[0, 0]["number"] = np.array([[len(points)]], dtype=np.int32)
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = info
    return {"image_info": cell, "annotation_metadata_json": json.dumps(entry, ensure_ascii=False)}


def export_review(data, output):
    entries = validate_manifest(data)
    output.mkdir(parents=True, exist_ok=False)
    images, file_rows = {}, []
    for entry in entries:
        source = Path(entry["source_image"]).resolve(strict=True)
        if output.resolve().is_relative_to(source.parent):
            raise ValueError("Output must not overlap source directory")
        before = source.stat()
        if not 0 < before.st_size <= 64*2**20 or sha256(source) != entry["source_sha256"]:
            raise ValueError("Source file size/hash mismatch")
        with Image.open(source) as picture:
            if picture.width*picture.height > 24_000_000:
                raise ValueError("Source image exceeds pixel budget")
            x, y, right, bottom = entry["box_xyxy"]
            if right > picture.width or bottom > picture.height:
                raise ValueError("ROI outside source")
            crop = picture.crop((x, y, right, bottom)).convert("RGB")
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            images[entry["name"]] = base64.b64encode(buffer.getvalue()).decode("ascii")
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed during preparation")
        mat = output/f"{entry['name']}-provisional.mat"
        savemat(mat, mat_payload(entry), do_compression=True)
        file_rows.append({"file": mat.name, "sha256": sha256(mat),
                          "supervision": entry["supervision"], "split": entry["split"]})
    template = Path(__file__).with_name("review_head_points.html").read_text()
    payload = json.dumps({"annotations": data, "images": images}, ensure_ascii=True).replace("<", "\\u003c")
    (output/"revisar-puntos.html").write_text(template.replace("/*ANNOTATION_DATA*/null", payload))
    (output/"annotations.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    (output/"export.json").write_text(json.dumps({"files": file_rows,
        "annotation_sha256": sha256(output/"annotations.json"),
        "script_sha256": sha256(__file__), "template_sha256": sha256(Path(__file__).with_name("review_head_points.html")),
        "warning": "Provisional local head IDs; partial/uncertain labels are not complete image GT."}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.stat().st_size > 256*1024 or sha256(args.manifest) != args.sha256:
        parser.error("Annotation manifest size/hash mismatch")
    export_review(json.loads(args.manifest.read_text()), args.output_dir.resolve())
    print(json.dumps({"output": str(args.output_dir), "sha256": args.sha256}))


if __name__ == "__main__":
    main()
