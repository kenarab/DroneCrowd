"""Prepare a private static review folder for an authenticated file browser."""
import argparse
import json
from pathlib import Path
import shutil

from render_annotation_review import crop_image, file_hash
from sample_video import sha256


def prepare(manifest, expected_hash, sheets, output):
    digest = file_hash(manifest, 256 * 1024)
    if digest != expected_hash:
        raise ValueError("Annotation manifest hash mismatch")
    annotations = json.loads(manifest.read_text())
    if annotations.get("schema") != "dronecrowd-head-points-v1":
        raise ValueError("Unexpected annotation schema")
    entries = annotations["entries"]
    names = [entry["name"] for entry in entries]
    if len(entries) != 7 or len(set(names)) != 7:
        raise ValueError("Expected seven distinct review regions")
    if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
           for name in names):
        raise ValueError("Unsafe region name")
    index = json.loads((sheets / "index.json").read_text())
    if index["manifest_sha256"] != digest:
        raise ValueError("Review sheets belong to another annotation manifest")
    for name in ("personas.png", "negativos.png"):
        if file_hash(sheets / name, 4 * 2**20) != index["png_sha256"][name]:
            raise ValueError("Review sheet hash mismatch")
    if output.resolve() != Path("revision").resolve():
        raise ValueError("The curated browser folder must be revision/ at repository root")
    output.mkdir(exist_ok=False)
    for name in ("personas.png", "negativos.png"):
        shutil.copyfile(sheets / name, output / name)
    lines = ["# DroneCrowd: revisar anotaciones", "",
        "Siete recortes provisionales: 22 marcas y 9 candidatos dudosos. "
        "Todavía no tienen aprobación humana ni constituyen fotos exhaustivamente anotadas.", "",
        "## Personas", "", "Izquierda: recorte original. Derecha: verde = cabeza marcada; "
        "naranja D = dudosa. Abrí la imagen para ampliarla.", "",
        "![Personas y puntos provisionales](personas.png)", "",
        "## Negativos", "", "Cuatro regiones sin personas discernibles, pendientes de revisión.", "",
        "![Regiones negativas provisionales](negativos.png)", "",
        "## Recortes originales", "",
        "| Región | División | Marcas | Dudas | Imagen sin marcas |",
        "| --- | --- | ---: | ---: | --- |"]
    for entry in entries:
        image, _, _ = crop_image(entry)
        name = entry["name"]
        image.save(output / f"{name}.png")
        lines.append(f"| {name} | {entry['split']} | {len(entry['points_confirmed_xy'])} | "
                     f"{len(entry['points_uncertain_xy'])} | [Ampliar]({name}.png) |")
    lines += ["", "Para proponer correcciones, indicar región y número de punto, "
        "o señalar una cabeza omitida. Ver las imágenes no aprueba las etiquetas. "
        "Las correcciones se guardarán en una versión nueva.", "",
        "Este visor es de sólo lectura. El editor offline existente sirve para "
        "editar y exportar puntos; su ejecución dentro del explorador no está habilitada.", "",
        f"Manifiesto SHA-256: `{digest}`.", ""]
    (output / "index.md").write_text("\n".join(lines))
    report = {"manifest_sha256": digest, "script_sha256": sha256(__file__),
              "review_sheet_index_sha256": sha256(sheets / "index.json"),
              "files": {p.name: sha256(p) for p in sorted(output.iterdir())}}
    (output / "index.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--sheets", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.annotations, args.annotations_sha256, args.sheets, Path("revision"))


if __name__ == "__main__":
    main()
