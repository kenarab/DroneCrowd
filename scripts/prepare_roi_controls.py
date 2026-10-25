"""Prepare frozen visual controls with context, preserving their source files."""
import argparse
import json
from pathlib import Path

from PIL import Image

from sample_video import memory_limits, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    references = args.references.resolve(strict=True)
    if references.stat().st_size > 256*1024 or sha256(references) != args.references_sha256:
        parser.error("Reference manifest size or hash mismatch")
    data = json.loads(references.read_text())
    if not 1 <= len(data["entries"]) <= 16:
        parser.error("Choose one to sixteen controls")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, reference in enumerate(data["entries"]):
        source = Path(reference["source_image"]).resolve(strict=True)
        if source.is_relative_to(output) or output.is_relative_to(source.parent):
            raise ValueError("Output overlaps source directory")
        if not 0 < source.stat().st_size <= 64*2**20 or sha256(source) != reference["source_sha256"]:
            raise ValueError("Source size or SHA-256 mismatch")
        before = source.stat()
        x, y, right, bottom = reference["box_xyxy"]
        if (any(type(v) is not int or v % 4 for v in (x,y,right,bottom))
                or not 32 <= right-x <= 512 or not 32 <= bottom-y <= 512):
            raise ValueError("ROI must be aligned and bounded to 32..512 per side")
        with Image.open(source) as picture:
            width, height = picture.size
            if not 0 <= x < right <= width or not 0 <= y < bottom <= height or width*height > 24_000_000:
                raise ValueError("ROI outside bounded source image")
            context = [max(0,x-128), max(0,y-128), min(width,right+128), min(height,bottom+128)]
            rgb = picture.crop(context).convert("RGB")
            rgb_width, rgb_height = rgb.size
            if rgb_width % 4 or rgb_height % 4:
                raise ValueError("Context geometry must align with density grid")
            payload = rgb.tobytes()
        for field in ("points_confirmed_xy", "points_uncertain_xy"):
            for px, py in reference[field]:
                if not 0 <= px < right-x or not 0 <= py < bottom-y:
                    raise ValueError("Reference point outside ROI")
        ppm = output/f"frame-{index:03d}.ppm"
        ppm.write_bytes(f"P6\n{rgb_width} {rgb_height}\n255\n".encode()+payload)
        entry = {**reference, "index":index, "width":rgb_width, "height":rgb_height,
                 "rgb_file":ppm.name, "rgb_sha256":sha256(ppm), "context_source_xyxy":context,
                 "roi_context_xyxy":[x-context[0],y-context[1],right-context[0],bottom-context[1]]}
        entries.append(entry)
        after = source.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError("Source changed during preparation")
    manifest = output/"controls.json"
    manifest.write_text(json.dumps({"reference_manifest":str(references),"reference_sha256":args.references_sha256,
        "annotation_provenance":data["annotation_provenance"],"selection_note":data["selection_note"],
        "entries":entries,"script_sha256":sha256(__file__),"memory_limits":memory_limits()},indent=2)+"\n")
    print(json.dumps({"controls":str(manifest),"sha256":sha256(manifest),"count":len(entries)}))


if __name__ == "__main__":
    main()
