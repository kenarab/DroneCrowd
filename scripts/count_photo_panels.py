"""Prepare private RGB panels and infer one panel at a time with original STNNet."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from sample_video import sha256

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "outputs/ajuste-dominio/fotos-completas"
ORIGINAL_CHECKPOINT_SHA256 = "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d"
CORE_HEIGHT, HALO = 2048, 96
TILE_WIDTH, TILE_HEIGHT = 320, 128
MAX_PANEL_PIXELS = 24_000_000
MAX_OUTPUT_BYTES = 2 * 2**30
MAX_MANIFEST_BYTES = 2**20


def dimension_plan(width, height):
    if type(width) is not int or type(height) is not int or width < 32 or height < 32:
        raise ValueError("Photo dimensions must be integers at least 32x32")
    cropped = width // 4 * 4, height // 4 * 4
    # Model phase starts stay on an eight-pixel grid; do not invent edge pixels.
    padded = (0, 0)
    effective = cropped
    if min(effective) < 32 or max(effective) > 16384:
        raise ValueError("Effective dimensions are too small")
    return {"crop_dimensions":list(cropped),"effective_dimensions":list(effective),
            "cropped_right_bottom_px":[width-cropped[0],height-cropped[1]],
            "padded_right_bottom_px":list(padded)}


def effective_dimensions(width, height):
    return tuple(dimension_plan(width,height)["effective_dimensions"])


def band_geometry(width, height, core_height=CORE_HEIGHT, halo=HALO):
    width, height = effective_dimensions(width, height)
    if core_height <= 0 or core_height % 8 or halo < 0 or halo % 8:
        raise ValueError("Core height and halo must align to eight pixels")
    bands = []
    for y0 in range(0, height, core_height):
        y1 = min(y0 + core_height, height)
        cy0, cy1 = max(0, y0 - halo), min(height, y1 + halo)
        core = [0, y0, width, y1]
        context = [0, cy0, width, cy1]
        if width * (cy1 - cy0) > MAX_PANEL_PIXELS:
            raise ValueError("Band context exceeds the 24-million-pixel cap")
        bands.append({"core_xyxy": core, "context_xyxy": context,
                      "width": width, "height": cy1-cy0,
                      "core_local_xyxy": [0, y0-cy0, width, y1-cy0]})
    coverage=np.zeros(height,dtype=np.uint8)
    for band in bands:
        _,y0,_,y1=band["core_xyxy"]
        if coverage[y0:y1].any(): raise ValueError("Band cores overlap")
        coverage[y0:y1]+=1
    if not np.all(coverage==1): raise ValueError("Band cores do not cover effective source height")
    return bands


def photo_tile_windows(width, height):
    """Photo-specific bounded core/context tiles; does not alter video limits."""
    if (type(width) is not int or type(height) is not int or width < 32 or height < 32
            or width % 4 or height % 4 or width * height > MAX_PANEL_PIXELS):
        raise ValueError("Photo panel must align to four and fit 24 million pixels")
    for y in range(0, height, TILE_HEIGHT):
        for x in range(0, width, TILE_WIDTH):
            right, bottom = min(x+TILE_WIDTH, width), min(y+TILE_HEIGHT, height)
            context = (max(0,x-HALO), max(0,y-HALO), min(width,right+HALO), min(height,bottom+HALO))
            yield ((x,y,right,bottom), context)


def assemble_panel(rgb, predict, deadline=None):
    height, width = rgb.shape[:2]
    if rgb.dtype != np.uint8 or rgb.shape != (height,width,3):
        raise ValueError("Panel must be RGB uint8")
    density = np.empty((height//2,width//2), dtype=np.float32)
    coverage = np.zeros(density.shape, dtype=np.uint8)
    tiles = 0
    for (x,y,right,bottom), (cx,cy,cr,cb) in photo_tile_windows(width,height):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Panel inference exceeded 120 seconds")
        if cr-cx > 512 or cb-cy > 320:
            raise ValueError("Model input tile exceeds 512x320")
        result = predict(rgb[cy:cb,cx:cr])
        if (not isinstance(result,np.ndarray) or result.dtype != np.float32
                or result.shape != ((cb-cy)//2,(cr-cx)//2) or not np.isfinite(result).all()):
            raise ValueError("Model returned invalid tile density")
        gx,gy,gr,gb = x//2,y//2,right//2,bottom//2
        sx,sy = (x-cx)//2,(y-cy)//2
        tile = result[sy:sy+(gb-gy),sx:sx+(gr-gx)]
        if tile.shape != (gb-gy,gr-gx) or coverage[gy:gb,gx:gr].any():
            raise ValueError("Tile cores overlap or have invalid geometry")
        density[gy:gb,gx:gr] = tile
        coverage[gy:gb,gx:gr] += 1
        tiles += 1
    if not np.all(coverage == 1):
        raise ValueError("Tile cores do not cover each panel pixel exactly once")
    return density, tiles


def stitch_photo_cores(width,height,cores):
    if width%4 or height%4 or width*height>70_000_000:
        raise ValueError("Invalid full effective photo geometry")
    full=np.zeros((height//2,width//2),dtype=np.float32)
    coverage=np.zeros(full.shape,dtype=np.uint8)
    for box,core in cores:
        x0,y0,x1,y1=box
        if (any(type(v)is not int for v in box) or x0%2 or y0%2 or x1%2 or y1%2
                or not (0<=x0<x1<=width and 0<=y0<y1<=height)
                or core.dtype!=np.float32 or core.shape!=((y1-y0)//2,(x1-x0)//2)
                or not np.isfinite(core).all()):
            raise ValueError("Invalid core density geometry/data")
        x0,y0,x1,y1=x0//2,y0//2,x1//2,y1//2
        if coverage[y0:y1,x0:x1].any(): raise ValueError("Photo cores overlap")
        full[y0:y1,x0:x1]=core; coverage[y0:y1,x0:x1]+=1
    if not np.all(coverage==1): raise ValueError("Photo cores do not cover every effective pixel once")
    return full


def photo_count_summary(full):
    if full.ndim!=2 or not np.isfinite(full).all() or full.shape[0]%2 or full.shape[1]%2:
        raise ValueError("Invalid full photo density map")
    hh,hw=full.shape[0]//2,full.shape[1]//2
    quadrants=(full[:hh,:hw],full[:hh,hw:],full[hh:,:hw],full[hh:,hw:])
    sums=[float(q.sum(dtype=np.float32)) for q in quadrants]
    negatives=full[full<0]
    return {"raw_total":float(full.sum(dtype=np.float64)),
        "negative_mass_signed":float(negatives.sum(dtype=np.float64)),
        "negative_mass_abs":float(-negatives.sum(dtype=np.float64)),
        "negative_pixel_count":int(len(negatives)),"raw_quadrant_sums":sums,
        "count":sum(max(0.0,value) for value in sums)/100.0}


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError(f"Private output must be under {PRIVATE_ROOT}")
    return path


def write_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest/report exceeds one MiB")
    path.write_bytes(data)


def prepare(source_dir, output_dir):
    from PIL import Image

    source_dir = Path(source_dir).resolve(strict=True)
    output = safe_output(output_dir)
    if output.exists() or output.is_relative_to(source_dir) or source_dir.is_relative_to(output):
        raise ValueError("Output must be new and separate from photo sources")
    photos = sorted((p.resolve(strict=True) for p in source_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")), key=lambda p:p.name.lower())
    if len(photos) != 8:
        raise ValueError(f"Expected exactly eight JPEG photos, found {len(photos)}")
    inventory, planned, panel_specs = [], 0, []
    for image_path in photos:
        before = image_path.stat()
        if before.st_size>64*2**20: raise ValueError(f"Source JPEG exceeds 64 MiB: {image_path}")
        digest = sha256(image_path)
        with Image.open(image_path) as image:
            width,height = image.size
            exif_orientation=image.getexif().get(274,1)
            if width*height>70_000_000: raise ValueError(f"Source exceeds 70 million pixels: {image_path}")
            plan=dimension_plan(width,height)
            effective_w,effective_h = effective_dimensions(width,height)
            bands = band_geometry(width,height)
        inventory.append({"source_image":str(image_path),"source_name":image_path.name,
            "source_sha256":digest,"source_dimensions":[width,height],
            "exif_orientation":exif_orientation,
            "orientation_policy":"RGB stored pixels; no automatic EXIF transposition",
            **plan,"full_effective_dimensions":plan["effective_dimensions"],
            "source_stat":[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns]})
        for band_index,band in enumerate(bands):
            spec={"panel_index":len(panel_specs),"image_index":len(inventory)-1,
                  "band_index":band_index,"band_count":len(bands),**band}
            crop_w,crop_h=plan["crop_dimensions"]
            x0,y0,x1,y1=band["core_xyxy"]
            cx0,cy0,cx1,cy1=band["context_xyxy"]
            spec["source_core_xyxy"]=[min(x0,crop_w),min(y0,crop_h),min(x1,crop_w),min(y1,crop_h)]
            spec["source_context_xyxy"]=[min(cx0,crop_w),min(cy0,crop_h),min(cx1,crop_w),min(cy1,crop_h)]
            spec["rgb_file"]=f"panel-{spec['panel_index']:03d}.ppm"
            spec["tile_count_estimate"]=(band["width"]+TILE_WIDTH-1)//TILE_WIDTH * ((band["height"]+TILE_HEIGHT-1)//TILE_HEIGHT)
            planned += band["width"]*band["height"]*3+64
            panel_specs.append(spec)
    if planned > MAX_OUTPUT_BYTES:
        raise ValueError("Planned private RGB panels exceed 2 GiB")
    output.mkdir(parents=True,exist_ok=False)
    completed, total_bytes = [], 0
    try:
        # Stream one source photo and one panel at a time; never cache all photos.
        for image_index,source in enumerate(inventory):
            image_path=Path(source["source_image"]); before=tuple(source["source_stat"])
            ew,eh=source["effective_dimensions"]
            with Image.open(image_path) as raw:
                image=raw.convert("RGB")
                for spec in (p for p in panel_specs if p["image_index"]==image_index):
                    _,cy0,_,cy1=spec["context_xyxy"]
                    crop_w,crop_h=source["crop_dimensions"]
                    source_y1=min(cy1,crop_h)
                    crop=image.crop((0,cy0,crop_w,source_y1))
                    pixels=np.asarray(crop,dtype=np.uint8)
                    panel_pixels=np.pad(pixels,((0,cy1-source_y1),(0,ew-crop_w),(0,0)),mode="edge")
                    if panel_pixels.shape!=(cy1-cy0,ew,3): raise ValueError("Padded panel geometry mismatch")
                    panel_bytes=panel_pixels.tobytes()
                    path=output/spec["rgb_file"]
                    with path.open("wb") as handle:
                        handle.write(f"P6\n{ew} {cy1-cy0}\n255\n".encode("ascii")); handle.write(panel_bytes)
                    total_bytes+=path.stat().st_size
                    completed.append({**spec,"rgb_sha256":sha256(path),"rgb_bytes":path.stat().st_size})
                    del panel_bytes,panel_pixels,pixels,crop
                image.close()
            st=image_path.stat()
            if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=before or sha256(image_path)!=source["source_sha256"]:
                raise ValueError(f"Source changed while preparing panels: {image_path}")
            source.pop("source_stat")
        if total_bytes > MAX_OUTPUT_BYTES:
            raise ValueError("Written private RGB panels exceed 2 GiB")
        manifest={"schema":"dronecrowd-photo-panels-v1","status":"complete",
            "prepare_script_sha256":sha256(__file__),"source_directory":str(source_dir),
            "sources":inventory,"panels":completed,
            "geometry":{"effective_rule":"crop right/bottom by at most 3 to multiples of four; never pad or invent source pixels",
                "band_core_height":CORE_HEIGHT,"band_halo":HALO,"tile_core":[TILE_WIDTH,TILE_HEIGHT],
                "tile_context_halo":HALO,"max_panel_pixels":MAX_PANEL_PIXELS},
            "panel_count":len(completed),"panel_bytes_total":total_bytes,
            "total_tile_count_estimate":sum(p["tile_count_estimate"] for p in completed),
            "estimated_gpu_seconds":None,
            "gpu_time_estimate_note":"Not estimated without measured throughput; panel and tile counts describe workload."}
        write_json(output/"inputs.json",manifest)
        return manifest
    except Exception as error:
        write_json(output/"prepare-status.json",{"status":"failed","error":str(error),
            "completed_panels":len(completed),"panel_bytes_total":total_bytes})
        raise


def read_panel(entry, directory):
    path=(directory/entry["rgb_file"]).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Panel RGB path/hash mismatch: {path}")
    width,height=entry["width"],entry["height"]
    if width*height>MAX_PANEL_PIXELS or path.stat().st_size>MAX_PANEL_PIXELS*3+64:
        raise ValueError("Panel exceeds RGB pixel bound")
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=entry["rgb_sha256"]:
        raise ValueError(f"Panel RGB hash mismatch: {path}")
    first,second,third,raw=data.split(b"\n",3)
    if first!=b"P6" or second!=f"{width} {height}".encode() or third!=b"255":
        raise ValueError("Unexpected PPM header")
    if len(raw)!=width*height*3:
        raise ValueError("PPM payload size mismatch")
    return np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3)


def infer(input_path, input_hash, panel_index, checkpoint, checkpoint_hash, output_dir):
    output=safe_output(output_dir)
    if output.exists(): raise ValueError("Inference output directory already exists")
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); report={"status":"failed","panel_index":panel_index}
    failure=None
    try:
        input_path=Path(input_path).resolve(strict=True)
        if input_path.stat().st_size>MAX_MANIFEST_BYTES or sha256(input_path)!=input_hash:
            raise ValueError("Input manifest size or SHA-256 mismatch")
        manifest=json.loads(input_path.read_text())
        if manifest.get("schema")!="dronecrowd-photo-panels-v1" or manifest.get("status")!="complete":
            raise ValueError("Invalid prepared RGB manifest")
        if not 0<=panel_index<len(manifest["panels"]): raise ValueError("Unknown panel index")
        panel=manifest["panels"][panel_index]
        if panel.get("panel_index")!=panel_index: raise ValueError("Panel index/manifest mismatch")
        source=manifest["sources"][panel["image_index"]]
        expected_band=band_geometry(*source["source_dimensions"])[panel["band_index"]]
        for key in ("width","height","core_xyxy","context_xyxy","core_local_xyxy"):
            if panel.get(key)!=expected_band[key]: raise ValueError("Panel geometry differs from source dimensions")
        if panel.get("rgb_file")!=f"panel-{panel_index:03d}.ppm":
            raise ValueError("Unexpected prepared panel filename")
        source_path=Path(source["source_image"]).resolve(strict=True)
        source_before=source_path.stat()
        if sha256(source_path)!=source["source_sha256"]: raise ValueError("Original photo SHA-256 mismatch")
        rgb=read_panel(panel,input_path.parent)
        if checkpoint_hash.lower()!=ORIGINAL_CHECKPOINT_SHA256:
            raise ValueError("Only the verified original density checkpoint is allowed")
        from sample_video import memory_limits
        limits=memory_limits()
        if not limits.get("memory.max","").isdigit() or int(limits["memory.max"])>2*2**30:
            raise ValueError("Cgroup memory.max exceeds the 2 GiB run bound")
        import torch
        from count_video import load_density_checkpoint, normalized_rgb
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; inference requires explicit GPU run")
        torch.set_num_threads(1); torch.cuda.reset_peak_memory_stats()
        cuda_free,cuda_total=torch.cuda.mem_get_info()
        model=load_density_checkpoint(checkpoint,checkpoint_hash).to("cuda").eval()
        if time.monotonic()-started>120: raise TimeoutError("Panel deadline exhausted loading checkpoint")
        deadline=started+120
        def predict(crop):
            if crop.shape[0]>320 or crop.shape[1]>512:
                raise ValueError("Tile input exceeds 512x320")
            with torch.inference_mode():
                result=model([normalized_rgb(crop).to("cuda")])[0][0][0,0].detach().cpu().numpy().astype(np.float32,copy=True)
            return result
        density,tile_count=assemble_panel(rgb,predict,deadline)
        if time.monotonic()>=deadline: raise TimeoutError("Panel deadline exhausted after tile assembly")
        cx0,cy0,_,_=panel["context_xyxy"]
        x0,y0,x1,y1=panel["core_xyxy"]
        core=density[(y0-cy0)//2:(y1-cy0)//2,x0//2:x1//2].copy()
        if core.shape!=((y1-y0)//2,(x1-x0)//2) or not np.isfinite(core).all():
            raise ValueError("Core density output shape/finiteness failure")
        signed_sum=float(core.sum(dtype=np.float64)); negative=core<0
        density_path=output/"core-density.npy"; np.save(density_path,core,allow_pickle=False)
        st=source_path.stat()
        if ((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(source_before.st_dev,source_before.st_ino,source_before.st_size,source_before.st_mtime_ns)
                or sha256(source_path)!=source["source_sha256"]):
            raise ValueError("Original source changed during inference")
        report.update(status="complete",input_manifest=str(input_path),input_sha256=input_hash,
            panel_index=panel_index,image_index=panel["image_index"],band_index=panel["band_index"],
            source_name=source["source_name"],
            input_panel_sha256=panel["rgb_sha256"],source_image=str(source_path),source_sha256=source["source_sha256"],
            source_dimensions=source["source_dimensions"],effective_dimensions=source["effective_dimensions"],
            full_effective_dimensions=source["full_effective_dimensions"],
            core_xyxy=panel["core_xyxy"],context_xyxy=panel["context_xyxy"],
            source_core_xyxy=panel["source_core_xyxy"],source_context_xyxy=panel["source_context_xyxy"],
            checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA256,script_sha256=sha256(__file__),device="cuda",
            count_video_sha256=sha256(ROOT/"scripts/count_video.py"),
            model_code_sha256=sha256(ROOT/"STNNet/stnmodel.py"),
            torch_version=torch.__version__,numpy_version=np.__version__,
            memory_limits=limits,tile_count=tile_count,core_density_shape=list(core.shape),
            core_density_sum_signed=signed_sum,negative_pixel_count=int(negative.sum()),
            negative_mass_signed=float(core[negative].sum(dtype=np.float64)),
            density_file=density_path.name,density_sha256=sha256(density_path),
            seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_free_bytes_before_load=cuda_free,cuda_total_bytes=cuda_total,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        model=None; del rgb,density,core
    except Exception as error:
        failure=error; report["error"]=f"{type(error).__name__}: {error}"
    finally:
        report.setdefault("seconds",time.monotonic()-started)
        report["status"]="failed" if failure else report["status"]
        try: write_json(output/"report.json",report)
        except Exception as report_error:
            if failure is None: failure=report_error
    if failure: raise RuntimeError(f"Photo panel inference failed; see {output/'report.json'}") from failure
    return report


def aggregate(input_path,input_hash,source_index,panel_report_paths,output_dir):
    output=safe_output(output_dir)
    if output.exists(): raise ValueError("Aggregate output directory already exists")
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"failed","source_index":source_index}; failure=None
    try:
        input_path=Path(input_path).resolve(strict=True)
        if input_path.stat().st_size>MAX_MANIFEST_BYTES or sha256(input_path)!=input_hash:
            raise ValueError("Input manifest size or SHA-256 mismatch")
        manifest=json.loads(input_path.read_text())
        if manifest.get("schema")!="dronecrowd-photo-panels-v1" or manifest.get("status")!="complete":
            raise ValueError("Invalid prepared RGB manifest")
        if not 0<=source_index<len(manifest["sources"]): raise ValueError("Unknown source index")
        source=manifest["sources"][source_index]
        bands=sorted((p for p in manifest["panels"] if p["image_index"]==source_index),key=lambda p:p["band_index"])
        if len(panel_report_paths)!=len(bands) or not bands: raise ValueError("Need exactly one completed report per source band")
        height,width=source["full_effective_dimensions"][1],source["full_effective_dimensions"][0]
        cores=[]; report_hashes=[]
        expected_script_hash=sha256(__file__)
        by_index={}
        for report_arg in panel_report_paths:
            report_path=Path(report_arg).resolve(strict=True)
            if not report_path.is_relative_to(PRIVATE_ROOT.resolve()) or report_path.stat().st_size>MAX_MANIFEST_BYTES:
                raise ValueError("Panel report must be a small private output file")
            item=json.loads(report_path.read_text())
            if (item.get("status")!="complete" or item.get("input_sha256")!=input_hash
                    or item.get("source_sha256")!=source["source_sha256"]
                    or item.get("script_sha256")!=expected_script_hash
                    or item.get("checkpoint_sha256")!=ORIGINAL_CHECKPOINT_SHA256):
                raise ValueError(f"Panel report provenance mismatch: {report_path}")
            panel_id=item.get("panel_index")
            if panel_id in by_index: raise ValueError("Duplicate panel report")
            by_index[panel_id]=(report_path,item)
        for band in bands:
            panel_id=band["panel_index"]
            if panel_id not in by_index: raise ValueError("Missing source band report")
            report_path,item=by_index[panel_id]
            if (item.get("core_xyxy")!=band["core_xyxy"] or item.get("context_xyxy")!=band["context_xyxy"]
                    or item.get("input_panel_sha256")!=band["rgb_sha256"]
                    or item.get("image_index")!=source_index or item.get("band_index")!=band["band_index"]
                    or item.get("source_name")!=source["source_name"]
                    or item.get("source_dimensions")!=source["source_dimensions"]
                    or item.get("full_effective_dimensions")!=source["full_effective_dimensions"]):
                raise ValueError("Panel report geometry or RGB hash differs from input manifest")
            density_path=(report_path.parent/item["density_file"]).resolve(strict=True)
            if not density_path.is_relative_to(report_path.parent.resolve()) or sha256(density_path)!=item["density_sha256"]:
                raise ValueError("Core density path/hash mismatch")
            core=np.load(density_path,allow_pickle=False)
            x0,y0,x1,y1=band["core_xyxy"]
            expected_shape=((y1-y0)//2,(x1-x0)//2)
            if core.dtype!=np.float32 or core.shape!=expected_shape or not np.isfinite(core).all():
                raise ValueError("Core density shape/dtype/finiteness mismatch")
            cores.append((band["core_xyxy"],core))
            report_hashes.append({"path":str(report_path),"sha256":sha256(report_path),
                "panel_index":panel_id,"density_sha256":item["density_sha256"]})
        full=stitch_photo_cores(width,height,cores)
        counts=photo_count_summary(full)
        density_path=output/"density-full.npy"; np.save(density_path,full,allow_pickle=False)
        image=Path(source["source_image"]).resolve(strict=True)
        if sha256(image)!=source["source_sha256"]: raise ValueError("Source photo hash changed before aggregate")
        report.update(status="complete",input_manifest=str(input_path),input_sha256=input_hash,
            source_image=str(image),source_sha256=source["source_sha256"],
            source_dimensions=source["source_dimensions"],crop_dimensions=source["crop_dimensions"],
            full_effective_dimensions=source["full_effective_dimensions"],
            global_panel_count=manifest["panel_count"],
            global_panel_bytes_total=manifest["panel_bytes_total"],
            global_tile_count_estimate=manifest["total_tile_count_estimate"],
            estimated_gpu_seconds=manifest["estimated_gpu_seconds"],
            gpu_time_estimate_note=manifest["gpu_time_estimate_note"],
            cropped_right_bottom_px=source["cropped_right_bottom_px"],padded_right_bottom_px=source["padded_right_bottom_px"],
            panel_reports=report_hashes,core_coverage="each effective density pixel exactly once",
            raw_total=counts["raw_total"],negative_mass_signed=counts["negative_mass_signed"],
            negative_mass_abs=counts["negative_mass_abs"],negative_pixel_count=counts["negative_pixel_count"],
            quadrant_order=["top-left","top-right","bottom-left","bottom-right"],
            raw_quadrant_sums=counts["raw_quadrant_sums"],count=counts["count"],
            count_rule="sum(max(0, raw float32 quadrant sum))/100; clamp once per complete-photo quadrant",
            density_file=density_path.name,density_sha256=sha256(density_path),
            script_sha256=expected_script_hash)
    except Exception as error:
        failure=error; report["error"]=f"{type(error).__name__}: {error}"
    finally:
        try: write_json(output/"report.json",report)
        except Exception as report_error:
            if failure is None: failure=report_error
    if failure: raise RuntimeError(f"Photo aggregate failed; see {output/'report.json'}") from failure
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest="mode",required=True)
    prep=modes.add_parser("prepare"); prep.add_argument("--sources-dir",type=Path,required=True); prep.add_argument("--output-dir",type=Path,required=True)
    run=modes.add_parser("infer"); run.add_argument("--input",type=Path,required=True); run.add_argument("--input-sha256",required=True)
    run.add_argument("--panel-index",type=int,required=True); run.add_argument("--checkpoint",type=Path,required=True)
    run.add_argument("--checkpoint-sha256",required=True); run.add_argument("--output-dir",type=Path,required=True)
    agg=modes.add_parser("aggregate"); agg.add_argument("--input",type=Path,required=True)
    agg.add_argument("--input-sha256",required=True); agg.add_argument("--source-index",type=int,required=True)
    agg.add_argument("--panel-reports",type=Path,nargs="+",required=True); agg.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.mode=="prepare":
        manifest=prepare(args.sources_dir,args.output_dir)
        path=Path(args.output_dir).resolve()/"inputs.json"
        print(json.dumps({"status":manifest["status"],"panels":len(manifest["panels"]),"manifest_sha256":sha256(path)}))
    elif args.mode=="infer":
        report=infer(args.input,args.input_sha256,args.panel_index,args.checkpoint,args.checkpoint_sha256,args.output_dir)
        print(json.dumps({"status":report["status"],"panel_index":report["panel_index"],"core_density_sum_signed":report.get("core_density_sum_signed")}))
    else:
        report=aggregate(args.input,args.input_sha256,args.source_index,args.panel_reports,args.output_dir)
        print(json.dumps({"status":report["status"],"source_index":args.source_index,"count":report["count"]}))


if __name__=="__main__": main()
