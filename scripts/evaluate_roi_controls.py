"""Evaluate provisional visual references and negative controls, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import torch

from count_video import load_density_checkpoint, normalized_rgb, read_rgb, validate_frames
from count_video_tiles import assemble_density, tile_windows
from sample_video import memory_limits, sha256


def roi_count(density, rectangle, scale):
    if (len(rectangle)!=4 or any(type(v) is not int or v%4 for v in rectangle)
            or scale not in (1,2)):
        raise ValueError("ROI coordinates must be aligned integers")
    x,y,right,bottom = [v*scale//2 for v in rectangle]
    if not (0<=x<right<=density.shape[1] and 0<=y<bottom<=density.shape[0]):
        raise ValueError("ROI outside density map")
    region = density[y:bottom,x:right]
    if region.size != (bottom-y)*(right-x) or not np.isfinite(region).all():
        raise ValueError("ROI does not match density geometry")
    count = float(region.sum(dtype=np.float64))/100
    if not np.isfinite(count):
        raise ValueError("Non-finite ROI integral")
    return max(0.0,count),count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls",type=Path,required=True)
    parser.add_argument("--controls-sha256",required=True)
    parser.add_argument("--scale",type=int,choices=[1,2],required=True)
    parser.add_argument("--variant",choices=["density","localization"],default="density")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started+120
    torch.set_num_threads(1)
    path=args.controls.resolve(strict=True)
    if path.stat().st_size>256*1024 or sha256(path)!=args.controls_sha256:
        parser.error("Control manifest size or hash mismatch")
    controls=json.loads(path.read_text())
    entries=controls["entries"]
    validate_frames(entries,path.parent,max_width=768,max_height=768,max_frames=16,require_timestamps=False)
    for entry in entries:
        list(tile_windows(entry["width"]*args.scale,entry["height"]*args.scale))
        rectangle=entry["roi_context_xyxy"]
        if (len(rectangle)!=4 or any(type(v) is not int or v%4 for v in rectangle)
                or not 0<=rectangle[0]<rectangle[2]<=entry["width"]
                or not 0<=rectangle[1]<rectangle[3]<=entry["height"]):
            parser.error("Invalid control ROI geometry")
        interval=entry["reference_interval"]
        if interval is not None and (not isinstance(interval,list) or len(interval)!=2
                or any(type(v) is not int for v in interval) or not 0<=interval[0]<=interval[1]<=512):
            parser.error("Invalid provisional count interval")
    output=args.output_dir.resolve()
    if path.is_relative_to(output) or any(output.is_relative_to(Path(e["source_image"]).parent) for e in entries):
        parser.error("Output overlaps source/control inputs")
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"started","variant":args.variant,"controls":str(path),"controls_sha256":args.controls_sha256,"scale":args.scale,
            "prediction_head":"den1 density; localization peaks not decoded",
            "transform":"identity" if args.scale==1 else "FFmpeg Lanczos 2x", "rows":[],
            "checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":args.checkpoint_sha256,
            "script_sha256":sha256(__file__),"loader_sha256":sha256(Path(__file__).with_name("count_video.py")),
            "tiles_code_sha256":sha256(Path(__file__).with_name("count_video_tiles.py")),
            "model_code_sha256":sha256(Path(__file__).resolve().parents[1]/"STNNet/stnmodel.py"),
            "memory_limits":memory_limits(),"device":"cuda","torch_version":torch.__version__,
            "numpy_version":np.__version__,
            "interpretation":"Provisional assistant visual references; no certified accuracy or fitted calibration."}
    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        model=load_density_checkpoint(args.checkpoint,args.checkpoint_sha256,use_loc=args.variant=="localization").to("cuda")
        report["loaded_state_tensors"]=len(model.state_dict())
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for entry in entries:
                source=path.parent/entry["rgb_file"]
                if sha256(source)!=entry["rgb_sha256"]:
                    raise ValueError("RGB hash mismatch")
                rgb=read_rgb(source,entry["width"],entry["height"],max_width=768,max_height=768)
                if args.scale==2:
                    ffmpeg=shutil.which("ffmpeg")
                    if not ffmpeg:
                        raise ValueError("FFmpeg required")
                    width,height=entry["width"]*2,entry["height"]*2
                    decoded=subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-threads","1",
                        "-filter_threads","1","-i",str(source),"-vf",f"scale={width}:{height}:flags=lanczos",
                        "-frames:v","1","-pix_fmt","rgb24","-f","rawvideo","pipe:1"],check=True,
                        capture_output=True,timeout=min(30,max(.01,deadline-time.monotonic())))
                    if len(decoded.stdout)!=width*height*3:
                        raise ValueError("Unexpected interpolation payload")
                    rgb=np.frombuffer(decoded.stdout,dtype=np.uint8).reshape(height,width,3)
                def predict(crop):
                    return model([normalized_rgb(crop).to("cuda")])[0][0][0,0].cpu().numpy().copy()
                transformed_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()
                density,tiles=assemble_density(rgb,predict,deadline)
                count,raw=roi_count(density,entry["roi_context_xyxy"],args.scale)
                name=f"density-{entry['index']:03d}.npy"
                np.save(output/name,density,allow_pickle=False)
                interval=entry["reference_interval"]
                discrepancy=None if interval is None else max(interval[0]-count,count-interval[1],0)
                row={"index":entry["index"],"name":entry["name"],"count":count,"raw_roi_count":raw,
                     "reference_interval":interval,"distance_to_provisional_interval":discrepancy,
                     "transformed_rgb_sha256":transformed_rgb_sha256,
                     "density_file":name,"density_sha256":sha256(output/name),"tile_count":tiles}
                report["rows"].append(row)
                print(json.dumps(row),flush=True)
        report.update(status="complete",seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
