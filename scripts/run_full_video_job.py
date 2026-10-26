"""Fixed-profile dispatcher entrypoint for one approved 0269 block/model."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

import full_video_campaign as campaign
import run_dispatched_inference as bounded


def load_plan(path, supplied_sha256, *, repo_root=bounded.ROOT):
    path = Path(path).resolve(strict=True)
    base = (Path(repo_root)/"outputs/video-completo-0269-20261025").resolve()
    if (path != base/"plan.json" or path.stat().st_size > 2**20
            or not bounded.HEX64.fullmatch(supplied_sha256)
            or bounded.sha256_file(path) != supplied_sha256):
        raise ValueError("Full video plan path, size or SHA mismatch")
    data = json.loads(path.read_text())
    if (data.get("schema_version") != 2 or data.get("project") != "DroneCrowd"
            or data.get("source_id") != "0269" or data.get("source_sha256") != campaign.SOURCE_HASH
            or data.get("frame_count") != 563 or data.get("frames_per_process") != 16
            or data.get("gpu_batch_size") != 1 or len(data.get("jobs", [])) != 108
            or bounded.sha256_file(base/"inventory.json") != data.get("inventory_sha256")):
        raise ValueError("Plan is outside the authorized 0269 campaign")
    jobs = data["jobs"]
    if len({j["job_id"] for j in jobs}) != 108 or {(j["block"], j["model"]) for j in jobs} != {
            (block, model) for block in range(36) for model in campaign.MODELS}:
        raise ValueError("Plan requires exactly one job per source block and model")
    inventory = json.loads((base/"inventory.json").read_text())
    if inventory["source_sha256"] != campaign.SOURCE_HASH or len(inventory["frames"]) != 563:
        raise ValueError("Invalid full source inventory")
    for block in range(36):
        path = base/f"inputs/block-{block:03d}.json"
        manifest = json.loads(path.read_text())
        expected = [{**frame, "index": i, "rgb_file": f"frame-{i:03d}.ppm"}
                    for i, frame in enumerate(inventory["frames"][block*16:(block+1)*16])]
        if manifest != {"source_sha256": campaign.SOURCE_HASH, "frames": expected}:
            raise ValueError("Block manifest differs from the frozen decoded inventory")
        for frame in expected:
            if frame["pts_seconds"] != float(frame["pts"]*Fraction(frame["time_base"])):
                raise ValueError("Original PTS units differ")
    return data


def select_job(plan, job_id):
    matches = [j for j in plan["jobs"] if j["job_id"] == job_id]
    if len(matches) != 1 or not bounded.HEX64.fullmatch(job_id):
        raise ValueError("Request ID does not select exactly one frozen block")
    job = matches[0]
    model, block = job["model"], job["block"]
    if (type(block) is not int or not 0 <= block < 36 or model not in campaign.MODELS
            or job["profile"] != f"dronecrowd-full0269-{model}-v1"
            or job["resources"] != campaign.RESOURCES
            or job["checkpoint"] != campaign.previous.CHECKPOINTS[model]["path"]
            or job["checkpoint_sha256"] != campaign.previous.CHECKPOINTS[model]["sha256"]
            or job["input_manifest"] != f"outputs/video-completo-0269-20261025/inputs/block-{block:03d}.json"
            or job["output_dir"] != f"outputs/video-completo-0269-20261025/results/{model}/block-{block:03d}"
            or job["rgb_cache"] != str(campaign.CACHE/f"block-{block:03d}")
            or job_id != campaign.previous.canonical_hash({"model": model,
                "input_sha256": job["input_sha256"], "checkpoint_sha256": job["checkpoint_sha256"],
                "code_digests": plan["code_digests"]})):
        raise ValueError("Job differs from the bounded campaign contract")
    return job


def command(job, paths, *, executable, clip_source=None):
    result = bounded.command_for_job(job, paths, executable=executable, clip_source=clip_source)
    result.extend(["--full-video-block", "--rgb-root", job["rgb_cache"]])
    return result


def completed(output_path, job, input_path, plan):
    if not bounded.completed_output(output_path, job, input_path, plan):
        return False
    try:
        manifest = json.loads(Path(input_path).read_text())
        frames = manifest["frames"]
        if (manifest["source_sha256"] != campaign.SOURCE_HASH
                or len(frames) != (3 if job["block"] == 35 else 16)):
            return False
        report = json.loads((Path(output_path)/"report.json").read_text())
        for index, (frame, result) in enumerate(zip(frames, report["results"])):
            if frame["source_ordinal"] != job["block"]*16+index:
                return False
            if any(frame[key] != result.get(key) for key in ("source_ordinal", "pts", "time_base")):
                return False
        return report.get("source_sha256") == campaign.SOURCE_HASH
    except (KeyError, ValueError, TypeError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--model", choices=campaign.MODELS, required=True)
    args = parser.parse_args()
    job_id = os.environ.get("NEXO_GPU_REQUEST_ID", "")
    try:
        plan = load_plan(args.plan, args.plan_sha256)
        job = select_job(plan, job_id)
        if (job["model"] != args.model or os.environ.get("NEXO_GPU_PROJECT") != "DroneCrowd"
                or os.environ.get("NEXO_GPU_PROFILE") != job["profile"]):
            raise ValueError("Dispatcher owner/profile/model does not match the frozen request")
        if not completed(bounded.ROOT/job["output_dir"], job, bounded.ROOT/job["input_manifest"], plan):
            campaign.validate_cache(job["block"], json.loads((bounded.ROOT/job["input_manifest"]).read_text()))
        unit, directory = bounded.cgroup_scope(Path("/proc/self/cgroup").read_text())
        subprocess.run(["systemctl", "--user", "set-property", "--runtime", unit,
                        "CPUQuota=100%"], check=True, capture_output=True, timeout=5)
        quota, period = (directory/"cpu.max").read_text().split()
        if not quota.isdigit() or int(quota) > int(period):
            raise ValueError("GPU job CPU quota must not exceed one core")
        result = bounded.run_job(plan_path=args.plan, plan_sha256=args.plan_sha256,
            job_id=job_id, environment_lock=bounded.ROOT/"ops/gpu-candidates.lock.json",
            plan_loader=load_plan, job_selector=select_job, command_builder=command,
            completion_checker=completed)
        print(json.dumps(result))
    except Exception as error:
        evidence = bounded.write_preflight_failure(error, plan_sha256=args.plan_sha256, job_id=job_id)
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "evidence_file": str(evidence)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
