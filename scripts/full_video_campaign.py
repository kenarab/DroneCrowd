"""CPU preparation and queue client for the frozen 0269 full-frame campaign.

No GPU work is launched here: inference belongs exclusively to approved dispatcher
profiles. RGB cache is one block, stored only in the authorized temporary root.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import prepare_long_video_jobs as previous
import run_dispatched_inference as bounded

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path("/home/alephbet/home.codex/tmp")
BASE = ROOT / "outputs/video-completo-0269-20261025"
CACHE = TEMP / "dronecrowd-full-0269-20261025"
SOURCE_MANIFEST = ROOT / "outputs/resolucion-original/0269/entrada-native/input.json"
SOURCE_HASH = "9a20838aed3912205eb2e19f8f70d53346328731df214c194bd148b8a8432199"
MODELS = ("sr", "clip-ebc", "mpcount")
RESOURCES = {**previous.RESOURCE_PROFILE, "disk_increment_bytes": 2**30}
CODE_FILES = tuple(previous.CODE_FILES) + (
    "scripts/run_dispatched_inference.py", "scripts/run_full_video_job.py",
    "scripts/full_video_campaign.py", "ops/gpu-candidates.lock.json")


def encoded(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded(data))
        handle.flush()
        os.fsync(handle.fileno())


def write_state(path, data):
    with tempfile.NamedTemporaryFile(dir=TEMP, prefix="dronecrowd-campaign-", delete=False) as handle:
        staging = Path(handle.name)
        try:
            handle.write(encoded(data))
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def preflight():
    free = shutil.disk_usage(ROOT).free
    available = previous._read_mem_available()
    if free < 128 * 2**30 or available < 3 * 2**30:
        raise RuntimeError("Need 128 GiB disk and 3 GiB available host RAM before campaign CPU work")
    return {"disk_free_bytes": free, "host_mem_available_bytes": available}


def source():
    data = json.loads(SOURCE_MANIFEST.read_text())
    path = Path(data["source"]).resolve(strict=True)
    if data["source_sha256"] != SOURCE_HASH or bounded.sha256_file(path) != SOURCE_HASH:
        raise ValueError("0269 original source hash mismatch")
    return path


def decode(path, start, count):
    """Yield one raw native RGB at a time; external timeout also bounds pipe reads."""
    command = ["/usr/bin/timeout", "--kill-after=5", "110", "ffmpeg", "-nostdin",
        "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-vf",
        f"select=between(n\\,{start}\\,{start+count-1})", "-frames:v", str(count),
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads", "1",
        "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile(dir=TEMP) as errors:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=errors, preexec_fn=bounded._set_child_file_limit)
        try:
            for _ in range(count):
                payload = child.stdout.read(3840 * 2160 * 3)
                if len(payload) != 3840 * 2160 * 3:
                    raise ValueError("Decoder returned fewer complete native frames than expected")
                yield payload
                del payload
            if child.stdout.read(1) or child.wait(timeout=10) != 0:
                raise ValueError("Decoder returned extra data or a failing exit status")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()


def ppm_header():
    return b"P6\n3840 2160\n255\n"


def cache_path(block):
    if CACHE.is_symlink() or not CACHE.resolve().is_relative_to(TEMP.resolve()):
        raise ValueError("Campaign cache root escaped the authorized temporary")
    path = CACHE/f"block-{block:03d}"
    if path.is_symlink() or not path.resolve().is_relative_to(CACHE.resolve()):
        raise ValueError("Campaign block cache is a symbolic link or escaped its root")
    return path


def validate_cache(block, manifest):
    path = cache_path(block)
    frames = manifest["frames"]
    expected = {frame["rgb_file"] for frame in frames}
    if (len(expected) != len(frames) or not path.is_dir()
            or {file.name for file in path.iterdir()} != expected):
        raise ValueError("Partial or unexpected cache; preserve it for explicit review")
    for frame in frames:
        name = frame["rgb_file"]
        if name != f"frame-{frame['index']:03d}.ppm":
            raise ValueError("Unsafe RGB basename")
        file = path/name
        if (file.is_symlink() or not file.is_file()
                or file.stat().st_size != len(ppm_header())+3840*2160*3
                or bounded.sha256_file(file) != frame["rgb_sha256"]):
            raise ValueError("Cached RGB is partial or differs from the frozen inventory")
    return path


def inventory():
    pre = preflight()
    path = source()
    before = path.stat()
    result = subprocess.run(["ffprobe", "-v", "error", "-threads", "1",
        "-select_streams", "v:0", "-show_frames", "-show_streams", "-show_entries",
        "frame=pts:stream=width,height,time_base,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=110)
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    frames = metadata["frames"]
    if ((stream["width"], stream["height"]) != (3840, 2160)
            or len(frames) != 563 or int(stream["nb_frames"]) != len(frames)):
        raise ValueError("0269 decoded inventory differs from the reviewed 563 native 4K frames")
    time_base = stream["time_base"]
    records = []
    for ordinal, payload in enumerate(decode(path, 0, len(frames))):
        pts = frames[ordinal]["pts"]
        if type(pts) is not int:
            raise ValueError("Original decoded PTS must be an integer")
        digest = hashlib.sha256(ppm_header())
        digest.update(payload)
        records.append({"source_ordinal": ordinal, "pts": pts, "time_base": time_base,
            "pts_seconds": float(pts * Fraction(time_base)), "width": 3840, "height": 2160,
            "rgb_sha256": digest.hexdigest()})
        del payload
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Original changed during complete decoding")
    write_new(BASE / "inventory.json", {"source": str(path), "source_sha256": SOURCE_HASH,
        "source_stat": [after.st_ino, after.st_size, after.st_mtime_ns], "frames": records,
        "decoded_frame_count": len(records), "preflight": pre, "gpu_launched": False})
    print(json.dumps({"status": "inventoried", "decoded_frames": len(records)}))


def freeze():
    data = json.loads((BASE / "inventory.json").read_text())
    if data["source_sha256"] != SOURCE_HASH or len(data["frames"]) != 563:
        raise ValueError("Full decoded inventory is required")
    # Reuse the exact audited upstream source manifest and checkpoint identities.
    old_plan = json.loads((ROOT / "outputs/validacion-extendida-cola-20261025/plan.json").read_text())
    prerequisites = old_plan["local_prerequisites"]
    code = {name: bounded.sha256_file(ROOT / name) for name in CODE_FILES}
    code.update({key: value for key, value in old_plan["code_digests"].items()
                 if key.startswith("clip-source/") or key == bounded.MPCOUNT_UPSTREAM_KEY})
    plan = {"schema_version": 2, "project": "DroneCrowd", "source_id": "0269",
        "source_sha256": SOURCE_HASH, "frame_count": 563, "frames_per_process": 16,
        "gpu_batch_size": 1, "code_digests": code, "local_prerequisites": prerequisites,
        "inventory_sha256": bounded.sha256_file(BASE / "inventory.json"), "jobs": []}
    bounded.verify_code_digests(plan)
    for checkpoint in previous.CHECKPOINTS.values():
        if bounded.sha256_file(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint mismatch")
    for start in range(0, 563, 16):
        block = start // 16
        records = [{**frame, "index": index, "rgb_file": f"frame-{index:03d}.ppm"}
                   for index, frame in enumerate(data["frames"][start:start+16])]
        manifest = {"source_sha256": SOURCE_HASH, "frames": records}
        relative = f"outputs/video-completo-0269-20261025/inputs/block-{block:03d}.json"
        write_new(ROOT / relative, manifest)
        input_hash = digest_bytes(encoded(manifest))
        for model in MODELS:
            checkpoint = previous.CHECKPOINTS[model]
            job_id = previous.canonical_hash({"model": model, "input_sha256": input_hash,
                "checkpoint_sha256": checkpoint["sha256"], "code_digests": code})
            job = {"job_id": job_id, "model": model, "block": block,
                "profile": f"dronecrowd-full0269-{model}-v1", "input_manifest": relative,
                "input_sha256": input_hash, "checkpoint": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"], "resources": RESOURCES,
                "rgb_cache": str(CACHE / f"block-{block:03d}"),
                "output_dir": f"outputs/video-completo-0269-20261025/results/{model}/block-{block:03d}"}
            plan["jobs"].append(job)
    write_new(BASE / "plan.json", plan)
    print(json.dumps({"status": "frozen", "jobs": len(plan["jobs"]),
                      "plan_sha256": bounded.sha256_file(BASE / "plan.json")}))


def prepare(block):
    if type(block) is not int or not 0 <= block < 36:
        raise ValueError("Block outside the frozen 0269 campaign")
    preflight()
    data = json.loads((BASE / "inventory.json").read_text())
    path = source()
    info = path.stat()
    if [info.st_ino, info.st_size, info.st_mtime_ns] != data["source_stat"]:
        raise ValueError("Source identity changed since inventory")
    target = cache_path(block)
    if target.exists():
        raise FileExistsError("Cache already exists; verify or remove only the owned block explicitly")
    target.mkdir(parents=True)
    records = data["frames"][block*16:(block+1)*16]
    for index, (payload, frame) in enumerate(zip(decode(path, block*16, len(records)), records)):
        digest = hashlib.sha256(ppm_header())
        digest.update(payload)
        if digest.hexdigest() != frame["rgb_sha256"]:
            raise ValueError("Re-decoded RGB differs from the frozen native inventory")
        with (target / f"frame-{index:03d}.ppm").open("xb") as handle:
            handle.write(ppm_header())
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        del payload
    print(json.dumps({"status": "prepared", "block": block, "frames": len(records)}))


def profiles():
    import run_full_video_job as worker
    plan_path = BASE / "plan.json"
    plan_hash = bounded.sha256_file(plan_path)
    worker.load_plan(plan_path, plan_hash)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("Commit the reviewed worker before generating profiles")
    proposals = {}
    for model in MODELS:
        name = f"dronecrowd-full0269-{model}-v1"
        proposals[name] = {"project": "DroneCrowd", "revision": revision,
            "entrypoint": "scripts/run_full_video_job.py", "environment_lock": "ops/gpu-candidates.lock.json",
            "argv": [str(ROOT / ".venv/gpu-candidates/bin/python"),
                str(ROOT / "scripts/run_full_video_job.py"), "--plan", str(plan_path),
                "--plan-sha256", plan_hash, "--model", model], "ram_gib": 1.5,
            "vram_mib": 1024, "scratch_gib": .5, "outputs_gib": 1,
            "seconds": 120, "max_runs": 36, "priority": "exploratory",
            "approval": "Owner authorized full video campaign 2026-10-25; 0269 only, 563 frames, batch1, 36 blocks/model; SPEC018; independent review and pilot gate."}
    write_new(BASE / "gpu-profiles.json", proposals)
    print(json.dumps({"status": "profiles_proposed_not_applied", "revision": revision}))


def queue_call(dispatcher, *arguments, timeout=15):
    result = subprocess.run(["/usr/bin/python3", str(dispatcher), *arguments],
        check=True, capture_output=True, text=True, timeout=timeout)
    return {"status": "cancellation_requested"} if arguments[0] == "cancel" else json.loads(result.stdout)


def stop_requests(dispatcher, plan, submitted, wait_seconds=150):
    """Reconcile frozen-plan ownership, including requests from an earlier client."""
    owned = {"DroneCrowd:"+job["job_id"] for job in plan["jobs"]}
    terminal = {"completed", "cancelled", "failed", "interrupted", "budget_exhausted"}
    errors = []
    deadline = time.monotonic()+wait_seconds
    def request(*arguments):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Global cancellation deadline exhausted")
        return queue_call(dispatcher, *arguments, timeout=min(15, remaining))
    try:
        current = request("status")
        targets = {r["key"] for r in current if r["key"] in owned and r["state"] not in terminal}
    except Exception as error:
        targets = owned
        errors.append("initial_status:"+str(error)[:256])
    # Favor the current block if the status API is unavailable after many blocks.
    preferred = [key for key in reversed(submitted[-3:]) if key in targets]
    ordered = preferred+sorted(targets-set(preferred))
    for key in ordered:
        if time.monotonic() >= deadline:
            errors.append("global_cancellation_deadline_exhausted")
            break
        try:
            request("cancel", "--key", key)
        except Exception as error:
            errors.append(key+":"+str(error)[:256])
    while True:
        try:
            rows = request("status")
            unsettled = [r["key"] for r in rows if r["key"] in owned and r["state"] not in terminal]
            if not unsettled:
                return {"queue_requests_released": True, "cancel_errors": errors, "unsettled_keys": []}
        except Exception as error:
            errors.append("release_status:"+str(error)[:256])
            return {"queue_requests_released": False, "cancel_errors": errors, "unsettled_keys": sorted(targets)}
        if time.monotonic() >= deadline:
            return {"queue_requests_released": False, "cancel_errors": errors, "unsettled_keys": unsettled}
        time.sleep(min(5, max(0, deadline-time.monotonic())))


def validate_anchor(output, manifest):
    report = json.loads((output / "report.json").read_text())
    # The completed reports themselves identify reference frames by exact RGB hash.
    references = {}
    for path in (ROOT / "outputs/validacion-extendida-cola-20261025/results" / report["model"]).glob("*/report.json"):
        for frame in json.loads(path.read_text())["results"]:
            references[frame["rgb_sha256"]] = frame["count"]
    compared = 0
    for row in report["results"]:
        if row["rgb_sha256"] in references:
            reference = references[row["rgb_sha256"]]
            if abs(row["count"]-reference) > max(.01, abs(reference)*1e-5):
                raise ValueError("Identical RGB anchor differs from the previous scientific recipe")
            compared += 1
    if not compared:
        raise ValueError("Pilot has no exact RGB anchor from the previous completed experiment")
    limits = report["memory_limits"]
    if (limits["memory.high"], limits["memory.max"], limits["memory.swap.max"]) != (
            "1610612736", "2147483648", "536870912"):
        raise ValueError("Pilot effective limits differ")
    if report["total_seconds"] >= 110 or report["max_rss_kib"] * 1024 > 1610612736:
        raise ValueError("Pilot has inadequate runtime or RAM margin")
    return {"anchors": compared, "total_seconds": report["total_seconds"],
            "max_rss_kib": report["max_rss_kib"], "peak_cuda_bytes": report["peak_cuda_bytes"]}


def cpu_command(quota_runner, *arguments):
    return ["/usr/bin/python3", str(quota_runner), "run", "--project", "DroneCrowd",
        "--ram-gib", "0.75", "--cpu-percent", "100", "--", "/usr/bin/timeout",
        "--kill-after=5", "120", str(ROOT/".venv/gpu-candidates/bin/python"),
        str(ROOT/"scripts/full_video_campaign.py"), *arguments]


def validate_maps(block):
    """CPU-only full-map geometry, finiteness and integral validation, one at a time."""
    import numpy as np
    plan = json.loads((BASE/"plan.json").read_text())
    verified = []
    for job in (j for j in plan["jobs"] if j["block"] == block):
        output = ROOT/job["output_dir"]
        report = json.loads((output/"report.json").read_text())
        stride = 1 if job["model"] == "mpcount" else 8
        for row in report["results"]:
            path = output/row["density_file"]
            if bounded.sha256_file(path) != row["density_sha256"]:
                raise ValueError("Density hash mismatch")
            density = np.load(path, allow_pickle=False)
            if (density.dtype != np.float32 or density.shape != (2160//stride, 3840//stride)
                    or not np.isfinite(density).all()
                    or abs(float(density.sum(dtype=np.float64))-row["count"]) > 1e-7
                    or abs(float(np.minimum(density, 0).sum(dtype=np.float64))-row["negative_mass"]) > 1e-7):
                raise ValueError("Density shape, dtype, finite values or integral differs")
            verified.append({"model": job["model"], "source_ordinal": row["source_ordinal"],
                             "density_sha256": row["density_sha256"]})
            del density
    expected = (3 if block == 35 else 16)*3
    if len(verified) != expected:
        raise ValueError("Unexpected number of verified maps")
    write_new(BASE/f"validated-block-{block:03d}.json", {"maps": verified, "status": "verified"})


def coordinate(dispatcher, quota_runner, plan_hash, pilot_only=False):
    """Single product client; no alternate GPU scheduler, reservation or retry loop."""
    import fcntl
    import run_full_video_job as worker
    plan = worker.load_plan(BASE / "plan.json", plan_hash)
    BASE.mkdir(exist_ok=True)
    with (BASE / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submitted = []
        try:
            for block in range(1 if pilot_only else 36):
                jobs = [j for j in plan["jobs"] if j["block"] == block]
                complete = [worker.completed(ROOT/j["output_dir"], j, ROOT/j["input_manifest"], plan) for j in jobs]
                if not all(complete):
                    preflight()
                    cache = cache_path(block)
                    if not cache.exists():
                        command = cpu_command(quota_runner, "prepare", "--block", str(block))
                        result = subprocess.run(command, capture_output=True, text=True, timeout=135)
                        write_new(BASE / f"prepare-{block:03d}.json", {"returncode": result.returncode,
                            "stdout": result.stdout[-8192:], "stderr": result.stderr[-8192:]})
                        if result.returncode:
                            raise RuntimeError("CPU block preparation failed")
                    validate_cache(block, json.loads((ROOT/jobs[0]["input_manifest"]).read_text()))
                    for job, done in zip(jobs, complete):
                        if done:
                            continue
                        row = queue_call(dispatcher, "submit", "--project", "DroneCrowd",
                            "--profile", job["profile"], "--request-id", job["job_id"])
                        submitted.append(row["key"])
                    deadline = time.monotonic() + 6 * 3600
                    while True:
                        rows = {r["key"]: r for r in queue_call(dispatcher, "status")}
                        selected = [rows.get("DroneCrowd:" + j["job_id"]) for j in jobs]
                        write_state(BASE / "state.json", {"state": "waiting", "block": block,
                            "requests": selected, "time_unix": time.time()})
                        if any(r and r["state"] not in ("queued", "waiting_profile", "running", "recovering", "completed") for r in selected):
                            raise RuntimeError("A dispatcher request failed or requires review; no automatic retry")
                        if all(r and r["state"] == "completed" for r in selected):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError("Dispatcher wait exceeded six hours for this block")
                        time.sleep(10)
                if not all(worker.completed(ROOT/j["output_dir"], j, ROOT/j["input_manifest"], plan) for j in jobs):
                    raise RuntimeError("Queue success without three verified complete outputs")
                if not (BASE/f"validated-block-{block:03d}.json").exists():
                    validation = subprocess.run(cpu_command(quota_runner, "validate", "--block", str(block)),
                                                capture_output=True, text=True, timeout=135)
                    if validation.returncode:
                        raise RuntimeError("CPU map validation failed: "+validation.stderr[-1024:])
                if block == 0:
                    checks = {j["model"]: validate_anchor(ROOT/j["output_dir"], ROOT/j["input_manifest"]) for j in jobs}
                    write_state(BASE / "pilot.json", {"status": "passed", "jobs_in_campaign": 3,
                        "checks": checks, "plan_sha256": plan_hash})
                # These PPM files are solely this campaign's regenerable temporary cache.
                cache = cache_path(block)
                if cache.exists():
                    manifest = json.loads((ROOT/jobs[0]["input_manifest"]).read_text())
                    validate_cache(block, manifest)
                    expected = {f["rgb_file"] for f in manifest["frames"]}
                    for name in expected:
                        (cache / name).unlink()
                    cache.rmdir()
                write_state(BASE / "state.json", {"state": "pilot_complete" if pilot_only else "running",
                    "completed_blocks": block+1, "completed_jobs": (block+1)*3,
                    "time_unix": time.time()})
            if not pilot_only:
                summarize(plan)
        except BaseException as error:
            release = stop_requests(dispatcher, plan, submitted)
            write_state(BASE / "state.json", {"state": "stopped", "error": str(error)[:1024],
                        "submitted_keys": submitted, **release,
                        "time_unix": time.time()})
            raise


def summarize(plan):
    import run_full_video_job as worker
    rows = {}
    for job in plan["jobs"]:
        if not worker.completed(ROOT/job["output_dir"], job, ROOT/job["input_manifest"], plan):
            raise ValueError("Cannot summarize an incomplete or changed campaign")
        report = json.loads((ROOT/job["output_dir"]/"report.json").read_text())
        for frame in report["results"]:
            row = rows.setdefault(frame["source_ordinal"], {"source_ordinal": frame["source_ordinal"],
                "pts": frame["pts"], "time_base": frame["time_base"], "pts_seconds": frame["pts_seconds"],
                "rgb_sha256": frame["rgb_sha256"]})
            row[job["model"]] = frame["count"]
    if set(rows) != set(range(563)) or any(not set(MODELS).issubset(row) for row in rows.values()):
        raise ValueError("Missing source ordinals or model predictions")
    with (BASE / "counts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[index] for index in range(563))
    write_state(BASE / "state.json", {"state": "complete", "completed_blocks": 36,
        "completed_jobs": 108, "frames": 563, "predictions": 1689, "time_unix": time.time()})


def main():
    bounded.set_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "freeze", "prepare", "profiles", "coordinate", "validate"])
    parser.add_argument("--block", type=int)
    parser.add_argument("--dispatcher", type=Path)
    parser.add_argument("--quota-runner", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.action == "inventory":
        inventory()
    elif args.action == "freeze":
        freeze()
    elif args.action == "prepare":
        prepare(args.block)
    elif args.action == "profiles":
        profiles()
    elif args.action == "validate":
        validate_maps(args.block)
    else:
        if not (args.dispatcher and args.quota_runner and args.plan_sha256):
            parser.error("coordinate requires dispatcher, quota runner and immutable plan digest")
        def interrupted(_signum, _frame):
            raise InterruptedError("Campaign client received a termination signal")
        signal.signal(signal.SIGTERM, interrupted)
        coordinate(args.dispatcher, args.quota_runner, args.plan_sha256, args.pilot_only)


if __name__ == "__main__":
    main()
