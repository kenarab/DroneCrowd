"""Bounded dispatcher client for the three explicitly approved complete clips.

The original 0269 scientific files remain unchanged. Queue replacement identity
is recorded separately from scientific identity; all GPU work belongs to Nexo.
"""
import argparse
import csv
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import full_video_campaign as legacy
import run_full_video_job as old_worker
import run_dispatched_inference as bounded

ROOT, TEMP, MODELS = legacy.ROOT, legacy.TEMP, legacy.MODELS
ORIGINAL_PLAN_SHA = "6b664fc855f6b227571179bfddb2035022c067b91d995ca66b63ae3a28c4a9b9"
PRIOR_PLAN = ROOT/"outputs/validacion-extendida-cola-20261025/plan.json"
PRIOR_SHA = "93405aba21aaef66f4a9bc85615d3de3e34ce267c382725b3e7455170adbd73b"
SOURCES = {
    "0269": {"count": 563, "width": 3840, "height": 2160, "crop": 0,
        "sha": legacy.SOURCE_HASH},
    "0265": {"count": 1370, "width": 3840, "height": 2160, "crop": 0,
        "sha": "c50a3cef29334d7b7e49d34c45412da1b7164c56177e3a9538fcdd5efc3d612b"},
    "trim": {"count": 659, "width": 960, "height": 528, "crop": 12,
        "sha": "6cf2a322bc1365fadd05238bc18b813db40291af857aa0db77ff56458a6284fd"},
}


def base(source):
    return legacy.BASE if source == "0269" else ROOT/f"outputs/video-completo-{source}-20261026"


def cache_root(source):
    return legacy.CACHE if source == "0269" else TEMP/f"dronecrowd-full-{source}-20261026"


def profile(source, model):
    return f"dronecrowd-full{source}-{model}-v1"


def blocks(source):
    return (SOURCES[source]["count"]+15)//16


def header(source):
    s = SOURCES[source]
    return f"P6\n{s['width']} {s['height']}\n255\n".encode()


def prior_plan():
    return bounded.load_plan(PRIOR_PLAN, PRIOR_SHA)


def source_path(source):
    entry = next(s for s in prior_plan()["sources"] if s["video_id"] == source)
    manifest = ROOT/entry["source_manifest"]
    if bounded.sha256_file(manifest) != entry["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    data = json.loads(manifest.read_text())
    path = Path(data["source"]).resolve(strict=True)
    if data["source_sha256"] != SOURCES[source]["sha"] or bounded.sha256_file(path) != SOURCES[source]["sha"]:
        raise ValueError("Original source changed")
    return path


def decode(source, path, start, count, frames=None):
    s = SOURCES[source]
    filters = f"select=between(n\\,{start}\\,{start+count-1})"
    if s["crop"]:
        filters += f",crop={s['width']}:{s['height']}:0:0"
    seek = []
    if frames is not None:
        first, last = frames[start], frames[start+count-1]
        if Fraction(first["time_base"]) != Fraction(last["time_base"]):
            raise ValueError("Changing time base requires review")
        # Seek earlier than the target and select by original integer PTS.
        # RGB hashes independently verify accurate seeking and conversion.
        earlier = max(Fraction(0), first["pts"]*Fraction(first["time_base"])-1)
        seek = ["-ss", str(earlier.numerator//earlier.denominator), "-copyts"]
        filters = f"select=between(pts\\,{first['pts']}\\,{last['pts']})"
        if s["crop"]:
            filters += f",crop={s['width']}:{s['height']}:0:0"
    command = ["/usr/bin/timeout", "--kill-after=5", "110", "/usr/bin/ffmpeg", "-nostdin",
        "-v", "error", "-threads", "1", "-filter_threads", "1", *seek, "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-vf", filters, "-frames:v", str(count),
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads", "1",
        "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile(dir=TEMP) as errors:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=errors, preexec_fn=bounded._set_child_file_limit)
        try:
            for _ in range(count):
                payload = child.stdout.read(s["width"]*s["height"]*3)
                if len(payload) != s["width"]*s["height"]*3:
                    raise ValueError("Incomplete decoded RGB frame")
                yield payload
                del payload
            if child.stdout.read(1) or child.wait(timeout=10) != 0:
                raise ValueError("Extra decoded bytes or decoder failure")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()


def metadata(source):
    if source == "0269":
        raise ValueError("Keep the existing 0269 inventory")
    legacy.preflight()
    path = source_path(source)
    before = path.stat()
    metadata = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error", "-threads", "1",
        "-select_streams", "v:0", "-show_frames", "-show_streams", "-show_entries",
        "frame=pts:stream=width,height,time_base,nb_frames", "-of", "json", str(path)], timeout=110))
    stream, frames, s = metadata["streams"][0], metadata["frames"], SOURCES[source]
    if (stream["width"], stream["height"], int(stream["nb_frames"]), len(frames)) != (
            s["width"], s["height"]+s["crop"], s["count"], s["count"]):
        raise ValueError("Decoded inventory differs from the approved source")
    after = path.stat()
    stat = lambda st: [st.st_ino, st.st_size, st.st_mtime_ns]
    if stat(before) != stat(after):
        raise ValueError("Source changed during metadata decoding")
    records = []
    for ordinal, frame in enumerate(frames):
        pts = frame["pts"]
        if type(pts) is not int:
            raise ValueError("Original integer PTS required")
        records.append({"source_ordinal": ordinal, "pts": pts, "time_base": stream["time_base"],
            "pts_seconds": float(pts*Fraction(stream["time_base"])), "width": s["width"],
            "height": s["height"]})
    if any(a["pts"] >= b["pts"] for a, b in zip(records, records[1:])):
        raise ValueError("This bounded PTS-seek decoder requires strictly increasing source PTS")
    legacy.write_new(base(source)/"metadata.json", {"source": str(path), "source_sha256": s["sha"],
        "source_stat": stat(after), "frames": records})


def inventory_part(source, part):
    legacy.preflight()
    data = json.loads((base(source)/"metadata.json").read_text())
    frames = data["frames"]
    if type(part) is not int or not 0 <= part < (len(frames)+127)//128:
        raise ValueError("Inventory part outside source")
    path = source_path(source)
    info = path.stat()
    if [info.st_ino, info.st_size, info.st_mtime_ns] != data["source_stat"]:
        raise ValueError("Source changed after metadata")
    start, count = part*128, min(128, len(frames)-part*128)
    records = []
    for i, payload in enumerate(decode(source, path, start, count, frames)):
        digest = hashlib.sha256(header(source))
        digest.update(payload)
        records.append({**frames[start+i], "rgb_sha256": digest.hexdigest()})
        del payload
    legacy.write_new(base(source)/f"inventory-part-{part:03d}.json", {"frames": records,
        "metadata_sha256": bounded.sha256_file(base(source)/"metadata.json")})


def inventory(source):
    if source == "0269":
        raise ValueError("Keep the existing0269 inventory")
    data = json.loads((base(source)/"metadata.json").read_text())
    records = []
    for part in range((SOURCES[source]["count"]+127)//128):
        entry = json.loads((base(source)/f"inventory-part-{part:03d}.json").read_text())
        if entry["metadata_sha256"] != bounded.sha256_file(base(source)/"metadata.json"):
            raise ValueError("Inventory part metadata mismatch")
        records += entry["frames"]
    if (len(records) != SOURCES[source]["count"] or [{k: v for k, v in f.items() if k != "rgb_sha256"}
            for f in records] != data["frames"]):
        raise ValueError("Incomplete or reordered decoded inventory")
    references = {f["rgb_sha256"] for batch in prior_plan()["batches"] if batch["video_id"] == source
                  for f in json.loads((ROOT/batch["input_manifest"]).read_text())["frames"]}
    if not references or not references.issubset({f["rgb_sha256"] for f in records}):
        raise ValueError("Native RGB inventory does not contain every previous anchor")
    legacy.write_new(base(source)/"inventory.json", {**data,
        "frames": records, "decoded_frame_count": len(records),
        "matched_previous_rgb_anchors": len(references), "gpu_launched": False})


def freeze(source):
    if source == "0269":
        raise ValueError("Keep the original 0269 plan")
    inv = base(source)/"inventory.json"
    data = json.loads(inv.read_text())
    s = SOURCES[source]
    if data["source_sha256"] != s["sha"] or len(data["frames"]) != s["count"]:
        raise ValueError("Complete inventory required")
    old = prior_plan()
    code = {name: bounded.sha256_file(ROOT/name) for name in (*legacy.CODE_FILES, "scripts/night_video_campaign.py")}
    code.update({k: v for k, v in old["code_digests"].items()
                 if k.startswith("clip-source/") or k == bounded.MPCOUNT_UPSTREAM_KEY})
    plan = {"schema_version": 3, "project": "DroneCrowd", "source_id": source,
        "source_sha256": s["sha"], "frame_count": s["count"], "frames_per_process": 16,
        "gpu_batch_size": 1, "code_digests": code, "local_prerequisites": old["local_prerequisites"],
        "inventory_sha256": bounded.sha256_file(inv), "jobs": []}
    bounded.verify_code_digests(plan)
    for checkpoint in legacy.previous.CHECKPOINTS.values():
        if bounded.sha256_file(ROOT/checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint mismatch")
    for block in range(blocks(source)):
        records = [{**f, "index": i, "rgb_file": f"frame-{i:03d}.ppm"}
                   for i, f in enumerate(data["frames"][block*16:(block+1)*16])]
        path = base(source)/f"inputs/block-{block:03d}.json"
        legacy.write_new(path, {"source_sha256": s["sha"], "frames": records})
        input_sha = bounded.sha256_file(path)
        for model in MODELS:
            ck = legacy.previous.CHECKPOINTS[model]
            job_id = legacy.previous.canonical_hash({"model": model, "input_sha256": input_sha,
                "checkpoint_sha256": ck["sha256"], "code_digests": code})
            plan["jobs"].append({"job_id": job_id, "model": model, "block": block,
                "profile": profile(source, model), "input_manifest": str(path.relative_to(ROOT)),
                "input_sha256": input_sha, "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256"],
                "resources": legacy.RESOURCES, "rgb_cache": str(cache_root(source)/f"block-{block:03d}"),
                "output_dir": str((base(source)/f"results/{model}/block-{block:03d}").relative_to(ROOT))})
    legacy.write_new(base(source)/"plan.json", plan)


def load_plan(source, supplied_sha):
    path = base(source)/"plan.json"
    if source == "0269":
        if supplied_sha != ORIGINAL_PLAN_SHA:
            raise ValueError("0269 plan must remain frozen")
        return old_worker.load_plan(path, supplied_sha)
    if (not bounded.HEX64.fullmatch(supplied_sha) or path.is_symlink()
            or path.stat().st_size > 2**20 or bounded.sha256_file(path) != supplied_sha):
        raise ValueError("Plan hash/path/size mismatch")
    p = json.loads(path.read_text())
    s = SOURCES[source]
    if (p.get("schema_version"), p.get("project"), p.get("source_id"), p.get("source_sha256"),
            p.get("frame_count"), p.get("frames_per_process"), p.get("gpu_batch_size")) != (
            3, "DroneCrowd", source, s["sha"], s["count"], 16, 1):
        raise ValueError("Plan outside the nightly campaign")
    if bounded.sha256_file(base(source)/"inventory.json") != p.get("inventory_sha256"):
        raise ValueError("Inventory hash mismatch")
    inv = json.loads((base(source)/"inventory.json").read_text())
    if inv.get("source_sha256") != s["sha"] or len(inv.get("frames", [])) != s["count"]:
        raise ValueError("Incomplete inventory")
    jobs = p.get("jobs", [])
    if (len(jobs) != blocks(source)*3 or len({j["job_id"] for j in jobs}) != len(jobs)
            or {(j["block"], j["model"]) for j in jobs} != {
                (b, m) for b in range(blocks(source)) for m in MODELS}):
        raise ValueError("Jobs omitted or duplicated")
    for b in range(blocks(source)):
        frames = [{**f, "index": i, "rgb_file": f"frame-{i:03d}.ppm"}
                  for i, f in enumerate(inv["frames"][b*16:(b+1)*16])]
        manifest = json.loads((base(source)/f"inputs/block-{b:03d}.json").read_text())
        if manifest != {"source_sha256": s["sha"], "frames": frames}:
            raise ValueError("Block differs from inventory")
        for i, f in enumerate(frames):
            if (f["source_ordinal"] != b*16+i or type(f["pts"]) is not int
                    or (f["width"], f["height"]) != (s["width"], s["height"])
                    or f["pts_seconds"] != float(f["pts"]*Fraction(f["time_base"]))):
                raise ValueError("Invalid ordinal, geometry or original PTS")
    for job in jobs:
        select_job(source, p, job["job_id"])
    return p


def select_job(source, plan, job_id):
    if source == "0269":
        return old_worker.select_job(plan, job_id)
    matches = [j for j in plan["jobs"] if j["job_id"] == job_id]
    if len(matches) != 1 or not bounded.HEX64.fullmatch(job_id):
        raise ValueError("Unknown scientific job ID")
    job = matches[0]
    b, m = job["block"], job["model"]
    if type(b) is not int or not 0 <= b < blocks(source) or m not in MODELS:
        raise ValueError("Block or model outside campaign")
    ck = legacy.previous.CHECKPOINTS[m]
    expected = {"profile": profile(source, m), "resources": legacy.RESOURCES,
        "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256"],
        "input_manifest": str((base(source)/f"inputs/block-{b:03d}.json").relative_to(ROOT)),
        "rgb_cache": str(cache_root(source)/f"block-{b:03d}"),
        "output_dir": str((base(source)/f"results/{m}/block-{b:03d}").relative_to(ROOT))}
    if any(job.get(k) != v for k, v in expected.items()):
        raise ValueError("Job outside fixed campaign paths/resources")
    if job_id != legacy.previous.canonical_hash({"model": m, "input_sha256": job["input_sha256"],
            "checkpoint_sha256": ck["sha256"], "code_digests": plan["code_digests"]}):
        raise ValueError("Scientific identity mismatch")
    if bounded.sha256_file(ROOT/job["input_manifest"]) != job["input_sha256"]:
        raise ValueError("Input digest mismatch")
    return job


def completed(source, output, job, manifest, plan):
    if source == "0269":
        return old_worker.completed(output, job, manifest, plan)
    if not bounded.completed_output(output, job, manifest, plan):
        return False
    report = json.loads((Path(output)/"report.json").read_text())
    frames = json.loads(Path(manifest).read_text())["frames"]
    return (report.get("source_sha256") == SOURCES[source]["sha"]
            and len(frames) == min(16, SOURCES[source]["count"]-job["block"]*16)
            and all(f["source_ordinal"] == job["block"]*16+i
                and all(r.get(k) == f[k] for k in ("source_ordinal", "pts", "time_base"))
                for i, (f, r) in enumerate(zip(frames, report["results"]))))


def cache_path(source, block):
    root = cache_root(source)
    path = root/f"block-{block:03d}"
    if (type(block) is not int or not 0 <= block < blocks(source) or root.is_symlink()
            or path.is_symlink() or not root.resolve().is_relative_to(TEMP.resolve())
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Unsafe cache path")
    return path


def validate_cache(source, block, manifest):
    path = cache_path(source, block)
    frames = manifest["frames"]
    names = {f"frame-{i:03d}.ppm" for i in range(len(frames))}
    if not path.is_dir() or {f.name for f in path.iterdir()} != names:
        raise ValueError("Partial or unexpected cache; preserve for review")
    s = SOURCES[source]
    for i, f in enumerate(frames):
        file = path/f"frame-{i:03d}.ppm"
        if (f["rgb_file"] != file.name or file.is_symlink() or not file.is_file()
                or file.stat().st_size != len(header(source))+s["width"]*s["height"]*3
                or bounded.sha256_file(file) != f["rgb_sha256"]):
            raise ValueError("Cache file mismatch")
    return path


def prepare(source, block, plan_sha):
    plan = load_plan(source, plan_sha)
    legacy.preflight()
    if source == "0269":
        return legacy.prepare(block)
    target = cache_path(source, block)
    if target.exists():
        raise FileExistsError("Existing cache must be verified, never overwritten")
    data = json.loads((base(source)/"inventory.json").read_text())
    path = source_path(source)
    info = path.stat()
    if [info.st_ino, info.st_size, info.st_mtime_ns] != data["source_stat"]:
        raise ValueError("Source identity changed")
    target.mkdir(parents=True)
    frames = json.loads((ROOT/next(j for j in plan["jobs"] if j["block"] == block)["input_manifest"]).read_text())["frames"]
    for i, (payload, frame) in enumerate(zip(decode(source, path, block*16, len(frames), data["frames"]), frames)):
        digest = hashlib.sha256(header(source))
        digest.update(payload)
        if digest.hexdigest() != frame["rgb_sha256"]:
            raise ValueError("Decoded RGB differs from inventory")
        with (target/f"frame-{i:03d}.ppm").open("xb") as handle:
            handle.write(header(source))
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        del payload
    validate_cache(source, block, {"frames": frames})


def migration(plan, sha):
    path = base("0269")/"night-migration.json"
    if (not bounded.HEX64.fullmatch(sha or "") or path.is_symlink()
            or path.stat().st_size > 256*1024 or bounded.sha256_file(path) != sha):
        raise ValueError("Pinned migration manifest required")
    data = json.loads(path.read_text())
    validate_reuse(data, plan, hash_maps=False)
    return validate_migration(data, plan)


def validate_reuse(data, plan, *, hash_maps=True):
    expected = {j["job_id"]: j for j in plan["jobs"] if j["block"] < 4}
    ledger = data.get("reused_results", [])
    if (len(expected) != 12 or len(ledger) != 12
            or {r.get("scientific_job_id") for r in ledger} != set(expected)):
        raise ValueError("Reuse ledger must contain exactly the twelve original jobs")
    total = 0
    for entry in ledger:
        job = expected[entry["scientific_job_id"]]
        receipt = entry["original_receipt"]
        output = ROOT/job["output_dir"]
        report_path = output/"report.json"
        if (receipt.get("key") != "DroneCrowd:"+job["job_id"]
                or receipt.get("request_id") != job["job_id"] or receipt.get("project") != "DroneCrowd"
                or receipt.get("profile") != job["profile"] or receipt.get("state") != "completed"
                or receipt.get("exit_code") != 0 or receipt.get("started") is None
                or entry.get("input_sha256") != job["input_sha256"]
                or bounded.sha256_file(ROOT/job["input_manifest"]) != job["input_sha256"]
                or entry.get("checkpoint_sha256") != job["checkpoint_sha256"]
                or bounded.sha256_file(report_path) != entry.get("report_sha256")):
            raise ValueError("Reused report, receipt or provenance differs from pinned ledger")
        report = json.loads(report_path.read_text())
        maps = [{"file": f["density_file"], "sha256": f["density_sha256"],
                 "source_ordinal": f["source_ordinal"]} for f in report["results"]]
        if (entry.get("maps") != maps or len(maps) != 16
                or [f["source_ordinal"] for f in maps] != list(range(job["block"]*16, (job["block"]+1)*16))):
            raise ValueError("Reused map inventory differs from pinned ledger")
        if hash_maps and not completed("0269", output, job, ROOT/job["input_manifest"], plan):
            raise ValueError("Reused density bytes or report identity changed")
        total += len(maps)
    if total != 192:
        raise ValueError("Exactly192reused maps required")


def validate_migration(data, plan):
    if data.get("plan_sha256") != ORIGINAL_PLAN_SHA or data.get("project") != "DroneCrowd":
        raise ValueError("Migration owner or original plan mismatch")
    pairs = data.get("replacements", [])
    if len(pairs) != 3:
        raise ValueError("Exactly three cancelled unstarted requests may be replaced")
    mapping = {}
    for item in pairs:
        old, new, receipt = item["scientific_job_id"], item["request_id"], item["original_receipt"]
        job = old_worker.select_job(plan, old)
        expected_new = hashlib.sha256(("supersedes:DroneCrowd:"+old).encode()).hexdigest()
        if (job["block"] != 4 or old == new or new != expected_new or not bounded.HEX64.fullmatch(new)
                or old in mapping or new in mapping.values()
                or new in {j["job_id"] for j in plan["jobs"]}
                or receipt.get("key") != "DroneCrowd:"+old
                or receipt.get("project") != "DroneCrowd" or receipt.get("request_id") != old
                or receipt.get("profile") != job["profile"] or receipt.get("state") != "cancelled"
                or "started" not in receipt or receipt["started"] is not None or receipt.get("launcher_pid") is not None
                or receipt.get("launcher_start") is not None):
            raise ValueError("Replacement is not a unique cancelled never-started job")
        mapping[old] = new
    if {old_worker.select_job(plan, k)["model"] for k in mapping} != set(MODELS):
        raise ValueError("One replacement per model required")
    return mapping


def prepare_migration(dispatcher):
    plan = load_plan("0269", ORIGINAL_PLAN_SHA)
    rows = {r["key"]: r for r in legacy.queue_call(dispatcher, "status") if r["project"] == "DroneCrowd"}
    data = {"project": "DroneCrowd", "plan_sha256": ORIGINAL_PLAN_SHA,
        "authorization": "Owner exclusive overnight run 2026-10-26; SPEC019",
        "replacements": [], "reused_results": [], "created_unix": time.time()}
    for job in plan["jobs"]:
        old = job["job_id"]
        row = rows.get("DroneCrowd:"+old)
        output = ROOT/job["output_dir"]
        if job["block"] < 4:
            if not row or row["state"] != "completed" or not completed(
                    "0269", output, job, ROOT/job["input_manifest"], plan):
                raise ValueError("Previously completed job or maps no longer verify")
            report = json.loads((output/"report.json").read_text())
            data["reused_results"].append({"scientific_job_id": old, "original_receipt": row,
                "input_sha256": job["input_sha256"], "checkpoint_sha256": job["checkpoint_sha256"],
                "report_sha256": bounded.sha256_file(output/"report.json"),
                "maps": [{"file": f["density_file"], "sha256": f["density_sha256"],
                          "source_ordinal": f["source_ordinal"]} for f in report["results"]]})
        elif job["block"] == 4:
            if output.exists() or not row:
                raise ValueError("Never-started replacement requires absent output and original receipt")
            data["replacements"].append({"scientific_job_id": old,
                "request_id": hashlib.sha256(("supersedes:DroneCrowd:"+old).encode()).hexdigest(),
                "original_receipt": row})
        elif row is not None or output.exists():
            raise ValueError("Unexpected later attempt; review before freezing migration")
    validate_migration(data, plan)
    if len(data["reused_results"]) != 12 or sum(len(r["maps"]) for r in data["reused_results"]) != 192:
        raise ValueError("Reuse ledger must preserve twelve complete jobs and192maps")
    legacy.write_new(base("0269")/"night-migration.json", data)
    print(json.dumps({"migration_sha256": bounded.sha256_file(base("0269")/"night-migration.json"),
                      "reused_jobs": 12, "reused_maps": 192}))


def scientific_id(request_id, mapping, plan):
    reverse = {v: k for k, v in mapping.items()}
    if request_id in mapping:
        raise ValueError("Cancelled original request must remain terminal")
    job_id = reverse.get(request_id, request_id)
    if not any(j["job_id"] == job_id for j in plan["jobs"]):
        raise ValueError("Unapproved scheduler request")
    return job_id


def work(source, plan_sha, model, migration_sha):
    plan = load_plan(source, plan_sha)
    mapping = migration(plan, migration_sha) if source == "0269" else {}
    request_id = os.environ.get("NEXO_GPU_REQUEST_ID", "")
    job_id = scientific_id(request_id, mapping, plan)
    job = select_job(source, plan, job_id)
    if (os.environ.get("NEXO_GPU_PROJECT") != "DroneCrowd"
            or os.environ.get("NEXO_GPU_PROFILE") != job["profile"] or job["model"] != model):
        raise ValueError("Dispatcher owner/profile/model mismatch")
    if not completed(source, ROOT/job["output_dir"], job, ROOT/job["input_manifest"], plan):
        validate_cache(source, job["block"], json.loads((ROOT/job["input_manifest"]).read_text()))
    unit, directory = bounded.cgroup_scope(Path("/proc/self/cgroup").read_text())
    subprocess.run(["systemctl", "--user", "set-property", "--runtime", unit, "CPUQuota=100%"],
                   check=True, capture_output=True, timeout=5)
    quota, period = (directory/"cpu.max").read_text().split()
    if not quota.isdigit() or int(quota) > int(period):
        raise ValueError("Worker CPU quota exceeds one core")
    result = bounded.run_job(plan_path=base(source)/"plan.json", plan_sha256=plan_sha,
        job_id=job_id, environment_lock=ROOT/"ops/gpu-candidates.lock.json",
        plan_loader=lambda path, sha, **kw: load_plan(source, sha),
        job_selector=lambda p, j: select_job(source, p, j), command_builder=old_worker.command,
        completion_checker=lambda *args: completed(source, *args))
    print(json.dumps({**result, "scheduler_request_id": request_id, "scientific_job_id": job_id}))


def cpu_command(quota_runner, source, action, plan_sha=None, block=None):
    argv = ["/usr/bin/python3", str(quota_runner), "run", "--project", "DroneCrowd",
        "--ram-gib", "0.75", "--cpu-percent", "100", "--", "/usr/bin/timeout", "--kill-after=5",
        "120", str(ROOT/".venv/gpu-candidates/bin/python"), str(ROOT/"scripts/night_video_campaign.py"),
        action, "--source", source]
    if plan_sha:
        argv += ["--plan-sha256", plan_sha]
    if block is not None:
        argv += ["--block", str(block)]
    return argv


def validate_maps(source, block, plan_sha):
    import numpy as np
    plan = load_plan(source, plan_sha)
    verified = []
    for job in (j for j in plan["jobs"] if j["block"] == block):
        output, manifest = ROOT/job["output_dir"], ROOT/job["input_manifest"]
        if not completed(source, output, job, manifest, plan):
            raise ValueError("Incomplete output before map validation")
        report = json.loads((output/"report.json").read_text())
        stride = 1 if job["model"] == "mpcount" else 8
        for row in report["results"]:
            density = np.load(output/row["density_file"], allow_pickle=False)
            if (density.dtype != np.float32 or density.shape != (row["height"]//stride, row["width"]//stride)
                    or not np.isfinite(density).all()
                    or abs(float(density.sum(dtype=np.float64))-row["count"]) > 1e-7
                    or abs(float(np.minimum(density, 0).sum(dtype=np.float64))-row["negative_mass"]) > 1e-7):
                raise ValueError("Invalid density geometry, values or integral")
            verified.append({"model": job["model"], "source_ordinal": row["source_ordinal"],
                             "density_sha256": row["density_sha256"]})
            del density
    if len(verified) != 3*min(16, SOURCES[source]["count"]-block*16):
        raise ValueError("Missing maps")
    legacy.write_new(base(source)/f"validated-block-{block:03d}.json", {"status": "verified", "maps": verified})


def prediction_rows(source, plan, *, verify_outputs=True):
    rows = {}
    for job in plan["jobs"]:
        if verify_outputs and not completed(source, ROOT/job["output_dir"], job, ROOT/job["input_manifest"], plan):
            raise ValueError("Cannot summarize an incomplete campaign")
        report = json.loads((ROOT/job["output_dir"]/"report.json").read_text())
        for f in report["results"]:
            if not math.isfinite(f["count"]):
                raise ValueError("Nonfinite count")
            row = rows.setdefault(f["source_ordinal"], {"source": source, "source_ordinal": f["source_ordinal"],
                "pts": f["pts"], "time_base": f["time_base"], "pts_seconds": f["pts_seconds"],
                "rgb_sha256": f["rgb_sha256"]})
            if job["model"] in row:
                raise ValueError("Duplicate prediction")
            row[job["model"]] = f["count"]
    if set(rows) != set(range(SOURCES[source]["count"])) or any(not set(MODELS).issubset(r) for r in rows.values()):
        raise ValueError("Missing predictions")
    return [rows[i] for i in range(len(rows))]


def counts_payload(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def summarize(source, plan):
    rows = prediction_rows(source, plan)
    payload = counts_payload(rows)
    path = base(source)/"counts.csv"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("Existing CSV differs from verified reports; preserve for review")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
    provenance = {"source": source, "plan_sha256": bounded.sha256_file(base(source)/"plan.json"),
        "counts_sha256": bounded.sha256_file(path), "reports": [{"path": j["output_dir"]+"/report.json",
            "sha256": bounded.sha256_file(ROOT/j["output_dir"]/"report.json")} for j in plan["jobs"]]}
    proof = base(source)/"counts-provenance.json"
    if proof.exists():
        if json.loads(proof.read_text()) != provenance:
            raise ValueError("Existing CSV provenance differs")
    else:
        legacy.write_new(proof, provenance)
    legacy.write_state(base(source)/"state.json", {"state": "complete", "completed_blocks": blocks(source),
        "completed_jobs": len(plan["jobs"]), "frames": len(rows), "predictions": len(rows)*3,
        "time_unix": time.time()})


def validate_transition(plan, mapping, dispatcher):
    """Measure new blocks4/5; old block0 is not evidence of RAM/API migration."""
    rows = {r["key"]: r for r in legacy.queue_call(dispatcher, "status") if r["project"] == "DroneCrowd"}
    ledger = json.loads((base("0269")/"night-migration.json").read_text())
    checks = []
    for job in (j for j in plan["jobs"] if j["block"] in (4, 5)):
        rid = mapping.get(job["job_id"], job["job_id"])
        receipt = rows.get("DroneCrowd:"+rid, {})
        report_path = ROOT/job["output_dir"]/"report.json"
        report = json.loads(report_path.read_text())
        limits = report["memory_limits"]
        if (receipt.get("state") != "completed" or receipt.get("exit_code") != 0
                or receipt.get("started", 0) < ledger["created_unix"]
                or receipt.get("ended", 0) <= receipt["started"]
                or tuple(limits.get(k) for k in ("memory.high", "memory.max", "memory.swap.max")) != (
                    "1610612736", "2147483648", "536870912")
                or report["total_seconds"] >= 110 or report["max_rss_kib"]*1024 > 1610612736):
            raise ValueError("New RAM/API transition lacks runtime, memory or receipt margin")
        checks.append({"block": job["block"], "model": job["model"], "request_id": rid,
            "scientific_job_id": job["job_id"], "receipt": receipt,
            "report_sha256": bounded.sha256_file(report_path), "total_seconds": report["total_seconds"],
            "max_rss_kib": report["max_rss_kib"], "peak_cuda_bytes": report["peak_cuda_bytes"]})
    if len(checks) != 6:
        raise ValueError("Transition requires two new cycles and six results")
    legacy.write_state(base("0269")/"night-transition.json", {"status": "passed", "checks": checks,
        "time_unix": time.time(), "original_plan_sha256": ORIGINAL_PLAN_SHA})


def coordinate(source, plan_sha, migration_sha, dispatcher, quota_runner, deadline_unix=None):
    plan = load_plan(source, plan_sha)
    bounded.verify_code_digests(plan)
    mapping = migration(plan, migration_sha) if source == "0269" else {}
    if source == "0269":
        validate_reuse(json.loads((base(source)/"night-migration.json").read_text()), plan)
    scheduler_plan = {"jobs": [{"job_id": mapping.get(j["job_id"], j["job_id"])} for j in plan["jobs"]]}
    submitted = []
    with (base(source)/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for block in range(blocks(source)):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError("Authorized overnight GPU window expired")
                jobs = [j for j in plan["jobs"] if j["block"] == block]
                complete = [completed(source, ROOT/j["output_dir"], j, ROOT/j["input_manifest"], plan) for j in jobs]
                for job, done in zip(jobs, complete):
                    if not done and (ROOT/job["output_dir"]).exists():
                        raise FileExistsError("Partial scientific output preserved; no automatic retry")
                if not all(complete):
                    legacy.preflight()
                    cache = cache_path(source, block)
                    if not cache.exists():
                        subprocess.run(cpu_command(quota_runner, source, "prepare", plan_sha, block),
                                       check=True, timeout=135)
                    validate_cache(source, block, json.loads((ROOT/jobs[0]["input_manifest"]).read_text()))
                    before = {r["key"]: r for r in legacy.queue_call(dispatcher, "status")}
                    for job, done in zip(jobs, complete):
                        if done:
                            continue
                        rid = mapping.get(job["job_id"], job["job_id"])
                        key = "DroneCrowd:"+rid
                        if job["job_id"] in mapping:
                            row = before.get(key)
                            if (not row or row.get("supersedes") not in (
                                    "DroneCrowd:"+job["job_id"], job["job_id"])):
                                raise ValueError("Nexo replacement receipt with supersedes required")
                        else:
                            row = legacy.queue_call(dispatcher, "submit", "--project", "DroneCrowd",
                                "--profile", job["profile"], "--request-id", rid)
                        if row.get("key") != key or row.get("profile") != job["profile"]:
                            raise ValueError("Dispatcher receipt identity mismatch")
                        submitted.append(key)
                        legacy.write_state(base(source)/f"night-receipt-{job['model']}-{block:03d}.json", row)
                    deadline = time.monotonic()+1800
                    while True:
                        rows = {r["key"]: r for r in legacy.queue_call(dispatcher, "status")}
                        selected = [rows.get("DroneCrowd:"+mapping.get(j["job_id"], j["job_id"]))
                                    for j, done in zip(jobs, complete) if not done]
                        legacy.write_state(base(source)/"state.json", {"state": "waiting", "block": block,
                            "completed_blocks": block, "requests": selected, "time_unix": time.time()})
                        if any(r and r["state"] not in ("queued", "waiting_profile", "running", "recovering", "completed") for r in selected):
                            raise RuntimeError("Dispatcher request terminal or requires review; no retry")
                        if selected and all(r and r["state"] == "completed" for r in selected):
                            break
                        if time.monotonic() > deadline or (deadline_unix is not None and time.time() >= deadline_unix):
                            raise TimeoutError("No completed block within 30 minutes")
                        time.sleep(5)
                if not all(completed(source, ROOT/j["output_dir"], j, ROOT/j["input_manifest"], plan) for j in jobs):
                    raise RuntimeError("Queue success without verified scientific outputs")
                if not (base(source)/f"validated-block-{block:03d}.json").exists():
                    subprocess.run(cpu_command(quota_runner, source, "validate", plan_sha, block),
                                   check=True, timeout=135)
                if block == 0:
                    checks = {j["model"]: legacy.validate_anchor(ROOT/j["output_dir"], ROOT/j["input_manifest"]) for j in jobs}
                    legacy.write_state(base(source)/"night-pilot.json", {"checks": checks, "status": "passed", "plan_sha256": plan_sha})
                if source == "0269" and block == 5:
                    validate_transition(plan, mapping, dispatcher)
                cache = cache_path(source, block)
                if cache.exists():
                    manifest = json.loads((ROOT/jobs[0]["input_manifest"]).read_text())
                    validate_cache(source, block, manifest)
                    for frame in manifest["frames"]:
                        (cache/frame["rgb_file"]).unlink()
                    cache.rmdir()
                legacy.write_state(base(source)/"state.json", {"state": "running", "completed_blocks": block+1,
                    "completed_jobs": (block+1)*3, "time_unix": time.time()})
            summarize(source, plan)
        except BaseException as error:
            release = legacy.stop_requests(dispatcher, scheduler_plan, submitted)
            legacy.write_state(base(source)/"state.json", {"state": "stopped", "error": str(error)[:1024],
                "submitted_keys": submitted, **release, "time_unix": time.time()})
            raise


def profiles(migration_sha):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("Commit the reviewed implementation first")
    proposals = {}
    for source in SOURCES:
        path = base(source)/"plan.json"
        sha = bounded.sha256_file(path)
        plan = load_plan(source, sha)
        bounded.verify_code_digests(plan)
        if source == "0269":
            migration(plan, migration_sha)
        for model in MODELS:
            argv = [str(ROOT/".venv/gpu-candidates/bin/python"), str(ROOT/"scripts/night_video_campaign.py"),
                "work", "--source", source, "--plan-sha256", sha, "--model", model]
            if source == "0269":
                argv += ["--migration-sha256", migration_sha]
            proposals[profile(source, model)] = {"project": "DroneCrowd", "revision": revision,
                "entrypoint": "scripts/night_video_campaign.py", "environment_lock": "ops/gpu-candidates.lock.json",
                "argv": argv, "ram_gib": 1.5, "vram_mib": 1024, "scratch_gib": .5, "outputs_gib": 1,
                "seconds": 120, "max_runs": blocks(source), "priority": "exploratory",
                "approval": "Owner authorizes exclusive overnight three-video campaign 2026-10-26; SPEC019; preserve 12 completed0269 jobs, no retries, batch1, reviewed immutable plans."}
    path = ROOT/"outputs/campana-nocturna-20261026/gpu-profiles.json"
    legacy.write_new(path, proposals)
    print(json.dumps({"profiles": str(path), "revision": revision, "status": "proposed_not_applied"}))


def main():
    bounded.set_runtime_environment()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["metadata", "inventory-part", "inventory", "freeze", "prepare", "validate", "coordinate", "work", "profiles", "prepare-migration", "all"])
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--plan-sha256")
    p.add_argument("--migration-sha256")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--block", type=int)
    p.add_argument("--dispatcher", type=Path)
    p.add_argument("--quota-runner", type=Path)
    p.add_argument("--deadline-unix", type=float)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--bundle-sha256")
    a = p.parse_args()
    def interrupted(*_):
        raise InterruptedError("Night campaign received termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    if a.action == "all":
        if (not a.bundle or not a.bundle_sha256 or not a.deadline_unix or not a.dispatcher or not a.quota_runner
                or not time.time() < a.deadline_unix <= time.time()+8*3600
                or bounded.sha256_file(a.bundle) != a.bundle_sha256):
            p.error("Pinned bundle, dispatcher, quota runner and <=8h deadline required")
        bundle = json.loads(a.bundle.read_text())
        if [x["source"] for x in bundle["campaigns"]] != list(SOURCES):
            raise ValueError("Bundle must contain0269,0265,trim in approved order")
        for entry in bundle["campaigns"]:
            coordinate(entry["source"], entry["plan_sha256"], bundle["migration_sha256"],
                       a.dispatcher, a.quota_runner, a.deadline_unix)
        # CPU-only deliverables follow verified inference; they do not submit GPU.
        review = ROOT/"scripts/render_full_video_review.py"
        python = ROOT/".venv/gpu-candidates/bin/python"
        prefix = ["/usr/bin/python3", str(a.quota_runner), "run", "--project", "DroneCrowd",
                  "--ram-gib", "0.75", "--cpu-percent", "100", "--", "/usr/bin/timeout", "--kill-after=5"]
        subprocess.run([*prefix, "120", str(python), str(review), "combine"], check=True, timeout=135)
        for source in SOURCES:
            if a.deadline_unix-time.time() < 630:
                legacy.write_state(ROOT/"outputs/campana-nocturna-20261026/render-pending.json",
                    {"source": source, "reason": "Window lacks630seconds for bounded CPU render"})
                break
            subprocess.run([*prefix, "610", str(python), str(review), "render", "--source", source],
                           check=True, timeout=625)
        return
    if a.action == "prepare-migration":
        if not a.dispatcher:
            p.error("--dispatcher required")
        prepare_migration(a.dispatcher)
        return
    if a.action == "profiles":
        profiles(a.migration_sha256)
        return
    if not a.source:
        p.error("--source required")
    if a.action == "metadata":
        metadata(a.source)
    elif a.action == "inventory-part":
        inventory_part(a.source, a.block)
    elif a.action == "inventory":
        inventory(a.source)
    elif a.action == "freeze":
        freeze(a.source)
    elif a.action in ("prepare", "validate"):
        if a.block is None or not a.plan_sha256:
            p.error("--block and --plan-sha256 required")
        (prepare if a.action == "prepare" else validate_maps)(a.source, a.block, a.plan_sha256)
    elif a.action == "work":
        try:
            work(a.source, a.plan_sha256, a.model, a.migration_sha256)
        except Exception as error:
            evidence = bounded.write_preflight_failure(error, plan_sha256=a.plan_sha256,
                job_id=os.environ.get("NEXO_GPU_REQUEST_ID", ""))
            print(json.dumps({"status": "failed", "error_type": type(error).__name__, "evidence_file": str(evidence)}))
            return 1
    else:
        if not a.dispatcher or not a.quota_runner or not a.plan_sha256:
            p.error("dispatcher, quota runner and plan digest required")
        coordinate(a.source, a.plan_sha256, a.migration_sha256, a.dispatcher, a.quota_runner, a.deadline_unix)


if __name__ == "__main__":
    sys.exit(main())
