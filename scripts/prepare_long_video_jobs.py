"""Prepare local, content-addressed jobs for the native long-video dispatcher.

This client only plans and samples inputs. It never contacts or submits to an
executor. Inference remains the responsibility of a separately registered queue.
"""
import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/home/alephbet/home.codex/tmp")
SOURCE_MANIFESTS = (
    ("0265", "outputs/resolucion-original/0265/entrada-native/input.json"),
    ("0269", "outputs/resolucion-original/0269/entrada-native/input.json"),
    ("trim", "outputs/video-breve/20261024-trim/entrada/input.json"),
)
CHECKPOINTS = {
    "sr": {"path": "outputs/modelos/SR/dronecrowd.tar",
           "sha256": "22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb"},
    "clip-ebc": {"path": "outputs/modelos/CLIP-EBC/resnet50/best_mae.pth",
                 "sha256": "da40b664a0a5a57bc33235e7334403e5000271e0741134e301ea706d754bf7df"},
    "mpcount": {"path": "outputs/modelos/MPCount/sta_deterministic.pth",
                "sha256": "4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0"},
}
CLIP_SOURCE = TMP_ROOT / "dronecrowd-models-audit/clip/source/Yiming-M-CLIP-EBC-b52a594"
CLIP_SOURCE_COMMIT = "b52a5944bd75beed34fd3d0da516aa833ae63354"
MPCOUNT_SOURCE_COMMIT = "6eb06772bcf7dfb771c43a14d67146fce767f103"
MPCOUNT_UPSTREAM_MODEL_SHA256 = "deb0550ee49c0fa8851372f2ff6469e150f75fdbde1b16be49b6492b5e20a176"
CLIP_PYTHON = TMP_ROOT / "dronecrowd-models-env/bin/python"
CLIP_MANIFEST = "ops/clip-ebc-sources.json"
CODE_FILES = (
    "scripts/prepare_long_video_jobs.py", "scripts/evaluate_crowd_candidates.py",
    "scripts/sample_video.py", "scripts/evaluate_sr.py", "scripts/sr_model.py",
    "scripts/clip_ebc_model.py", "scripts/mpcount_model.py",
    "ops/clip-ebc-sources.json", "third_party/mpcount-LICENSE",
)
MAX_FRAMES_PER_BATCH = 2
MAX_BATCH_BYTES = 120 * 2**20
MIN_RETAINED_DISK_BYTES = 20 * 2**30
ESTIMATED_OUTPUT_BYTES = 4 * 2**30
DISK_ESTIMATE_MARGIN = 0.30
RESOURCE_PROFILE = {"ram_limit_bytes": 2*2**30, "swap_limit_bytes": 512*2**20,
                    "expected_ram_bytes": int(1.5*2**30), "expected_vram_bytes": 1*2**30,
                    "max_seconds": 120, "cpu_threads": 1,
                    "disk_increment_bytes": 256*2**20}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_batches(times, max_frames=MAX_FRAMES_PER_BATCH):
    if max_frames != MAX_FRAMES_PER_BATCH:
        raise ValueError("The prepared dispatcher plan fixes batches at at most two frames")
    return [list(times[index:index+max_frames]) for index in range(0, len(times), max_frames)]


def integer_sample_times(duration):
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source duration must be finite and positive")
    return list(range(math.ceil(duration)))


def plan_batches(video_id, duration, width, height, crop_last_rows=0):
    times = integer_sample_times(duration)
    target_width, target_height = width, height-crop_last_rows
    if (crop_last_rows < 0 or target_height < 32 or width % 16 or target_height % 16
            or not 32 <= width <= 3840 or not 32 <= height <= 2160):
        raise ValueError("Video/crop geometry must be bounded and align to sixteen")
    batches = []
    for batch_index, batch_times in enumerate(split_batches(times)):
        # Match sample_video's conservative raw-RGB/preview budget per batch.
        estimate = len(batch_times)*(2*width*height*3+65536)
        if estimate > MAX_BATCH_BYTES:
            raise ValueError("Batch exceeds the sampler's 120 MiB output budget")
        batches.append({"batch_id": f"{video_id}-batch-{batch_index:03d}",
                        "video_id": video_id, "batch_index": batch_index,
                        "frame_times_seconds": batch_times,
                        "frame_count": len(batch_times),
                        "native_dimensions": [width, height],
                        "effective_dimensions": [target_width, target_height],
                        "crop_xyxy": [0, 0, width, target_height] if crop_last_rows else None,
                        "cropped_bottom_rows": crop_last_rows,
                        "sampler_output_budget_estimate_bytes": estimate})
    return batches


def idempotency_key(model, input_sha256, checkpoint_sha256, code_digests):
    if not input_sha256:
        raise ValueError("An idempotency key requires a materialized input SHA-256")
    return canonical_hash({"model": model, "input_sha256": input_sha256,
                           "checkpoint_sha256": checkpoint_sha256,
                           "code_digests": code_digests})


def _read_mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])*1024
    raise RuntimeError("Could not determine host MemAvailable")


def preflight(output_dir, *, free_bytes=None, available_ram_bytes=None):
    free_bytes = shutil.disk_usage(output_dir).free if free_bytes is None else free_bytes
    available_ram_bytes = _read_mem_available() if available_ram_bytes is None else available_ram_bytes
    required_disk = MIN_RETAINED_DISK_BYTES + int(ESTIMATED_OUTPUT_BYTES*(1+DISK_ESTIMATE_MARGIN))
    if free_bytes < required_disk:
        raise RuntimeError("Preflight requires 20 GiB retained plus estimated outputs and 30% margin (25.2 GiB total)")
    if available_ram_bytes < 2*2**30:
        raise RuntimeError("Preflight requires at least 2 GiB available host RAM")
    return {"free_disk_bytes": free_bytes, "available_ram_bytes": available_ram_bytes,
            "minimum_retained_disk_bytes": MIN_RETAINED_DISK_BYTES,
            "estimated_additional_artifacts_bytes": ESTIMATED_OUTPUT_BYTES,
            "disk_estimate_margin_fraction": DISK_ESTIMATE_MARGIN,
            "minimum_free_disk_bytes": required_disk,
            "minimum_available_ram_bytes": 2*2**30}


def _load_source_specs(repo_root=ROOT, verify_videos=True):
    from sample_video import sha256 as sampler_sha256
    specs = []
    for video_id, relative_manifest in SOURCE_MANIFESTS:
        manifest_path = (repo_root/relative_manifest).resolve(strict=True)
        if not manifest_path.is_relative_to(repo_root.resolve()):
            raise ValueError("Source manifest escaped repository")
        manifest_digest = sampler_sha256(manifest_path)
        data = json.loads(manifest_path.read_text())
        source_path = Path(data["source"]).resolve(strict=True)
        expected_digest = data["source_sha256"]
        before = source_path.stat()
        actual_digest = sampler_sha256(source_path) if verify_videos else expected_digest
        after = source_path.stat()
        if actual_digest != expected_digest:
            raise ValueError(f"Source video hash differs from frozen input: {video_id}")
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source video changed during verification: {video_id}")
        metadata = data["metadata"]
        stream = metadata["streams"][0]
        fmt = metadata["format"]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(fmt["duration"])
        crop = 12 if video_id == "trim" else 0
        if video_id == "trim" and (width, height) != (960, 540):
            raise ValueError("The frozen trim source must be 960x540 before its 12-row crop")
        specs.append({"video_id": video_id, "source_manifest": relative_manifest,
            "source_manifest_sha256": manifest_digest, "source_sha256": expected_digest,
            "source_bytes": before.st_size, "duration_seconds": duration,
            "width": width, "height": height, "source_path_local": str(source_path),
            "crop_last_rows": crop,
            "batches": plan_batches(video_id, duration, width, height, crop)})
    return specs


def _collect_code_digests(repo_root=ROOT):
    digests = {}
    for relative in CODE_FILES:
        path = repo_root/relative
        if not path.is_file():
            raise FileNotFoundError(f"Required local runtime code missing: {relative}")
        digests[relative] = sha256_file(path)
    # The adapted state-dict-compatible model is tied to this exact upstream source.
    digests[f"upstream/mpcount/models/models.py@{MPCOUNT_SOURCE_COMMIT}"] = MPCOUNT_UPSTREAM_MODEL_SHA256
    clip_manifest_path = repo_root/CLIP_MANIFEST
    clip_manifest = json.loads(clip_manifest_path.read_text())
    if clip_manifest.get("commit") != CLIP_SOURCE_COMMIT:
        raise ValueError("Pinned CLIP-EBC source manifest commit changed")
    clip_files = {}
    for name, expected in clip_manifest["files"].items():
        source_file = CLIP_SOURCE/name
        if not source_file.is_file() or sha256_file(source_file) != expected:
            raise ValueError(f"Pinned local CLIP-EBC source missing or changed: {name}")
        clip_files[f"clip-source/{name}"] = expected
    digests.update(clip_files)
    return digests


def _load_model_specs(repo_root=ROOT):
    models = {}
    for model, spec in CHECKPOINTS.items():
        path = (repo_root/spec["path"]).resolve(strict=True)
        if not path.is_relative_to(repo_root.resolve()) or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Frozen checkpoint missing or hash mismatch: {model}")
        models[model] = dict(spec)
    if not CLIP_PYTHON.is_file():
        raise FileNotFoundError("Pinned offline CLIP-EBC Python environment is unavailable")
    return models


def _git_revision(repo_root=ROOT):
    result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def _source_tree_prerequisites(repo_root=ROOT):
    manifest_path = repo_root/CLIP_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    return {"clip_ebc_source": {"path_local": str(CLIP_SOURCE.resolve()),
            "commit": CLIP_SOURCE_COMMIT, "source_manifest": CLIP_MANIFEST,
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_files": manifest["files"], "verified": True},
            "clip_ebc_python": {"path_local": str(CLIP_PYTHON), "required": True}}


def _repo_relative(path, repo_root=ROOT):
    return str(Path(path).resolve().relative_to(repo_root.resolve()))


def build_plan(output_dir, sources, models, code_digests, git_revision,
               prerequisites, *, prepared_batches=None, batch_errors=None,
               repo_root=ROOT):
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if not output_dir.is_relative_to((repo_root/"outputs").resolve()):
        raise ValueError("Plan output directory must be inside repository outputs/")
    output_rel = _repo_relative(output_dir, repo_root)
    prepared_batches = prepared_batches or {}
    batch_errors = batch_errors or {}
    batches, jobs = [], []
    for source in sources:
        for batch in source["batches"]:
            batch_rel = Path(output_rel)/"batches"/batch["batch_id"]
            input_rel = str(batch_rel/"input.json")
            materialized = prepared_batches.get(batch["batch_id"])
            batch_entry = {**batch, "directory": str(batch_rel),
                "input_manifest": input_rel,
                "status": "prepared" if materialized else "planned",
                "input_sha256": materialized["input_sha256"] if materialized else None,
                "pts_seconds": materialized["pts_seconds"] if materialized else None}
            if batch["batch_id"] in batch_errors:
                batch_entry["status"] = "failed"
                batch_entry["preparation_error"] = batch_errors[batch["batch_id"]]
            batches.append(batch_entry)
            for model, checkpoint in models.items():
                output_rel_job = str(Path(output_rel)/"results"/model/batch["batch_id"])
                input_hash = materialized["input_sha256"] if materialized else None
                key = idempotency_key(model, input_hash, checkpoint["sha256"], code_digests) if input_hash else None
                arguments = {"model": model, "input": input_rel,
                    "expected_input_sha256": input_hash,
                    "checkpoint": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"], "output_dir": output_rel_job}
                if model == "clip-ebc":
                    arguments["clip_source"] = prerequisites["clip_ebc_source"]["path_local"]
                job = {"job_id": key, "idempotency_key": key,
                    "materialized": bool(materialized),
                    "status": "prepared_not_submitted" if materialized else "planned",
                    "model": model, "batch_id": batch["batch_id"],
                    "input_manifest": input_rel, "input_sha256": input_hash,
                    "checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"],
                    "output_dir": output_rel_job, "arguments": arguments,
                    "code_digests": code_digests, "resources": RESOURCE_PROFILE,
                    "priority": {"name": "brief_validation",
                        "reason": "Inspect response and temporal variation across full available videos; not an attendance estimate."}}
                jobs.append(job)
    completed = sum(batch["status"] == "prepared" for batch in batches)
    failed = any(batch["status"] == "failed" for batch in batches)
    preparation = ("prepared" if completed == len(batches) else
                   ("partial" if completed or failed else "planned"))
    return {"schema_version": 1, "project": "dronecrowd",
        "preparation_state": preparation, "submission_state": "prepared_not_submitted",
        "proposed_executor_profile": "dronecrowd-native-count-v1",
        "executor_profile_registered": False, "git_revision": git_revision,
        "code_digests": code_digests, "local_prerequisites": prerequisites,
        "sources": [{k:v for k,v in source.items() if k not in ("source_path_local", "batches")}
                    for source in sources],
        "batches": batches, "estimated_unique_frames": sum(s["batches"][i]["frame_count"]
            for s in sources for i in range(len(s["batches"]))),
        "estimated_jobs": len(batches)*len(models), "jobs": jobs,
        "not_submitted": True,
        "interpretation": "Sparse one-frame-per-second inspection; not continuous video coverage, ground truth, or attendance.",
        "preflight": None}


def _atomic_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="long-video-plan-", suffix=".json", dir=TMP_ROOT)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ppm_rgb(path, width, height):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.resolve() != path.resolve(strict=True)):
        raise ValueError("RGB must be a regular unlinked PPM file")
    with path.open("rb") as handle:
        header = (handle.readline(), handle.readline(), handle.readline())
        expected = (b"P6\n", f"{width} {height}\n".encode(), b"255\n")
        if header != expected:
            raise ValueError("RGB PPM header/geometry mismatch")
        payload = handle.read(width*height*3+1)
    if len(payload) != width*height*3:
        raise ValueError("RGB PPM payload size mismatch")
    return payload


def _crop_trim_batch(batch_dir, frame):
    from PIL import Image
    width, height = frame["width"], frame["height"]
    if (width, height) != (960, 540):
        raise ValueError("Trim must be decoded natively at 960x540 before cropping")
    ppm = batch_dir/frame["rgb_file"]
    _ppm_rgb(ppm, width, height)
    with Image.open(ppm) as image:
        cropped = image.crop((0, 0, width, height-12))
        cropped.save(ppm, format="PPM")
    preview = frame.get("preview_file")
    if preview:
        preview_path = batch_dir/preview
        with Image.open(preview_path) as image:
            image.crop((0, 0, width, height-12)).save(preview_path, format="PNG")
    frame["width"], frame["height"] = width, height-12
    frame["rgb_sha256"] = sha256_file(ppm)
    frame["cropped_bottom_rows"] = 12
    frame["crop_xyxy"] = [0, 0, width, height-12]


def _validate_batch(batch_dir, batch, source, *, max_seconds_tolerance=0.05):
    manifest_path = batch_dir/"input.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 256*1024:
        raise ValueError(f"Existing batch lacks a complete small input.json: {batch['batch_id']}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_sha256") != source["source_sha256"]:
        raise ValueError("Existing batch belongs to another source video")
    frames = manifest.get("frames")
    times = batch["frame_times_seconds"]
    if not isinstance(frames, list) or len(frames) != len(times):
        raise ValueError("Existing batch frame count differs from plan")
    pts_values = []
    expected_width, expected_height = batch["effective_dimensions"]
    for index, (frame, requested) in enumerate(zip(frames, times)):
        pts = frame.get("pts_seconds")
        requested_seconds = frame.get("requested_seconds")
        if (frame.get("index") != index or type(pts) not in (int, float)
                or not math.isfinite(pts) or type(requested_seconds) not in (int, float)
                or not math.isfinite(requested_seconds)
                or abs(requested_seconds-requested) > max_seconds_tolerance
                or abs(pts-requested) > max_seconds_tolerance
                or (pts_values and pts <= pts_values[-1])
                or frame.get("width") != expected_width or frame.get("height") != expected_height
                or Path(frame.get("rgb_file", "")).name != frame.get("rgb_file")):
            raise ValueError("Existing batch frame timestamp or geometry differs from plan")
        rgb = batch_dir/frame["rgb_file"]
        if sha256_file(rgb) != frame.get("rgb_sha256"):
            raise ValueError("Existing batch RGB hash mismatch")
        _ppm_rgb(rgb, expected_width, expected_height)
        pts_values.append(pts)
    if batch["cropped_bottom_rows"]:
        if manifest.get("post_sampling_crop") != {"cropped_bottom_rows": 12,
                "crop_xyxy": [0, 0, 960, 528], "resized": False}:
            raise ValueError("Existing trim batch lacks the required explicit native crop record")
    return {"input_sha256": sha256_file(manifest_path), "pts_seconds": pts_values}


def _materialize_batch(batch_dir, batch, source, *, sampler=None):
    if sampler is None:
        from sample_video import sample as sampler
    if batch_dir.exists():
        return _validate_batch(batch_dir, batch, source)
    fractions = [t/source["duration_seconds"] for t in batch["frame_times_seconds"]]
    if not all(0 <= value < 1 for value in fractions):
        raise ValueError("Integer sample time falls outside source duration")
    estimate = batch["sampler_output_budget_estimate_bytes"]
    if estimate > MAX_BATCH_BYTES:
        raise ValueError("Batch exceeds sampler output budget")
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="long-video-batch-", dir=TMP_ROOT) as temporary:
        stage = Path(temporary)/"batch"
        report = sampler(source["source_path_local"], stage,
            width=batch["native_dimensions"][0], fractions=fractions,
            max_seconds=120, native=True)
        if report.get("source_sha256") != source["source_sha256"]:
            raise ValueError("Sampler source hash differs from the source manifest")
        manifest_path = stage/"input.json"
        manifest = json.loads(manifest_path.read_text())
        if len(manifest.get("frames", [])) != len(batch["frame_times_seconds"]):
            raise ValueError("Sampler returned an unexpected frame count")
        for frame, requested in zip(manifest["frames"], batch["frame_times_seconds"]):
            if (abs(frame.get("requested_seconds", float("inf"))-requested) > 0.05
                    or abs(frame.get("pts_seconds", float("inf"))-requested) > 0.05):
                raise ValueError("Sampler effective PTS differs from requested integer by over 50 ms")
            if batch["cropped_bottom_rows"]:
                _crop_trim_batch(stage, frame)
        if batch["cropped_bottom_rows"]:
            manifest["sampling_transform"] = "native RGB decode, no resize; crop bottom 12 rows to 960x528"
            manifest["post_sampling_crop"] = {"cropped_bottom_rows": 12,
                "crop_xyxy": [0, 0, 960, 528], "resized": False}
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        validated = _validate_batch(stage, batch, source)
        batch_dir.parent.mkdir(parents=True, exist_ok=True)
        if batch_dir.exists():
            raise FileExistsError("Batch destination appeared during staging; refusing to replace it")
        try:
            os.rename(stage, batch_dir)
        except OSError as error:
            if error.errno == errno.EXDEV:
                raise OSError(errno.EXDEV,
                    "Cannot atomically promote TMP_ROOT staging across filesystems; batch remains uncommitted") from error
            raise
        return validated


def _preparation_map(plan):
    return {batch["batch_id"]: {"input_sha256": batch["input_sha256"],
            "pts_seconds": batch["pts_seconds"]}
        for batch in plan["batches"] if batch["status"] == "prepared"}


def run(output_dir, *, prepare=False, repo_root=ROOT, sampler=None,
        verify_sources=True, disk_free_bytes=None, available_ram_bytes=None):
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to((repo_root/"outputs").resolve()):
        raise ValueError("Output directory must be under repo outputs/")
    prior_plan_path = output_dir/"plan.json"
    if output_dir.exists() and not prior_plan_path.is_file():
        raise FileExistsError("Output directory exists without a plan; refusing to overwrite unknown files")
    sources = _load_source_specs(repo_root, verify_videos=verify_sources)
    models = _load_model_specs(repo_root)
    code_digests = _collect_code_digests(repo_root)
    prerequisites = _source_tree_prerequisites(repo_root)
    prepared = {}
    errors = {}
    if prior_plan_path.is_file():
        previous = json.loads(prior_plan_path.read_text())
        if previous.get("schema_version") != 1 or previous.get("project") != "dronecrowd":
            raise ValueError("Existing output plan has an unknown schema/project")
        prepared = _preparation_map(previous)
        errors = {batch["batch_id"]: batch["preparation_error"]
                  for batch in previous.get("batches", [])
                  if batch.get("status") == "failed" and batch.get("preparation_error")}
        # Never trust a prior plan's prepared marker without rechecking its manifest,
        # source identity, frame geometry, timestamps, and RGB bytes.
        source_by_id = {source["video_id"]: source for source in sources}
        batch_by_id = {batch["batch_id"]: (batch, source_by_id[batch["video_id"]])
                       for source in sources for batch in source["batches"]}
        for batch_id in list(prepared):
            if batch_id not in batch_by_id:
                raise ValueError("Prior plan contains an unknown materialized batch")
            batch, source = batch_by_id[batch_id]
            verified = _validate_batch(output_dir/"batches"/batch_id, batch, source)
            expected_hash = prepared[batch_id]["input_sha256"]
            if verified["input_sha256"] != expected_hash:
                raise ValueError("Previously prepared input.json hash differs from the frozen plan")
            prepared[batch_id] = verified
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(output_dir, sources, models, code_digests,
                      _git_revision(repo_root), prerequisites, prepared_batches=prepared,
                      batch_errors=errors, repo_root=repo_root)
    if prepare:
        preflight_report = preflight(output_dir, free_bytes=disk_free_bytes,
                                     available_ram_bytes=available_ram_bytes)
        plan["preflight"] = preflight_report
        _atomic_json(prior_plan_path, plan)
        for source in sources:
            for batch in source["batches"]:
                if batch["batch_id"] in prepared:
                    continue
                try:
                    result = _materialize_batch(output_dir/"batches"/batch["batch_id"],
                                                batch, source, sampler=sampler)
                    prepared[batch["batch_id"]] = result
                    errors.pop(batch["batch_id"], None)
                except Exception as error:
                    errors[batch["batch_id"]] = f"{type(error).__name__}: {error}"
                    plan = build_plan(output_dir, sources, models, code_digests,
                        _git_revision(repo_root), prerequisites,
                        prepared_batches=prepared, batch_errors=errors, repo_root=repo_root)
                    plan["preflight"] = preflight_report
                    _atomic_json(prior_plan_path, plan)
                    continue
                plan = build_plan(output_dir, sources, models, code_digests,
                    _git_revision(repo_root), prerequisites,
                    prepared_batches=prepared, batch_errors=errors, repo_root=repo_root)
                plan["preflight"] = preflight_report
                _atomic_json(prior_plan_path, plan)
        plan = build_plan(output_dir, sources, models, code_digests,
            _git_revision(repo_root), prerequisites,
            prepared_batches=prepared, batch_errors=errors, repo_root=repo_root)
        plan["preflight"] = preflight_report
    _atomic_json(prior_plan_path, plan)
    return plan


def main():
    os.environ["TMPDIR"] = str(TMP_ROOT)
    os.environ["TMP"] = os.environ["TEMP"] = str(TMP_ROOT)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true",
                        help="Extract and verify native frame batches; never submit jobs")
    args = parser.parse_args()
    result = run(args.output_dir, prepare=args.prepare)
    print(json.dumps({"preparation_state": result["preparation_state"],
        "submission_state": result["submission_state"],
        "estimated_frames": result["estimated_unique_frames"],
        "estimated_jobs": result["estimated_jobs"], "not_submitted": True}))


if __name__ == "__main__":
    main()
