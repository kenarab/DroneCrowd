"""Validate and launch one explicitly selected prepared inference job.

This stdlib-only wrapper fails closed before importing model libraries, applies
limits to its existing Nexo scope, and invokes only the typed local runner.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import selectors
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PLAN_SHA256 = "93405aba21aaef66f4a9bc85615d3de3e34ce267c382725b3e7455170adbd73b"
MAX_PLAN_BYTES = 1024 * 1024
MAX_LOCK_BYTES = 128 * 1024
MAX_JOB_SECONDS = 120
RUNNER_TIMEOUT_SECONDS = 115
MAX_LOG_BYTES = 16 * 2**20
MAX_CHILD_FILE_BYTES = 64 * 2**20
SCOPE_LIMITS = {"MemoryHigh": 1610612736, "MemoryMax": 2147483648,
                "MemorySwapMax": 536870912}
SYSFS_LIMIT_FILES = {"MemoryHigh": "memory.high", "MemoryMax": "memory.max",
                     "MemorySwapMax": "memory.swap.max"}
MPCOUNT_UPSTREAM_DIGEST = "deb0550ee49c0fa8851372f2ff6469e150f75fdbde1b16be49b6492b5e20a176"
MPCOUNT_UPSTREAM_KEY = "upstream/mpcount/models/models.py@6eb06772bcf7dfb771c43a14d67146fce767f103"
CLIP_COMMIT = "b52a5944bd75beed34fd3d0da516aa833ae63354"
MODEL_CODE_KEYS = {"sr": "scripts/sr_model.py", "clip-ebc": "scripts/clip_ebc_model.py",
                   "mpcount": "scripts/mpcount_model.py"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SCOPE_RE = re.compile(r"nexo-job-[0-9a-f]{32}\.scope\Z")
MODEL_ARGUMENTS = {
    "sr": {"model", "input", "expected_input_sha256", "checkpoint",
            "checkpoint_sha256", "output_dir"},
    "mpcount": {"model", "input", "expected_input_sha256", "checkpoint",
                "checkpoint_sha256", "output_dir"},
    "clip-ebc": {"model", "input", "expected_input_sha256", "checkpoint",
                 "checkpoint_sha256", "output_dir", "clip_source"},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_output_path(relative, repo_root, *, must_exist):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Expected a repository-relative outputs path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "outputs":
        raise ValueError("Path must remain within repository outputs/")
    root = Path(repo_root).resolve()
    outputs = (root / "outputs").resolve()
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise ValueError("Output references may not be symbolic links")
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(outputs):
        raise ValueError("Resolved path escaped repository outputs/")
    return resolved


def load_plan(path, supplied_sha256, *, repo_root=ROOT):
    if supplied_sha256 != FROZEN_PLAN_SHA256 or not HEX64.fullmatch(supplied_sha256):
        raise ValueError("Plan digest is not the frozen authorized digest")
    plan_path = Path(path).resolve(strict=True)
    if not plan_path.is_relative_to((Path(repo_root).resolve() / "outputs").resolve()):
        raise ValueError("Plan must be stored under repository outputs/")
    if plan_path.stat().st_size > MAX_PLAN_BYTES or sha256_file(plan_path) != supplied_sha256:
        raise ValueError("Plan size or SHA-256 does not match the frozen plan")
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema_version") != 1 or plan.get("project") != "dronecrowd"
            or plan.get("preparation_state") != "prepared"
            or plan.get("submission_state") != "prepared_not_submitted"
            or plan.get("not_submitted") is not True):
        raise ValueError("Plan is not a fully prepared DroneCrowd plan")
    return plan


def select_job(plan, job_id):
    if not HEX64.fullmatch(job_id):
        raise ValueError("Job ID must be a 64-character lowercase digest")
    matches = [job for job in plan.get("jobs", []) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError("Job ID must identify exactly one plan entry")
    job = matches[0]
    model = job.get("model")
    if model not in MODEL_ARGUMENTS or job.get("materialized") is not True:
        raise ValueError("Job is not a materialized supported model job")
    if job.get("status") != "prepared_not_submitted" or job.get("idempotency_key") != job_id:
        raise ValueError("Job state or idempotency key differs from the frozen plan")
    if job.get("resources") != {"ram_limit_bytes": 2147483648,
            "swap_limit_bytes": 536870912, "expected_ram_bytes": 1610612736,
            "expected_vram_bytes": 1073741824, "max_seconds": 120,
            "cpu_threads": 1, "disk_increment_bytes": 268435456}:
        raise ValueError("Job resource profile differs from the fixed Nexo contract")
    arguments = job.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != MODEL_ARGUMENTS[model]:
        raise ValueError("Job arguments are outside the model-specific whitelist")
    expected_arguments = {"model": job.get("model"), "input": job.get("input_manifest"),
        "expected_input_sha256": job.get("input_sha256"), "checkpoint": job.get("checkpoint"),
        "checkpoint_sha256": job.get("checkpoint_sha256"), "output_dir": job.get("output_dir")}
    if model == "clip-ebc":
        expected_arguments["clip_source"] = plan.get("local_prerequisites", {}).get(
            "clip_ebc_source", {}).get("path_local")
    for key, expected in expected_arguments.items():
        if arguments.get(key) != expected:
            raise ValueError(f"Typed argument {key} differs from the plan entry")
    if (not HEX64.fullmatch(str(job.get("input_sha256", "")))
            or not HEX64.fullmatch(str(job.get("checkpoint_sha256", "")))):
        raise ValueError("Input/checkpoint hashes are missing or inconsistent")
    checkpoint_path = PurePosixPath(str(job.get("checkpoint", "")))
    if checkpoint_path.parts[:2] != ("outputs", "modelos"):
        raise ValueError("Checkpoint must be stored under repository outputs/modelos/")
    return job


def verify_clip_sources(plan, *, repo_root=ROOT):
    prerequisites = plan.get("local_prerequisites", {})
    clip = prerequisites.get("clip_ebc_source", {})
    if clip.get("commit") != CLIP_COMMIT or not clip.get("verified"):
        raise ValueError("Pinned local CLIP-EBC source prerequisite is missing")
    source_root = Path(clip.get("path_local", "")).resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("Pinned CLIP-EBC source path is not a directory")
    source_manifest_rel = clip.get("source_manifest")
    manifest_pure = PurePosixPath(str(source_manifest_rel))
    if manifest_pure.is_absolute() or ".." in manifest_pure.parts:
        raise ValueError("Invalid CLIP-EBC source manifest path")
    manifest_path = Path(repo_root).joinpath(*manifest_pure.parts)
    if not manifest_path.resolve(strict=True).is_relative_to(Path(repo_root).resolve()):
        raise ValueError("CLIP-EBC source manifest escaped repository")
    if (not source_manifest_rel or sha256_file(manifest_path) != clip.get("source_manifest_sha256")):
        raise ValueError("CLIP-EBC source manifest digest mismatch")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("commit") != CLIP_COMMIT:
        raise ValueError("CLIP-EBC local source manifest commit mismatch")
    expected_files = source_manifest.get("files", {})
    if clip.get("source_files") != expected_files:
        raise ValueError("CLIP-EBC prerequisite file list differs from its manifest")
    digests = plan.get("code_digests", {})
    for name, expected in expected_files.items():
        pure_name = PurePosixPath(name)
        if pure_name.is_absolute() or ".." in pure_name.parts:
            raise ValueError("Invalid pinned CLIP-EBC source path")
        key = f"clip-source/{name}"
        source_file = source_root.joinpath(*pure_name.parts)
        if (not source_file.resolve(strict=True).is_relative_to(source_root)
                or digests.get(key) != expected or sha256_file(source_file) != expected):
            raise ValueError(f"Pinned CLIP-EBC source file mismatch: {name}")
    return source_root


def verify_code_digests(plan, *, repo_root=ROOT):
    digests = plan.get("code_digests")
    if not isinstance(digests, dict) or not digests:
        raise ValueError("Plan lacks code digests")
    clip_root = verify_clip_sources(plan, repo_root=repo_root)
    for relative, expected in digests.items():
        if not HEX64.fullmatch(str(expected)):
            raise ValueError(f"Invalid code digest: {relative}")
        if relative == MPCOUNT_UPSTREAM_KEY:
            if expected != MPCOUNT_UPSTREAM_DIGEST:
                raise ValueError("Pinned upstream MPCount source digest mismatch")
            continue
        if relative.startswith("clip-source/"):
            suffix = PurePosixPath(relative.removeprefix("clip-source/"))
            if suffix.is_absolute() or ".." in suffix.parts:
                raise ValueError("Invalid CLIP source path in code digest list")
            path = clip_root.joinpath(*suffix.parts)
        else:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("Invalid repository code path in digest list")
            path = Path(repo_root).joinpath(*pure.parts)
            if not path.resolve(strict=True).is_relative_to(Path(repo_root).resolve()):
                raise ValueError("Code digest path escaped repository")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Code digest mismatch: {relative}")
    if MPCOUNT_UPSTREAM_KEY not in digests:
        raise ValueError("Plan omits the pinned upstream MPCount provenance digest")
    return clip_root


def verify_job_files(job, *, repo_root=ROOT):
    input_path = _repo_output_path(job["input_manifest"], repo_root, must_exist=True)
    checkpoint_path = _repo_output_path(job["checkpoint"], repo_root, must_exist=True)
    output_path = _repo_output_path(job["output_dir"], repo_root, must_exist=False)
    if input_path.stat().st_size > 256 * 1024:
        raise ValueError("Input manifest exceeds the fixed size limit")
    if sha256_file(input_path) != job["input_sha256"]:
        raise ValueError("Input manifest SHA-256 mismatch")
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    return input_path, checkpoint_path, output_path


def validate_environment_lock(path, *, expected_python=None, version_lookup=None):
    lock_path = Path(path).resolve(strict=True)
    if lock_path.stat().st_size > MAX_LOCK_BYTES:
        raise ValueError("Environment lock exceeds the size limit")
    lock = json.loads(lock_path.read_text())
    if set(lock) != {"python", "packages"} or not isinstance(lock["packages"], dict):
        raise ValueError("Environment lock must have exactly python and packages")
    python_version = expected_python or platform.python_version()
    if lock["python"] != python_version:
        raise ValueError("Python version differs from the environment lock")
    required_packages = {"torch", "numpy", "einops", "ftfy", "wcwidth", "regex"}
    if not required_packages.issubset(lock["packages"]):
        raise ValueError("Environment lock omits a required model/runtime package")
    if not lock["packages"] or any(not isinstance(name, str) or not isinstance(version, str)
                                    for name, version in lock["packages"].items()):
        raise ValueError("Environment lock package entries must be string versions")
    version_lookup = version_lookup or importlib.metadata.version
    for package, expected in lock["packages"].items():
        try:
            actual = version_lookup(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Locked package is unavailable: {package}") from error
        if actual != expected:
            raise ValueError(f"Installed {package} version differs from the environment lock")
    return lock


def validate_interpreter(executable, *, repo_root=ROOT):
    expected = Path(repo_root) / ".venv/gpu-candidates/bin/python"
    actual = Path(os.path.abspath(executable))
    if actual != Path(os.path.abspath(expected)):
        raise ValueError("Wrapper must run from the pinned .venv/gpu-candidates/bin/python")


def cgroup_scope(cgroup_text, *, fs_root=Path("/sys/fs/cgroup")):
    entries = [line.split(":", 2)[2] for line in cgroup_text.splitlines()
               if line.startswith("0::")]
    if len(entries) != 1:
        raise RuntimeError("A unified cgroup-v2 scope is required")
    entry = entries[0]
    unit = entry.rstrip("/").split("/")[-1]
    if not SCOPE_RE.fullmatch(unit) or ".." in PurePosixPath(entry).parts:
        raise RuntimeError("Current process is not inside an authorized nexo-job-<uuid>.scope")
    root = Path(fs_root).resolve()
    directory = (root / entry.lstrip("/")).resolve(strict=True)
    if not directory.is_relative_to(root):
        raise RuntimeError("Current cgroup path escaped the cgroup-v2 mount")
    return unit, directory


def lower_current_scope(unit, directory, *, runner=subprocess.run):
    if not SCOPE_RE.fullmatch(unit):
        raise ValueError("Refusing to modify a cgroup other than the current authorized job scope")
    command = ["systemctl", "--user", "set-property", "--runtime", unit,
               f"MemoryHigh={SCOPE_LIMITS['MemoryHigh']}",
               f"MemoryMax={SCOPE_LIMITS['MemoryMax']}",
               f"MemorySwapMax={SCOPE_LIMITS['MemorySwapMax']}"]
    result = runner(command, check=False, capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise RuntimeError("Could not lower memory limits on current Nexo scope")
    for property_name, filename in SYSFS_LIMIT_FILES.items():
        value = (Path(directory) / filename).read_text().strip()
        if value != str(SCOPE_LIMITS[property_name]):
            raise RuntimeError(f"Cgroup limit verification failed for {property_name}")


def set_runtime_environment(environ=None, *, temp_root=Path("/home/alephbet/home.codex/tmp")):
    env = os.environ if environ is None else environ
    temp_root = Path(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    values = {"TMPDIR": str(temp_root), "TMP": str(temp_root), "TEMP": str(temp_root),
        "npm_config_cache": str(temp_root/"npm-cache"),
        "PIP_CACHE_DIR": str(temp_root/"pip-cache"),
        "UV_CACHE_DIR": str(temp_root/"uv-cache"),
        "TORCH_HOME": str(temp_root/"dronecrowd-torch-cache"),
        "CUDA_CACHE_PATH": str(temp_root/"dronecrowd-cuda-cache"),
        "XDG_CACHE_HOME": str(temp_root/"dronecrowd-xdg-cache"),
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.update(values)
    return env


def command_for_job(job, paths, *, executable, clip_source=None):
    input_path, checkpoint_path, output_path = paths
    command = [str(executable), str(ROOT/"scripts/evaluate_crowd_candidates.py"),
        "--model", job["model"], "--input", str(input_path),
        "--expected-input-sha256", job["input_sha256"], "--checkpoint", str(checkpoint_path),
        "--checkpoint-sha256", job["checkpoint_sha256"], "--output-dir", str(output_path)]
    if job["model"] == "clip-ebc":
        command.extend(["--clip-source", str(clip_source)])
    return command


def completed_output(output_path, job, input_manifest_path, plan):
    report_path = Path(output_path)/"report.json"
    if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size > 1024 * 1024:
        return False
    try:
        report = json.loads(report_path.read_text())
        manifest = json.loads(Path(input_manifest_path).read_text())
        frames = manifest.get("frames")
        results = report.get("results")
        if (report.get("status") != "complete" or report.get("model") != job["model"]
                or report.get("input_sha256") != job["input_sha256"]
                or report.get("checkpoint_sha256") != job["checkpoint_sha256"]
                or report.get("script_sha256") != plan.get("code_digests", {}).get(
                    "scripts/evaluate_crowd_candidates.py")
                or report.get("model_code_sha256") != plan.get("code_digests", {}).get(
                    MODEL_CODE_KEYS[job["model"]])
                or not isinstance(frames, list) or not isinstance(results, list)
                or len(results) != len(frames) or not results):
            return False
        for index, (result, frame) in enumerate(zip(results, frames)):
            result_pts, frame_pts = result.get("pts_seconds"), frame.get("pts_seconds")
            if (type(result.get("index")) is not int or type(frame.get("index")) is not int
                    or result.get("index") != index or frame.get("index") != index
                    or type(result_pts) not in (int, float) or not math.isfinite(result_pts)
                    or type(frame_pts) not in (int, float) or not math.isfinite(frame_pts)
                    or result_pts != frame_pts
                    or result.get("rgb_sha256") != frame.get("rgb_sha256")
                    or result.get("width") != frame.get("width")
                    or result.get("height") != frame.get("height")):
                return False
            for key in ("count", "negative_mass"):
                value = result.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    return False
            name = result.get("density_file")
            if not isinstance(name, str) or Path(name).name != name:
                return False
            density = Path(output_path)/name
            if (density.is_symlink() or not density.is_file()
                    or not HEX64.fullmatch(str(result.get("density_sha256", "")))
                    or sha256_file(density) != result["density_sha256"]):
                return False
        return True
    except (OSError, ValueError, TypeError, OverflowError, AttributeError, json.JSONDecodeError):
        return False


def _set_child_file_limit():
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (MAX_CHILD_FILE_BYTES, MAX_CHILD_FILE_BYTES))


def run_bounded_process(command, *, cwd, env, timeout, stdout_path, stderr_path,
                        popen_factory=subprocess.Popen, monotonic=time.monotonic):
    """Run one child with bounded regular-file writes and bounded captured logs."""
    started = monotonic()
    deadline = started + timeout
    truncation = {"stdout": False, "stderr": False}
    counts = {"stdout": 0, "stderr": 0}
    with Path(stdout_path).open("xb") as stdout_file, Path(stderr_path).open("xb") as stderr_file:
        process = popen_factory(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, close_fds=True,
            preexec_fn=_set_child_file_limit)
        selector = selectors.DefaultSelector()
        streams = ((process.stdout, "stdout", stdout_file),
                   (process.stderr, "stderr", stderr_file))
        try:
            for stream, name, destination in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, (name, destination))
            while selector.get_map():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(command, timeout)
                events = selector.select(min(0.25, remaining))
                for key, _ in events:
                    stream = key.fileobj
                    name, destination = key.data
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    room = max(0, MAX_LOG_BYTES - counts[name])
                    if room:
                        kept = chunk[:room]
                        destination.write(kept)
                        counts[name] += len(kept)
                    if len(chunk) > room:
                        truncation[name] = True
            return_code = process.wait(timeout=max(0, deadline-monotonic()))
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            selector.close()
            for stream, _, _ in streams:
                if not stream.closed:
                    stream.close()
    return subprocess.CompletedProcess(command, return_code), truncation


def _append_status(path, status, **fields):
    record = {"status": status, "time_unix": time.time(), **fields}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_preflight_failure(error, *, plan_sha256=None, job_id=None, repo_root=ROOT):
    """Persist a concise, non-overwriting failure record at one fixed safe path."""
    failure_dir = Path(repo_root)/"outputs/dispatcher-validation-failures"
    if failure_dir.is_symlink() or (failure_dir.exists() and not failure_dir.is_dir()):
        raise OSError("Fixed dispatcher failure-record directory is unsafe")
    failure_dir.mkdir(parents=True, exist_ok=True)
    root_outputs = (Path(repo_root)/"outputs").resolve()
    if not failure_dir.resolve(strict=True).is_relative_to(root_outputs):
        raise OSError("Fixed dispatcher failure-record path escaped outputs/")
    record = {"error_type": type(error).__name__, "message": str(error)[:1024]}
    if isinstance(plan_sha256, str) and HEX64.fullmatch(plan_sha256):
        record["plan_sha256"] = plan_sha256
    if isinstance(job_id, str) and HEX64.fullmatch(job_id):
        record["job_id"] = job_id
    path = failure_dir/(uuid.uuid4().hex+".json")
    descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def run_job(*, plan_path, plan_sha256, job_id, environment_lock, repo_root=ROOT,
            executable=None, started_at=None, monotonic=time.monotonic,
            subprocess_run=subprocess.run, cgroup_reader=None, cgroup_fs=Path("/sys/fs/cgroup"),
            environment=None, package_version_lookup=None,
            process_runner=run_bounded_process, plan_loader=load_plan,
            job_selector=select_job, command_builder=command_for_job,
            completion_checker=completed_output):
    repo_root = Path(repo_root).resolve()
    started_at = monotonic() if started_at is None else started_at
    plan = plan_loader(plan_path, plan_sha256, repo_root=repo_root)
    job = job_selector(plan, job_id)
    clip_root = verify_code_digests(plan, repo_root=repo_root)
    paths = verify_job_files(job, repo_root=repo_root)
    input_path, checkpoint_path, output_path = paths
    if output_path.exists():
        if completion_checker(output_path, job, input_path, plan):
            return {"status": "complete", "already_complete": True, "job_id": job_id}
        raise FileExistsError("Output exists but is not a verified complete result; require a new attempt/output path")
    lock_path = Path(environment_lock).resolve(strict=True)
    expected_lock = (repo_root/"ops/gpu-candidates.lock.json").resolve(strict=True)
    if lock_path != expected_lock:
        raise ValueError("Environment lock must be the tracked ops/gpu-candidates.lock.json")
    validate_environment_lock(lock_path, version_lookup=package_version_lookup)
    executable = executable or sys.executable
    validate_interpreter(executable, repo_root=repo_root)
    if monotonic()-started_at >= MAX_JOB_SECONDS:
        raise TimeoutError("The 120-second job budget is already exhausted")
    if cgroup_reader is None:
        cgroup_reader = Path("/proc/self/cgroup").read_text()
    unit, cgroup_dir = cgroup_scope(cgroup_reader, fs_root=cgroup_fs)
    lower_current_scope(unit, cgroup_dir, runner=subprocess_run)
    env = os.environ.copy() if environment is None else environment
    set_runtime_environment(env)

    # Create only an output parent and an immutable, unique sidecar for wrapper logs.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_root = output_path.parent/(output_path.name+".wrapper-attempts")
    if attempts_root.is_symlink() or (attempts_root.exists() and not attempts_root.is_dir()):
        raise ValueError("Wrapper attempt log path is not a private directory")
    attempts_root.mkdir(exist_ok=True)
    attempt_dir = attempts_root/uuid.uuid4().hex
    attempt_dir.mkdir(exist_ok=False)
    status_path = attempt_dir/"status.jsonl"
    stdout_path, stderr_path = attempt_dir/"stdout.log", attempt_dir/"stderr.log"
    _append_status(status_path, "starting", job_id=job_id, model=job["model"])
    cmd = command_builder(job, paths, executable=executable, clip_source=clip_root)
    remaining = MAX_JOB_SECONDS-(monotonic()-started_at)
    if remaining <= 0:
        _append_status(status_path, "failed", reason="budget exhausted before runner")
        raise TimeoutError("The 120-second job budget is exhausted before inference")
    timeout = min(RUNNER_TIMEOUT_SECONDS, remaining)
    failure_recorded = False
    try:
        result, truncated = process_runner(cmd, cwd=repo_root, env=env,
            timeout=timeout, stdout_path=stdout_path, stderr_path=stderr_path)
        if any(truncated.values()):
            _append_status(status_path, "logs_truncated", **truncated)
        if result.returncode != 0:
            _append_status(status_path, "failed", returncode=result.returncode)
            failure_recorded = True
            raise subprocess.CalledProcessError(result.returncode, cmd)
        if not completion_checker(output_path, job, input_path, plan):
            _append_status(status_path, "failed", reason="runner output did not verify complete")
            failure_recorded = True
            raise RuntimeError("Runner returned success without a verifiable complete report")
        _append_status(status_path, "complete", returncode=0)
        return {"status": "complete", "already_complete": False, "job_id": job_id,
                "output_dir": str(output_path), "attempt_dir": str(attempt_dir)}
    except subprocess.TimeoutExpired:
        if not failure_recorded:
            _append_status(status_path, "failed", reason="runner timeout", timeout_seconds=timeout)
        raise
    except Exception as error:
        # A nonzero exit was already recorded above; otherwise keep a failure event
        # alongside stdout/stderr without ever rewriting the attempt history.
        if not failure_recorded:
            _append_status(status_path, "failed", error=f"{type(error).__name__}: {error}")
        raise


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"Invalid wrapper arguments: {message}")


def _option_value(tokens, option):
    try:
        index = tokens.index(option)
        return tokens[index+1]
    except (ValueError, IndexError):
        return None


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = None
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    try:
        args = parser.parse_args(tokens)
        result = run_job(plan_path=args.plan, plan_sha256=args.plan_sha256,
            job_id=args.job_id, environment_lock=args.environment_lock)
    except SystemExit as error:
        if error.code == 0:
            return 0
        exception = ValueError("Invalid wrapper arguments")
        try:
            evidence = write_preflight_failure(exception,
                plan_sha256=_option_value(tokens, "--plan-sha256"),
                job_id=_option_value(tokens, "--job-id"), repo_root=ROOT)
        except Exception:
            evidence = None
        print(json.dumps({"status": "failed", "error_type": type(exception).__name__,
            "evidence_file": str(evidence) if evidence else None}))
        return 1
    except Exception as error:
        try:
            evidence = write_preflight_failure(error,
                plan_sha256=(getattr(args, "plan_sha256", None) if args else
                             _option_value(tokens, "--plan-sha256")),
                job_id=(getattr(args, "job_id", None) if args else
                        _option_value(tokens, "--job-id")), repo_root=ROOT)
        except Exception:
            evidence = None
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
            "evidence_file": str(evidence) if evidence else None}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
