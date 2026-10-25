"""Prepare a pinned registration handoff; never edit policy or submit jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from run_dispatched_inference import (ROOT, FROZEN_PLAN_SHA256, load_plan,
    select_job, verify_code_digests, verify_job_files, validate_environment_lock)

PLAN = "outputs/validacion-extendida-cola-20261025/plan.json"
LOCK = "ops/gpu-candidates.lock.json"
ENTRYPOINT = "scripts/run_dispatched_inference.py"


def build_package(root, plan, revision):
    """Fixed three-job infrastructure trial, not the entire evaluation queue."""
    profiles, requests = {}, []
    for model in ("sr", "clip-ebc", "mpcount"):
        candidates = [j for j in plan["jobs"]
                      if j["batch_id"] == "0265-batch-000" and j["model"] == model]
        if len(candidates) != 1:
            raise ValueError("Initial batch must contain exactly one job per model")
        job = select_job(plan, candidates[0]["job_id"])
        name = "dronecrowd-smoke-" + model + "-v1"
        profiles[name] = {
            "project": "dronecrowd", "revision": revision,
            "entrypoint": ENTRYPOINT, "environment_lock": LOCK,
            "argv": [str(root/".venv/gpu-candidates/bin/python"), str(root/ENTRYPOINT),
                     "--plan", str(root/PLAN), "--plan-sha256", FROZEN_PLAN_SHA256,
                     "--job-id", job["job_id"], "--environment-lock", str(root/LOCK)],
            "ram_gib": 1.5, "vram_mib": 1024,
            "scratch_gib": 0.25, "outputs_gib": 0.25,
            "seconds": 120, "max_runs": 1, "priority": "exploratory",
            "approval": "Autor 2026-10-25: enviar pruebas GPU y preparar alta; SPEC015, revision independiente documentada."
        }
        requests.append({"project": "dronecrowd", "profile": name,
                         "request_id": job["job_id"], "job_id": job["job_id"]})
    return {
        "project-registration.json": {"id": "dronecrowd", "root": "DroneCrowd",
                                      "label": "DroneCrowd", "aliases": ["conteo de multitudes"],
                                      "enabled": True},
        "resource-registration-change.json": {
            "remove_additional_projects": [{"id": "DroneCrowd", "root": "DroneCrowd"}],
            "reason": "Promote existing resource ownership to the main registry without a duplicate memory slice",
            "apply_together_with": "project-registration.json"},
        "gpu-profiles.json": profiles,
        "requests.json": requests,
        "handoff.json": {"schema_version": 1, "revision": revision,
            "plan_sha256": FROZEN_PLAN_SHA256, "state": "prepared_not_applied",
            "submitted": False, "initial_jobs": 3, "remaining_prepared_jobs": 66,
            "max_started_attempts": 3, "seconds_per_attempt": 120,
            "wrapper_memory_max_bytes": 2147483648,
            "wrapper_swap_max_bytes": 536870912}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT/"outputs/alta-dispatcher-20261025")
    args = parser.parse_args()
    root = ROOT.resolve()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if dirty:
        raise ValueError("Commit tracked changes before preparing the pinned package")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for path in (LOCK, ENTRYPOINT, "scripts/prepare_gpu_registration.py"):
        subprocess.run(["git", "cat-file", "-e", revision+":"+path], cwd=root, check=True)
    plan = load_plan(root/PLAN, FROZEN_PLAN_SHA256)
    verify_code_digests(plan)
    validate_environment_lock(root/LOCK)
    for job in plan["jobs"]:
        if job["batch_id"] == "0265-batch-000":
            verify_job_files(select_job(plan, job["job_id"]))
    package = build_package(root, plan, revision)
    output = args.output_dir.absolute()
    if not output.resolve().is_relative_to(root/"outputs") or output.exists():
        raise ValueError("Choose a new output directory inside project outputs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dronecrowd-registration-",
                                     dir="/home/alephbet/home.codex/tmp") as temporary:
        stage = Path(temporary)/"package"
        stage.mkdir()
        for name, value in package.items():
            (stage/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
        hashes = {name: hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in package}
        (stage/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
        os.rename(stage, output)
    print(json.dumps({"state": "prepared_not_applied", "output": str(output),
                      "revision": revision, "profiles": 3, "submitted": False}))


if __name__ == "__main__":
    main()
