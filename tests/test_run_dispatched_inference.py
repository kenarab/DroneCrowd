import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import io
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TMP = Path("/home/alephbet/home.codex/tmp")
SPEC = importlib.util.spec_from_file_location(
    "run_dispatched_inference", ROOT/"scripts/run_dispatched_inference.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DispatchedInferenceTests(unittest.TestCase):
    def fixture(self, temp):
        base = Path(temp)
        repo = base/"repo"
        output_root = repo/"outputs/dispatch-test"
        (repo/"scripts").mkdir(parents=True)
        (repo/"ops").mkdir()
        output_root.mkdir(parents=True)
        runner = repo/"scripts/evaluate_crowd_candidates.py"
        runner.write_text("# fixed runner fixture\n")
        model_code = {}
        for relative in ("scripts/sr_model.py", "scripts/clip_ebc_model.py", "scripts/mpcount_model.py"):
            path = repo/relative
            path.write_text("# model code fixture\n")
            model_code[relative] = digest(path)
        clip_root = base/"local-clip"
        clip_file = clip_root/"models/clip.py"
        clip_file.parent.mkdir(parents=True)
        clip_file.write_text("# pinned clip source\n")
        clip_hash = digest(clip_file)
        clip_manifest = {"commit": wrapper.CLIP_COMMIT,
                         "files": {"models/clip.py": clip_hash}}
        clip_manifest_path = repo/"ops/clip-ebc-sources.json"
        clip_manifest_path.write_text(json.dumps(clip_manifest))
        manifest_hash = digest(clip_manifest_path)
        checkpoint = repo/"outputs/modelos/SR/dronecrowd.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint fixture")
        input_manifest = output_root/"input.json"
        input_manifest.write_text(json.dumps({"frames": [{"index": 0, "pts_seconds": 1.25,
            "rgb_sha256": "c"*64, "width": 32, "height": 32}]}))
        input_hash, checkpoint_hash = digest(input_manifest), digest(checkpoint)
        code = {"scripts/evaluate_crowd_candidates.py": digest(runner),
                "ops/clip-ebc-sources.json": manifest_hash,
                "clip-source/models/clip.py": clip_hash,
                wrapper.MPCOUNT_UPSTREAM_KEY: wrapper.MPCOUNT_UPSTREAM_DIGEST}
        code.update(model_code)
        job_id = "a"*64
        job = {"job_id": job_id, "idempotency_key": job_id,
            "materialized": True, "status": "prepared_not_submitted", "model": "sr",
            "batch_id": "fake-batch", "input_manifest": "outputs/dispatch-test/input.json",
            "input_sha256": input_hash, "checkpoint": "outputs/modelos/SR/dronecrowd.tar",
            "checkpoint_sha256": checkpoint_hash, "output_dir": "outputs/dispatch-test/result",
            "resources": {"ram_limit_bytes": 2147483648, "swap_limit_bytes": 536870912,
                "expected_ram_bytes": 1610612736, "expected_vram_bytes": 1073741824,
                "max_seconds": 120, "cpu_threads": 1, "disk_increment_bytes": 268435456}}
        job["arguments"] = {"model": "sr", "input": job["input_manifest"],
            "expected_input_sha256": input_hash, "checkpoint": job["checkpoint"],
            "checkpoint_sha256": checkpoint_hash, "output_dir": job["output_dir"]}
        plan = {"schema_version": 1, "project": "dronecrowd",
            "preparation_state": "prepared", "submission_state": "prepared_not_submitted",
            "not_submitted": True, "local_prerequisites": {"clip_ebc_source": {
                "path_local": str(clip_root), "commit": wrapper.CLIP_COMMIT,
                "source_manifest": "ops/clip-ebc-sources.json",
                "source_manifest_sha256": manifest_hash,
                "source_files": clip_manifest["files"], "verified": True}},
            "code_digests": code, "jobs": [job]}
        plan_path = output_root/"plan.json"
        plan_path.write_text(json.dumps(plan))
        plan_hash = digest(plan_path)
        lock_path = repo/"ops/gpu-candidates.lock.json"
        lock_path.write_text(json.dumps({"python": "3.13.13",
            "packages": {"torch": "2.14.1", "numpy": "2.5.4", "einops": "0.8.1",
                         "ftfy": "6.3.1", "wcwidth": "0.2.14", "regex": "2026.9.29"}}))
        return {"repo": repo, "output_root": output_root, "clip_root": clip_root,
                "input": input_manifest, "checkpoint": checkpoint,
                "plan_path": plan_path, "plan_hash": plan_hash, "job": job,
                "job_id": job_id, "lock": lock_path, "plan": plan}

    def test_wrapper_has_no_ml_imports(self):
        tree = ast.parse((ROOT/"scripts/run_dispatched_inference.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("numpy", imported)

    def test_frozen_plan_job_whitelist_and_file_hashes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            fixture = self.fixture(temp)
            with patch.object(wrapper, "FROZEN_PLAN_SHA256", fixture["plan_hash"]):
                plan = wrapper.load_plan(fixture["plan_path"], fixture["plan_hash"],
                                         repo_root=fixture["repo"])
                job = wrapper.select_job(plan, fixture["job_id"])
                wrapper.verify_code_digests(plan, repo_root=fixture["repo"])
                paths = wrapper.verify_job_files(job, repo_root=fixture["repo"])
                self.assertEqual(paths[0], fixture["input"].resolve())
                self.assertEqual(paths[1], fixture["checkpoint"].resolve())
                job["arguments"]["shell"] = "arbitrary"
                with self.assertRaisesRegex(ValueError, "whitelist"):
                    wrapper.select_job(plan, fixture["job_id"])

    def test_select_job_uses_frozen_plan_schema(self):
        plan_path = ROOT/"outputs/validacion-extendida-cola-20261025/plan.json"
        plan = wrapper.load_plan(plan_path, wrapper.FROZEN_PLAN_SHA256)
        self.assertEqual(len(plan["jobs"]), 69)
        first_batch = [job for job in plan["jobs"] if job["batch_id"] == "0265-batch-000"]
        self.assertEqual([job["model"] for job in first_batch], ["sr", "clip-ebc", "mpcount"])
        self.assertTrue(all(wrapper.select_job(plan, job["job_id"]) is job for job in first_batch))

    def test_environment_lock_and_lexical_interpreter(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            fixture = self.fixture(temp)
            wrapper.validate_environment_lock(fixture["lock"], expected_python="3.13.13",
                version_lookup=lambda name: {"torch": "2.14.1", "numpy": "2.5.4",
                    "einops": "0.8.1", "ftfy": "6.3.1", "wcwidth": "0.2.14",
                    "regex": "2026.9.29"}[name])
            with self.assertRaisesRegex(ValueError, "differs from the environment lock"):
                wrapper.validate_environment_lock(fixture["lock"], expected_python="3.12.0",
                    version_lookup=lambda name: "wrong")
            expected = fixture["repo"]/(".venv/gpu-candidates/bin/python")
            wrapper.validate_interpreter(str(expected), repo_root=fixture["repo"])
            with self.assertRaisesRegex(ValueError, "must run from"):
                wrapper.validate_interpreter("/usr/bin/python3", repo_root=fixture["repo"])

    def test_scope_is_validated_and_only_own_scope_is_lowered(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            fs = Path(temp)/"cgroup"
            unit = "nexo-job-"+"b"*32+".scope"
            directory = fs/"user.slice"/unit
            directory.mkdir(parents=True)
            entry = "/user.slice/"+unit
            got_unit, got_dir = wrapper.cgroup_scope("0::"+entry+"\n", fs_root=fs)
            self.assertEqual((got_unit, got_dir), (unit, directory.resolve()))
            for filename in wrapper.SYSFS_LIMIT_FILES.values():
                (directory/filename).write_text("1")
            commands = []
            def fake_systemctl(argv, **kwargs):
                commands.append(argv)
                for property_name, filename in wrapper.SYSFS_LIMIT_FILES.items():
                    (directory/filename).write_text(str(wrapper.SCOPE_LIMITS[property_name]))
                return types.SimpleNamespace(returncode=0)
            wrapper.lower_current_scope(unit, directory, runner=fake_systemctl)
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][:5], ["systemctl", "--user", "set-property", "--runtime", unit])
            self.assertIn("MemoryMax=2147483648", commands[0])
            with self.assertRaisesRegex(RuntimeError, "authorized"):
                wrapper.cgroup_scope("0::/user.slice/user-1.service\n", fs_root=fs)

    def test_runner_command_has_fixed_argv_and_clip_only_arg(self):
        job = {"model": "sr", "input_sha256": "a"*64,
               "checkpoint_sha256": "b"*64}
        command = wrapper.command_for_job(job,
            (Path("/repo/input.json"), Path("/repo/model.pth"), Path("/repo/result")),
            executable="/repo/.venv/gpu-candidates/bin/python")
        self.assertIn("--expected-input-sha256", command)
        self.assertNotIn("--clip-source", command)
        self.assertTrue(all(isinstance(token, str) for token in command))
        job["model"] = "clip-ebc"
        command = wrapper.command_for_job(job,
            (Path("/repo/input.json"), Path("/repo/model.pth"), Path("/repo/result")),
            executable="/repo/.venv/gpu-candidates/bin/python", clip_source="/local/clip")
        self.assertEqual(command[-2:], ["--clip-source", "/local/clip"])

    def test_completed_result_requires_hashes_and_report_identity(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            output = Path(temp)/"result"
            output.mkdir()
            density = output/"density-000.npy"
            density.write_bytes(b"density")
            job = {"model": "sr", "input_sha256": "a"*64,
                   "checkpoint_sha256": "b"*64}
            plan = {"code_digests": {"scripts/evaluate_crowd_candidates.py": "e"*64,
                                    "scripts/sr_model.py": "f"*64}}
            input_manifest = Path(temp)/"input.json"
            frame = {"index": 0, "pts_seconds": 1.25, "rgb_sha256": "c"*64,
                     "width": 32, "height": 32}
            frame2 = {"index": 1, "pts_seconds": 2.5, "rgb_sha256": "d"*64,
                      "width": 32, "height": 32}
            density2 = output/"density-001.npy"
            density2.write_bytes(b"density 2")
            input_manifest.write_text(json.dumps({"frames": [frame, frame2]}))
            report = {"status": "complete", "model": "sr", "input_sha256": "a"*64,
                "checkpoint_sha256": "b"*64, "results": [{"index": 0, "pts_seconds": 1.25,
                    "rgb_sha256": "c"*64, "width": 32, "height": 32,
                    "count": 2.5, "negative_mass": -0.1,
                    "density_file": density.name, "density_sha256": digest(density)},
                    {"index": 1, "pts_seconds": 2.5, "rgb_sha256": "d"*64,
                     "width": 32, "height": 32, "count": 3, "negative_mass": 0,
                     "density_file": density2.name,
                     "density_sha256": digest(density2)}]}
            report["script_sha256"] = "e"*64
            report["model_code_sha256"] = "f"*64
            (output/"report.json").write_text(json.dumps(report))
            self.assertTrue(wrapper.completed_output(output, job, input_manifest, plan))
            report["checkpoint_sha256"] = "c"*64
            (output/"report.json").write_text(json.dumps(report))
            self.assertFalse(wrapper.completed_output(output, job, input_manifest, plan))
            report["checkpoint_sha256"] = "b"*64
            report["results"] = report["results"][:1]
            (output/"report.json").write_text(json.dumps(report))
            self.assertFalse(wrapper.completed_output(output, job, input_manifest, plan))
            report["results"] = report["results"]+[{
                "index": 1, "pts_seconds": 2.5, "rgb_sha256": "d"*64,
                "width": 32, "height": 32, "count": float("nan"), "negative_mass": 0,
                "density_file": density2.name, "density_sha256": digest(density2)}]
            (output/"report.json").write_text(json.dumps(report))
            self.assertFalse(wrapper.completed_output(output, job, input_manifest, plan))

    def test_end_to_end_mock_launch_then_verified_complete_skips_scope(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            fixture = self.fixture(temp)
            repo = fixture["repo"]
            executable = repo.resolve()/".venv/gpu-candidates/bin/python"
            values = {"torch": "2.14.1", "numpy": "2.5.4", "einops": "0.8.1",
                      "ftfy": "6.3.1", "wcwidth": "0.2.14", "regex": "2026.9.29"}
            calls = []
            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                Path(kwargs["stdout_path"]).write_bytes(b"")
                Path(kwargs["stderr_path"]).write_bytes(b"")
                if command[1].endswith("evaluate_crowd_candidates.py"):
                    output_path = Path(command[command.index("--output-dir")+1])
                    output_path.mkdir()
                    density = output_path/"density-000.npy"
                    density.write_bytes(b"mock density")
                    report = {"status": "complete", "model": "sr",
                        "input_sha256": fixture["job"]["input_sha256"],
                        "checkpoint_sha256": fixture["job"]["checkpoint_sha256"],
                        "results": [{"index": 0, "pts_seconds": 1.25,
                                     "rgb_sha256": "c"*64, "width": 32, "height": 32,
                                     "count": 1.0, "negative_mass": 0.0,
                                     "density_file": density.name,
                                     "density_sha256": digest(density)}],
                        "script_sha256": fixture["plan"]["code_digests"][
                            "scripts/evaluate_crowd_candidates.py"],
                        "model_code_sha256": fixture["plan"]["code_digests"][
                            "scripts/sr_model.py"]}
                    (output_path/"report.json").write_text(json.dumps(report))
                return types.SimpleNamespace(returncode=0), {"stdout": False, "stderr": False}
            common = {"plan_path": fixture["plan_path"], "plan_sha256": fixture["plan_hash"],
                "job_id": fixture["job_id"], "environment_lock": fixture["lock"],
                "repo_root": repo, "executable": str(executable), "started_at": 0.0,
                "monotonic": lambda: 1.0, "subprocess_run": fake_run,
                "cgroup_reader": "0::/user.slice/nexo-job-"+"c"*32+".scope\n",
                "cgroup_fs": Path(temp)/"cgroup", "environment": {},
                "process_runner": fake_run}
            cgroup_dir = common["cgroup_fs"]/"user.slice"/("nexo-job-"+"c"*32+".scope")
            cgroup_dir.mkdir(parents=True)
            with patch.object(wrapper, "FROZEN_PLAN_SHA256", fixture["plan_hash"]), \
                 patch.object(wrapper, "lower_current_scope") as lower, \
                 patch.object(wrapper, "set_runtime_environment") as set_env:
                first = wrapper.run_job(**common,
                    package_version_lookup=lambda name: values[name])
                self.assertEqual(first["status"], "complete")
                self.assertEqual(len(calls), 1)
                command, kwargs = calls[0]
                self.assertIsInstance(command, list)
                self.assertEqual(command[0], str(executable))
                attempt = Path(first["attempt_dir"])
                self.assertTrue((attempt/"stdout.log").is_file())
                self.assertTrue((attempt/"stderr.log").is_file())
                self.assertIn('"status": "complete"', (attempt/"status.jsonl").read_text())
                second = wrapper.run_job(**common,
                    package_version_lookup=lambda name: values[name])
                self.assertTrue(second["already_complete"])
                lower.assert_called_once()
                set_env.assert_called_once()

    def test_child_file_and_log_limits(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            root = Path(temp)
            out, err = root/"stdout.log", root/"stderr.log"
            code = "import sys;sys.stdout.write('x'*1024);sys.stderr.write('y'*1024)"
            with patch.object(wrapper, "MAX_LOG_BYTES", 32):
                result, truncated = wrapper.run_bounded_process(
                    [sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(), timeout=5,
                    stdout_path=out, stderr_path=err)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(out.stat().st_size, 32)
            self.assertEqual(err.stat().st_size, 32)
            self.assertTrue(truncated["stdout"])
            self.assertTrue(truncated["stderr"])

    def test_preflight_failure_evidence_is_fixed_bounded_and_non_overwriting(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            repo = Path(temp)/"repo"
            (repo/"outputs").mkdir(parents=True)
            error = ValueError("E"*1500)
            first = wrapper.write_preflight_failure(error, plan_sha256="a"*64,
                job_id="b"*64, repo_root=repo)
            second = wrapper.write_preflight_failure(error, plan_sha256="bad",
                job_id="../not-a-digest", repo_root=repo)
            record1, record2 = json.loads(first.read_text()), json.loads(second.read_text())
            self.assertEqual(first.parent, repo/"outputs/dispatcher-validation-failures")
            self.assertEqual(len(record1["message"]), 1024)
            self.assertEqual(record1["error_type"], "ValueError")
            self.assertEqual(record1["plan_sha256"], "a"*64)
            self.assertEqual(record1["job_id"], "b"*64)
            self.assertNotIn("plan_sha256", record2)
            self.assertNotIn("job_id", record2)
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(first.parent.glob("*.json"))), 2)

    def test_main_records_preflight_failure_without_traceback(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            repo = Path(temp)/"repo"
            (repo/"outputs").mkdir(parents=True)
            output = io.StringIO()
            args = ["--plan", "/untrusted/ignored", "--plan-sha256", "a"*64,
                    "--job-id", "b"*64, "--environment-lock", "/ignored"]
            with patch.object(wrapper, "ROOT", repo), \
                 patch.object(wrapper, "run_job", side_effect=ValueError("preflight rejected")), \
                 contextlib.redirect_stdout(output):
                code = wrapper.main(args)
            self.assertEqual(code, 1)
            summary = json.loads(output.getvalue())
            evidence = Path(summary["evidence_file"])
            self.assertEqual(evidence.parent, repo/"outputs/dispatcher-validation-failures")
            record = json.loads(evidence.read_text())
            self.assertEqual(record["message"], "preflight rejected")
            self.assertNotIn("traceback", evidence.read_text().lower())




if __name__ == "__main__":
    unittest.main()
