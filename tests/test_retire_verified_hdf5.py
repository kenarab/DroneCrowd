"""Retirement guards exercised on disposable fixtures, never on author data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retire_verified_hdf5 import (allowed_parts, assert_identity, digest, execute,
                                  plan, preserved_inventory)


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dronecrowd-retirement-test-", dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve() / "dataset"
        self.evidence = self.root.parent / "evidence"
        self.evidence.mkdir()
        self.target = self.root / "train_data/ground_truth/img001002.h5"
        self.retained = self.root / "test_data/ground_truth/img001003_max.h5"
        for path in (self.target, self.retained):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"disposable fixture")
        self.annotation = self.root / "annotations/00001.mat"
        self.annotation.parent.mkdir()
        self.annotation.write_bytes(b"original scientific annotation")
        backup = self.evidence / "sources/annotations/00001.mat"
        backup.parent.mkdir(parents=True)
        backup.write_bytes(self.annotation.read_bytes())
        info = self.target.stat()
        self.record = {"path": str(self.target.relative_to(self.root)), "device": info.st_dev,
                       "inode": info.st_ino, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                       "allocated": info.st_blocks * 512, "payload_verified": True, "differences": {}}
        error_record = {"path": str(self.retained.relative_to(self.root)), "error": "unreadable HDF5"}
        self.write_jsonl("files.jsonl", [self.record, error_record])
        self.write_jsonl("metadata-files.jsonl", [dict(self.record, metadata_verified=True),
                                                  dict(error_record, metadata_verified=False)])
        self.write_jsonl("preserved-files.jsonl", preserved_inventory(self.root).values())
        self.write_json("summary.json", {"root": str(self.root), "payload": True, "files": 2})
        self.write_json("metadata-summary.json", {"root": str(self.root),
                          "metadata_manifest_sha256": digest(self.evidence / "metadata-files.jsonl")})
        self.write_json("regeneration.json", {"root": str(self.root), "all_hashes_equal": True,
                                               "files": [{"hashes_equal": True}]})
        self.write_json("sources.json", {"root": str(self.root), "files": [{
            "relative_path": "annotations/00001.mat", "backup": "sources/annotations/00001.mat",
            "sha256": digest(self.annotation)}]})
        auditor = Path(__file__).resolve().parents[1] / "scripts/audit_hdf5.py"
        self.write_json("provenance.json", {"source_sha256": {"scripts/audit_hdf5.py": digest(auditor)}})

    def write_json(self, name, value):
        (self.evidence / name).write_text(json.dumps(value) + "\n")

    def write_jsonl(self, name, values):
        (self.evidence / name).write_text("".join(json.dumps(value) + "\n" for value in values))

    def test_path_guard_rejects_traversal_other_directories_and_other_names(self):
        for name in ("../img001002.h5", "/train_data/ground_truth/img001002.h5",
                     "train_data/images/img001002.h5", "train_data/ground_truth/weights.h5",
                     "train_data/../ground_truth/img001002.h5"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                allowed_parts(name)

    def test_plan_selects_only_verified_file_and_does_not_modify_it(self):
        before = self.target.read_bytes()
        candidates, retained = plan(self.root, self.evidence)
        self.assertEqual(candidates, [self.record])
        self.assertEqual(retained, [str(self.retained.relative_to(self.root))])
        self.assertEqual(before, self.target.read_bytes())

    def test_changed_source_blocks_retirement(self):
        self.annotation.write_bytes(b"changed scientific annotation")
        with self.assertRaisesRegex(ValueError, "Source"):
            plan(self.root, self.evidence)
        self.assertTrue(self.target.exists())

    def test_changed_target_blocks_retirement(self):
        self.target.write_bytes(b"new map data")
        with self.assertRaisesRegex(ValueError, "File changed"):
            plan(self.root, self.evidence)

    def test_hardlinks_and_symlinks_are_rejected(self):
        alias = self.root.parent / "alias"
        os.link(self.target, alias)
        with self.assertRaisesRegex(ValueError, "one link"):
            assert_identity(self.target.lstat(), self.record)
        alias.unlink()
        alias.symlink_to(self.target)
        with self.assertRaisesRegex(ValueError, "regular file"):
            assert_identity(alias.lstat(), self.record)

    def test_changed_metadata_evidence_blocks_retirement(self):
        with (self.evidence / "metadata-files.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "metadata evidence changed"):
            plan(self.root, self.evidence)

    def test_regeneration_from_another_dataset_is_rejected(self):
        self.write_json("regeneration.json", {"root": str(self.root.parent), "all_hashes_equal": True,
                                               "files": [{"hashes_equal": True}]})
        with self.assertRaisesRegex(ValueError, "regeneration"):
            plan(self.root, self.evidence)

    def test_prior_retirement_report_blocks_execution_before_first_unlink(self):
        self.write_json("retirement.json", {"previous": True})
        with self.assertRaises(FileExistsError):
            execute(self.root, self.evidence, [self.record])
        self.assertTrue(self.target.exists())

    def test_execution_rechecks_identity_and_records_failure(self):
        candidates, _ = plan(self.root, self.evidence)
        self.target.write_bytes(b"changed after planning")
        with self.assertRaisesRegex(ValueError, "File changed"):
            execute(self.root, self.evidence, candidates)
        self.assertTrue(self.target.exists())
        failure = json.loads((self.evidence / "retirement-failure.json").read_text())
        self.assertEqual(failure["deleted_files"], 0)
        self.assertFalse(failure["complete"])

    def test_execute_preserves_annotations_and_unverified_hdf5(self):
        candidates, _ = plan(self.root, self.evidence)
        expected = self.annotation.read_bytes()
        result = execute(self.root, self.evidence, candidates)
        self.assertEqual(result["deleted_files"], 1)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.retained.exists())
        self.assertEqual(expected, self.annotation.read_bytes())
        journal = (self.evidence / "deletions.jsonl").read_text().splitlines()
        self.assertEqual(len(journal), 1)
        self.assertTrue(json.loads(journal[0])["deleted"])


if __name__ == "__main__":
    unittest.main()
