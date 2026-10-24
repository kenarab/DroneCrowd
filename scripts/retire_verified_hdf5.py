"""Plan (default) or retire only HDF5 maps proven regenerable by both audits.

Requires permanent evidence, unchanged source annotations and per-file identity
checks. Never follows links, deletes directories, or selects files by a glob.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_records(path):
    records = {}
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            key = record["path"]
            if key in records:
                raise ValueError(f"Duplicate evidence path: {key}")
            records[key] = record
    return records


def allowed_parts(relative):
    parts = Path(relative).parts
    if (len(parts) != 3 or parts[0] not in ("train_data", "test_data")
            or parts[1] != "ground_truth"
            or not re.fullmatch(r"img\d{6}(?:_max)?\.h5", parts[2])):
        raise ValueError(f"Path outside retirement scope: {relative}")
    return parts


def assert_identity(info, record):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Expected a regular file with one link: {record['path']}")
    current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    expected = tuple(record[k] for k in ("device", "inode", "size", "mtime_ns"))
    if current != expected:
        raise ValueError(f"File changed since verification: {record['path']}")


def preserved_inventory(root):
    records = {}
    for directory, subdirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() == ".h5":
                continue
            info = path.lstat()
            relative = str(path.relative_to(root))
            records[relative] = {"path": relative, "device": info.st_dev, "inode": info.st_ino,
                                 "size": info.st_size, "mtime_ns": info.st_mtime_ns, "mode": info.st_mode}
    return records


def verify_sources(root, evidence):
    sources = json.loads((evidence / "sources.json").read_text())
    if sources["root"] != str(root):
        raise ValueError("Source manifest belongs to another dataset")
    for record in sources["files"]:
        relative = Path(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path")
        source = root / relative
        backup = evidence / record["backup"]
        if (source.resolve(strict=True) != source or not backup.resolve(strict=True).is_relative_to(evidence)
                or digest(source) != record["sha256"] or digest(backup) != record["sha256"]):
            raise ValueError(f"Source or backup changed: {relative}")
    provenance = json.loads((evidence / "provenance.json").read_text())
    script = Path(__file__).resolve().with_name("audit_hdf5.py")
    if digest(script) != provenance["source_sha256"]["scripts/audit_hdf5.py"]:
        raise ValueError("Auditor differs from the audited version")
    for source, expected in provenance["source_sha256"].items():
        path = Path(source)
        if path.is_absolute() and path.is_relative_to(root) and digest(path) != expected:
            raise ValueError(f"Original source changed since audit launch: {path}")
    if preserved_inventory(root) != read_records(evidence / "preserved-files.jsonl"):
        raise ValueError("Preserved non-HDF5 inventory changed")


def plan(root, evidence):
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("Dataset root must be a real directory without parent links")
    if evidence.resolve(strict=True) != evidence or evidence.is_relative_to(root):
        raise ValueError("Evidence must be a separate real directory")
    summary = json.loads((evidence / "summary.json").read_text())
    metadata_summary = json.loads((evidence / "metadata-summary.json").read_text())
    regeneration = json.loads((evidence / "regeneration.json").read_text())
    if (summary["root"] != str(root) or summary["payload"] is not True
            or metadata_summary["root"] != str(root)):
        raise ValueError("Missing complete payload/metadata audits for this root")
    if (regeneration.get("root") != str(root)
            or regeneration["all_hashes_equal"] is not True or not regeneration["files"]
            or any(r["hashes_equal"] is not True for r in regeneration["files"])):
        raise ValueError("On-disk regeneration has not passed")
    if digest(evidence / "metadata-files.jsonl") != metadata_summary["metadata_manifest_sha256"]:
        raise ValueError("Independent metadata evidence changed")
    records = read_records(evidence / "files.jsonl")
    metadata = read_records(evidence / "metadata-files.jsonl")
    if len(records) != summary["files"] or set(records) != set(metadata):
        raise ValueError("Incomplete or mismatched evidence inventories")
    # New/unseen HDF5 files are not silently swept into the retirement.
    present = {str(p.relative_to(root)) for p in root.rglob("*.h5")}
    if present != set(records):
        raise ValueError("HDF5 inventory changed since audit")
    verify_sources(root, evidence)
    candidates, retained = [], []
    for relative, record in records.items():
        other = metadata[relative]
        if (record.get("payload_verified") is not True or record.get("differences") != {}
                or record.get("error") or other.get("metadata_verified") is not True or other.get("error")):
            retained.append(relative)
            continue
        allowed_parts(relative)
        for field in ("device", "inode", "size", "mtime_ns"):
            if record[field] != other[field]:
                raise ValueError(f"Audits refer to different file versions: {relative}")
        path = root / relative
        if path.resolve(strict=True) != path:
            raise ValueError(f"Parent/file link detected: {relative}")
        assert_identity(path.lstat(), record)
        candidates.append(record)
    return candidates, retained


def execute(root, evidence, candidates):
    for name in ("deletions.jsonl", "retirement.json", "retirement-failure.json"):
        if (evidence / name).exists() or (evidence / name).is_symlink():
            raise FileExistsError(f"Prior retirement evidence exists: {name}")
    # Open parents with O_NOFOLLOW and operate by dir_fd, avoiding ancestor races.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parents, root_fd = {}, os.open(root, flags)
    before = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    started = time.monotonic()
    deleted_count = 0
    try:
        for split in ("train_data", "test_data"):
            split_fd = os.open(split, flags, dir_fd=root_fd)
            try:
                parents[split] = os.open("ground_truth", flags, dir_fd=split_fd)
            finally:
                os.close(split_fd)
        with (evidence / "deletions.jsonl").open("x") as journal:
            for index, record in enumerate(candidates, 1):
                split, _, name = allowed_parts(record["path"])
                parent_fd = parents[split]
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                assert_identity(info, record)
                os.unlink(name, dir_fd=parent_fd)
                deleted_count += 1
                journal.write(json.dumps({"path": record["path"], "allocated": record["allocated"],
                                          "inode": info.st_ino, "deleted": True}) + "\n")
                if index % 100 == 0:
                    journal.flush()
                    os.fsync(journal.fileno())
            journal.flush()
            os.fsync(journal.fileno())
        for parent in parents.values():
            os.fsync(parent)
    except Exception as error:
        with (evidence / "retirement-failure.json").open("x") as report:
            json.dump({"deleted_files": deleted_count, "complete": False,
                       "error": f"{type(error).__name__}: {error}"}, report)
            report.write("\n")
        raise
    finally:
        for parent in parents.values():
            os.close(parent)
        os.close(root_fd)
    verify_sources(root, evidence)
    after = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    result = {"deleted_files": len(candidates), "allocated_bytes_retired": sum(r["allocated"] for r in candidates),
              "available_bytes_before": before, "available_bytes_after": after,
              "observed_available_increase": after - before, "preserved_inventory_unchanged": True,
              "seconds": round(time.monotonic() - started, 1)}
    with (evidence / "retirement.json").open("x") as report:
        json.dump(result, report, indent=2)
        report.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    root, evidence = args.root.absolute(), args.evidence_dir.absolute()
    candidates, retained = plan(root, evidence)
    manifest_hash = digest(evidence / "files.jsonl")
    print(json.dumps({"candidates": len(candidates), "retained": retained,
                      "allocated_bytes": sum(r["allocated"] for r in candidates),
                      "manifest_sha256": manifest_hash, "execute": args.execute}), flush=True)
    if args.execute:
        if args.expected_count != len(candidates) or args.expected_manifest_sha256 != manifest_hash:
            parser.error("Execution requires the exact reviewed candidate count and manifest SHA-256")
        print(json.dumps(execute(root, evidence, candidates)), flush=True)


if __name__ == "__main__":
    main()
