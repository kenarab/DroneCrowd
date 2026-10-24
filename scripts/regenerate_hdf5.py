"""Regenerate selected historical maps into a separate, new destination.

The original sequence MAT files are the source of truth; XML and MAT have
historical coordinate differences. Existing output files are never overwritten.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import resource

import numpy as np

from audit_hdf5 import Annotations, HDF5Reader, reconstruct_density, reconstruct_localization


def reference_metadata(root, evidence, selected):
    """Validate preserved sources and find hashes even after originals retire."""
    sources = json.loads((evidence / "sources.json").read_text())
    if sources["root"] != str(root):
        raise ValueError("Reference evidence belongs to another dataset")
    for record in sources["files"]:
        relative = Path(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path")
        path = root / relative
        if path.resolve(strict=True) != path:
            raise ValueError("Source links are not allowed")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Preserved source changed: {relative}")
    metadata = {}
    with (evidence / "files.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["path"] in selected:
                if (record.get("payload_verified") is not True
                        or record.get("differences") != {} or record.get("error")):
                    raise ValueError("Reference map was not verified")
                if record["path"] in metadata:
                    raise ValueError("Duplicate reference map")
                metadata[record["path"]] = record["datasets"]
    if set(metadata) != set(selected):
        raise ValueError("Selected maps are missing from the reference evidence")
    return metadata


def write_new_file(path, arrays, reader):
    """Write numeric arrays exclusively; caller chooses a separate destination."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    lib = reader.lib
    signatures = {
        "H5Fcreate": (ct.c_int64, [ct.c_char_p, ct.c_uint, ct.c_int64, ct.c_int64]),
        "H5Screate_simple": (ct.c_int64, [ct.c_int, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)]),
        "H5Dcreate2": (ct.c_int64, [ct.c_int64, ct.c_char_p] + [ct.c_int64] * 5),
        "H5Dwrite": (ct.c_int, [ct.c_int64] * 5 + [ct.c_void_p]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    file_id = reader.checked(lib.H5Fcreate(os.fsencode(path), 4, 0, 0))
    try:
        for name, array in arrays.items():
            native = {np.dtype("float32"): "H5T_NATIVE_FLOAT_g",
                      np.dtype("float64"): "H5T_NATIVE_DOUBLE_g",
                      np.dtype("int16"): "H5T_NATIVE_SHORT_g"}[array.dtype]
            datatype = ct.c_int64.in_dll(lib, native).value
            if not array.flags.c_contiguous:
                raise ValueError("Expected contiguous array")
            dimensions = (ct.c_uint64 * array.ndim)(*array.shape)
            space = reader.checked(lib.H5Screate_simple(array.ndim, dimensions, None))
            try:
                dataset = reader.checked(lib.H5Dcreate2(file_id, name.encode(), datatype, space, 0, 0, 0))
                try:
                    reader.checked(lib.H5Dwrite(dataset, datatype, 0, 0, 0, array.ctypes.data))
                finally:
                    lib.H5Dclose(dataset)
            finally:
                lib.H5Sclose(space)
    finally:
        lib.H5Fclose(file_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="Relative path of a map to regenerate")
    parser.add_argument("--reference-evidence", type=Path,
                        help="Preserved audit evidence; enables regeneration after original maps were retired")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 2**20, 1536 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
    root = args.root.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    temporary = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not output.is_relative_to(temporary) or output == temporary:
        parser.error("Bounded regeneration must use its own authorized temporary directory")
    if any(output.iterdir()):
        parser.error("Output directory must be empty; existing files or links are not allowed")
    if output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("Choose an output directory separate from the dataset")
    if len(args.source) > 12:
        parser.error("This bounded validation accepts at most 12 files")
    if (output / "regeneration.json").exists():
        parser.error("Output report already exists")
    reader, annotations, records = HDF5Reader(), Annotations(root), []
    references = (reference_metadata(root, args.reference_evidence.resolve(strict=True), args.source)
                  if args.reference_evidence else None)
    for relative in args.source:
        parts = Path(relative).parts
        if (len(parts) != 3 or parts[0] not in ("train_data", "test_data")
                or parts[1] != "ground_truth"):
            parser.error("Expected split/ground_truth/map.h5")
        match = re.fullmatch(r"img(\d{3})(\d{3})(_max)?\.h5", parts[2])
        if not match:
            parser.error("Unrecognized map name")
        source = root / relative
        if source.is_symlink() or source.parent.resolve(strict=True) != source.parent:
            parser.error("Source links are not allowed")
        sequence, frame = map(int, match.groups()[:2])
        points = annotations.frame(sequence, frame)
        arrays = (reconstruct_localization(points) if match[3]
                  else {"density": reconstruct_density(points)})
        if references is None:
            original_metadata, _ = reader.read(source, True)
        else:
            original_metadata = references[relative]
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.resolve(strict=True) != destination.parent:
            raise ValueError("Output parent links are not allowed")
        write_new_file(destination, arrays, reader)
        regenerated_metadata, _ = reader.read(destination, True)
        if json.dumps(original_metadata, sort_keys=True) != json.dumps(regenerated_metadata, sort_keys=True):
            raise ValueError(f"Regenerated hashes/shapes/types differ: {relative}")
        records.append({"source": str(source), "regenerated": str(destination),
                        "datasets": regenerated_metadata, "bytes": destination.stat().st_size,
                        "hashes_equal": True})
    report = {"root": str(root), "output": str(output), "files": records,
              "total_bytes": sum(r["bytes"] for r in records), "all_hashes_equal": True,
              "reference_evidence": str(args.reference_evidence) if args.reference_evidence else None}
    with (output / "regeneration.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"files": len(records), "bytes": report["total_bytes"], "all_hashes_equal": True}))


if __name__ == "__main__":
    main()
