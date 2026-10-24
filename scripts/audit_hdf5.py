"""Audit historical DroneCrowd maps without modifying the dataset.

Uses the installed HDF5 C library (no h5py installation). Every deletion
candidate must match a reconstruction from the original sequence annotations.
Reports and per-file evidence belong outside Git. This script never deletes.
"""
import argparse
from collections import Counter
import ctypes as ct
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
from scipy import io, spatial


class HDF5Reader:
    """Read only flat numeric files; reject attributes and unfamiliar storage."""

    def __init__(self):
        library = ctypes.util.find_library("hdf5_serial")
        if not library:
            raise RuntimeError("System HDF5 library is unavailable")
        self.lib = ct.CDLL(library)
        class LinkInfo(ct.Structure):
            _fields_ = [("type", ct.c_int), ("corder_valid", ct.c_bool),
                        ("corder", ct.c_int64), ("cset", ct.c_int), ("value", ct.c_uint64)]

        self.LinkInfo = LinkInfo
        identifier, integer, size = ct.c_int64, ct.c_int, ct.c_uint64
        signatures = {
            "H5Fopen": (identifier, [ct.c_char_p, ct.c_uint, identifier]),
            "H5Fclose": (integer, [identifier]),
            "H5Gopen2": (identifier, [identifier, ct.c_char_p, identifier]),
            "H5Gclose": (integer, [identifier]),
            "H5Gget_num_objs": (integer, [identifier, ct.POINTER(size)]),
            "H5Gget_objname_by_idx": (ct.c_ssize_t, [identifier, size, ct.c_char_p, ct.c_size_t]),
            "H5Gget_objtype_by_idx": (integer, [identifier, size]),
            "H5Lget_info1": (integer, [identifier, ct.c_char_p, ct.POINTER(LinkInfo), identifier]),
            "H5Aget_num_attrs": (integer, [identifier]),
            "H5Dopen2": (identifier, [identifier, ct.c_char_p, identifier]),
            "H5Dclose": (integer, [identifier]),
            "H5Dget_type": (identifier, [identifier]),
            "H5Dget_space": (identifier, [identifier]),
            "H5Dget_create_plist": (identifier, [identifier]),
            "H5Dread": (integer, [identifier, identifier, identifier, identifier, identifier, ct.c_void_p]),
            "H5Dget_storage_size": (size, [identifier]),
            "H5Sget_simple_extent_ndims": (integer, [identifier]),
            "H5Sget_simple_extent_dims": (integer, [identifier, ct.POINTER(size), ct.POINTER(size)]),
            "H5Sclose": (integer, [identifier]),
            "H5Tget_class": (integer, [identifier]),
            "H5Tget_size": (ct.c_size_t, [identifier]),
            "H5Tget_sign": (integer, [identifier]),
            "H5Tget_order": (integer, [identifier]),
            "H5Tclose": (integer, [identifier]),
            "H5Pget_layout": (integer, [identifier]),
            "H5Pget_nfilters": (integer, [identifier]),
            "H5Pget_external_count": (integer, [identifier]),
            "H5Pclose": (integer, [identifier]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, arguments

    @staticmethod
    def checked(value):
        if value < 0:
            raise ValueError("HDF5 operation failed")
        return value

    def read(self, path, payload=False):
        lib = self.lib
        file_id = self.checked(lib.H5Fopen(os.fsencode(path), 0, 0))
        result, arrays = {}, {}
        try:
            group = self.checked(lib.H5Gopen2(file_id, b"/", 0))
            try:
                if lib.H5Aget_num_attrs(group) != 0:
                    raise ValueError("Root attributes require manual preservation")
                count = ct.c_uint64()
                self.checked(lib.H5Gget_num_objs(group, ct.byref(count)))
                if count.value > 16:
                    raise ValueError("Unexpected number of datasets")
                for index in range(count.value):
                    if lib.H5Gget_objtype_by_idx(group, index) != 1:
                        raise ValueError("Non-dataset object requires manual preservation")
                    name_buffer = ct.create_string_buffer(256)
                    length = self.checked(lib.H5Gget_objname_by_idx(group, index, name_buffer, 256))
                    if length >= 256:
                        raise ValueError("Unexpectedly long dataset name")
                    name = name_buffer.value.decode("utf-8")
                    link = self.LinkInfo()
                    self.checked(lib.H5Lget_info1(group, name_buffer.value, ct.byref(link), 0))
                    if link.type != 0:
                        raise ValueError("Soft/external link requires manual preservation")
                    dataset = self.checked(lib.H5Dopen2(group, name_buffer.value, 0))
                    handles = []
                    try:
                        if lib.H5Aget_num_attrs(dataset) != 0:
                            raise ValueError("Dataset attributes require manual preservation")
                        datatype = self.checked(lib.H5Dget_type(dataset))
                        handles.append((lib.H5Tclose, datatype))
                        space = self.checked(lib.H5Dget_space(dataset))
                        handles.append((lib.H5Sclose, space))
                        properties = self.checked(lib.H5Dget_create_plist(dataset))
                        handles.append((lib.H5Pclose, properties))
                        rank = self.checked(lib.H5Sget_simple_extent_ndims(space))
                        if rank not in (2, 3):
                            raise ValueError("Unexpected dataset rank")
                        dimensions = (ct.c_uint64 * rank)()
                        self.checked(lib.H5Sget_simple_extent_dims(space, dimensions, None))
                        shape = tuple(dimensions)
                        kind, width = lib.H5Tget_class(datatype), lib.H5Tget_size(datatype)
                        if kind not in (0, 1) or width not in (2, 4, 8):
                            raise ValueError("Unexpected non-numeric datatype")
                        if lib.H5Tget_order(datatype) != 0:
                            raise ValueError("Expected little-endian data")
                        if kind == 0 and lib.H5Tget_sign(datatype) != 1:
                            raise ValueError("Expected signed integer data")
                        dtype = np.dtype(("<f" if kind == 1 else "<i") + str(width))
                        storage = lib.H5Dget_storage_size(dataset)
                        byte_count = int(np.prod(shape)) * width
                        if not 0 < byte_count <= 16 * 2**20:
                            raise ValueError("Dataset exceeds bounded read budget")
                        if (lib.H5Pget_layout(properties) != 1
                                or lib.H5Pget_nfilters(properties) != 0
                                or lib.H5Pget_external_count(properties) != 0
                                or storage != byte_count):
                            raise ValueError("Unexpected compressed, external or unallocated storage")
                        result[name] = {"shape": shape, "dtype": dtype.str, "bytes": storage}
                        if payload:
                            array = np.empty(shape, dtype=dtype)
                            self.checked(lib.H5Dread(dataset, datatype, 0, 0, 0, array.ctypes.data))
                            if not np.isfinite(array).all():
                                raise ValueError("Non-finite numeric data")
                            arrays[name] = array
                            result[name]["sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
                    finally:
                        for close, handle in reversed(handles):
                            close(handle)
                        lib.H5Dclose(dataset)
            finally:
                lib.H5Gclose(group)
        finally:
            lib.H5Fclose(file_id)
        return result, arrays


def reconstruct_localization(points, height=540, width=960):
    identity = np.zeros((height, width), dtype=np.int16)
    offsets = [np.zeros((height // scale, width // scale, 2), dtype=np.float64)
               for scale in (1, 2, 4)]
    for point in points:
        for level, offset in enumerate(offsets):
            scale = 2 ** (level + 1)
            rows, columns = offset.shape[:2]
            y, x = int(point[1] / scale), int(point[0] / scale)
            if x < columns and y < rows:
                if y < 0 or x < 0:
                    raise ValueError("Negative annotation coordinates")
                if level == 0:
                    identity[y, x] = int(point[2])
                radius = int(5 / (level + 1.0))
                left, right = max(0, x - radius), min(columns - 1, x + radius)
                top, bottom = max(0, y - radius), min(rows - 1, y + radius)
                offset[top:bottom + 1, left:right + 1, 0] = y - np.arange(top, bottom + 1)[:, None]
                offset[top:bottom + 1, left:right + 1, 1] = x - np.arange(left, right + 1)[None, :]
    return dict(identity=identity, **{f"offset{i + 1}": a for i, a in enumerate(offsets)})


def reconstruct_density(points, height=540, width=960):
    """Same Gaussian operations/order as make_data_density.py, on finite support.

    scipy.ndimage.gaussian_filter applies axis 0 then axis 1, retaining float32
    intermediates. Restricting each impulse to its finite support avoids scanning
    the full image for every annotated head; edge contributions are truncated.
    """
    occupied = np.zeros((height, width), dtype=np.bool_)
    for point in points:
        y, x = int(point[1] / 2), int(point[0] / 2)
        if x < width and y < height:
            if x < 0 or y < 0:
                raise ValueError("Negative annotation coordinates")
            occupied[y, x] = True
    ys, xs = np.nonzero(occupied)
    count = len(xs)
    density = np.zeros((height, width), dtype=np.float32)
    if not count:
        return density
    if count < 4 and count > 1:
        raise ValueError("Historical generator has infinite sigma for 2 or 3 occupied pixels")
    coordinates = np.column_stack((xs, ys))
    distances, _ = spatial.KDTree(coordinates.copy(), leafsize=2048).query(coordinates, k=4)
    for index, (x, y) in enumerate(coordinates):
        sigma = float(sum(distances[index, 1:4]) * 0.1) if count > 1 else (height + width) / 8.0
        radius = int(4.0 * sigma + 0.5)
        positions = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 / sigma**2 * positions**2)
        kernel /= kernel.sum()
        left, right = max(0, x - radius), min(width, x + radius + 1)
        top, bottom = max(0, y - radius), min(height, y + radius + 1)
        vertical = kernel[top - y + radius:bottom - y + radius].astype(np.float32)
        horizontal = kernel[left - x + radius:right - x + radius]
        contribution = (vertical[:, None].astype(np.float64) * horizontal[None, :]).astype(np.float32)
        density[top:bottom, left:right] += contribution
    return density


class Annotations:
    def __init__(self, root):
        self.root, self.sequence, self.frames = root, None, None

    def frame(self, sequence, frame):
        if self.sequence != sequence:
            path = self.root / "annotations" / f"{sequence:05d}.mat"
            data = io.loadmat(path)["anno"]
            self.frames = {}
            for index in np.unique(data[:, 0]).astype(int):
                rows = data[data[:, 0] == index]
                self.frames[index + 1] = np.column_stack(((rows[:, 2] + rows[:, 4]) / 2,
                                                         (rows[:, 3] + rows[:, 5]) / 2,
                                                         rows[:, 1] + 1))
            self.sequence = sequence
        return self.frames.get(frame, np.empty((0, 3)))


def audit_file(path, root, reader, annotations, payload):
    before = path.lstat()
    if path.is_symlink() or before.st_nlink != 1:
        raise ValueError("Symlink/hardlink requires manual preservation")
    match = re.fullmatch(r"img(\d{3})(\d{3})(_max)?\.h5", path.name)
    if not match:
        raise ValueError("Unrecognized filename")
    sequence, frame = map(int, match.groups()[:2])
    localization = bool(match[3])
    metadata, arrays = reader.read(path, payload)
    expected = ({"identity": ((540, 960), "<i2"), "offset1": ((540, 960, 2), "<f8"),
                 "offset2": ((270, 480, 2), "<f8"), "offset3": ((135, 240, 2), "<f8")}
                if localization else {"density": ((540, 960), "<f4")})
    if set(metadata) != set(expected):
        raise ValueError(f"Unexpected datasets: {list(metadata)}")
    for name, (shape, dtype) in expected.items():
        if metadata[name]["shape"] != shape or metadata[name]["dtype"] != dtype:
            raise ValueError(f"Unexpected shape/dtype: {metadata}")
    points = annotations.frame(sequence, frame)
    if not (root / "annotations" / f"{sequence:05d}.xml").is_file():
        raise ValueError("Missing original XML annotations")
    mat_path = path.parent / f"GT_img{sequence:03d}{frame:03d}.mat"
    if mat_path.is_file():
        per_image = io.loadmat(mat_path)["image_info"][0, 0][0, 0][0]
        if not np.array_equal(points, per_image):
            raise ValueError("Original sequence annotation and per-image MAT differ")
    record = {"path": str(path.relative_to(root)), "kind": "localization" if localization else "density",
              "size": before.st_size, "allocated": before.st_blocks * 512,
              "inode": before.st_ino, "device": before.st_dev, "mtime_ns": before.st_mtime_ns,
              "sequence": sequence, "frame": frame, "heads": len(points), "datasets": metadata,
              "has_image": (path.parent.parent / "images" / f"img{sequence:03d}{frame:03d}.jpg").is_file(),
              "has_per_image_annotation": mat_path.is_file(), "payload_verified": False}
    if payload:
        regenerated = (reconstruct_localization(points) if localization
                       else {"density": reconstruct_density(points)})
        differences = {}
        for name, actual in arrays.items():
            wanted = regenerated[name]
            if not np.array_equal(actual, wanted):
                differences[name] = {"different_values": int(np.count_nonzero(actual != wanted)),
                                     "max_absolute_error": float(np.max(np.abs(actual - wanted)))}
        record["differences"] = differences
        record["payload_verified"] = not differences
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("File changed during audit")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--payload", action="store_true", help="Read every value and compare reconstruction")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kind", choices=("density", "localization"))
    args = parser.parse_args()
    # A sequential CPU reader, never a model/GPU load. Bounded scratch <128MiB.
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 2**20, 1536 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 2**20, 128 * 2**20))
    root = args.root.resolve(strict=True)
    report_dir = args.report_dir.resolve(strict=True)
    temporary = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not report_dir.is_relative_to(temporary):
        parser.error("Reports must be inside the authorized temporary directory")
    paths = sorted(root.rglob("*.h5"))
    if args.kind:
        paths = [p for p in paths if p.name.endswith("_max.h5") == (args.kind == "localization")]
    if args.limit:
        paths = paths[:args.limit]
    reader, annotations = HDF5Reader(), Annotations(root)
    counts, allocated = Counter(), Counter()
    started = time.monotonic()
    with (report_dir / "files.jsonl").open("x") as report:
        for index, path in enumerate(paths, 1):
            try:
                record = audit_file(path, root, reader, annotations, args.payload)
                status = ("verified" if record["payload_verified"] else "different") if args.payload else "metadata_ok"
                counts[f"{record['kind']}:{status}"] += 1
                allocated[f"{record['kind']}:{status}"] += record["allocated"]
            except Exception as error:
                record = {"path": str(path.relative_to(root)), "error": f"{type(error).__name__}: {error}"}
                counts["errors"] += 1
            report.write(json.dumps(record, sort_keys=True) + "\n")
            if index % 100 == 0:
                report.flush()
                print(json.dumps({"done": index, "total": len(paths), "seconds": round(time.monotonic() - started, 1),
                                  "counts": counts}), flush=True)
    summary = {"root": str(root), "payload": args.payload, "files": len(paths), "counts": counts,
               "allocated_bytes": allocated, "seconds": round(time.monotonic() - started, 1),
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
