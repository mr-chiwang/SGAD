"""Extract numeric XYWH arrays from saved SAM region records."""

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def sam_records_to_bboxes(records) -> np.ndarray:
    """Return SAM ``bbox`` values as a C-contiguous ``(N, 4)`` float32 array.

    Records are preserved in their input order. No scaling, filtering, sorting,
    deduplication, or geometry recomputation is performed.
    """
    bboxes = []
    for index, record in enumerate(records):
        if isinstance(record, np.ndarray) and record.shape == ():
            record = record.item()
        if not isinstance(record, Mapping):
            raise TypeError(f"Region {index} must be a mapping, got {type(record)!r}")
        if "bbox" not in record:
            raise KeyError(f"Region {index} does not contain a 'bbox' field")

        bbox = np.asarray(record["bbox"], dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"Region {index} bbox must have shape (4,), got {bbox.shape}")
        if not np.isfinite(bbox).all():
            raise ValueError(f"Region {index} bbox contains non-finite values: {bbox.tolist()}")
        bboxes.append(bbox)

    if not bboxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.ascontiguousarray(np.stack(bboxes), dtype=np.float32)


def convert_sam_file(input_path: Path, output_path: Path, overwrite: bool = False) -> tuple[int, int]:
    """Convert one SAM object-array file and return ``(region_count, byte_count)``."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    records = np.load(input_path, allow_pickle=True)
    bboxes = sam_records_to_bboxes(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.npy")
    try:
        np.save(temporary_path, bboxes, allow_pickle=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return len(bboxes), output_path.stat().st_size


def convert_sam_tree(input_root: Path, output_root: Path, overwrite: bool = False) -> dict[str, int]:
    """Convert every ``.npy`` below ``input_root`` while preserving relative paths."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist or is not a directory: {input_root}")

    input_files = sorted(input_root.rglob("*.npy"))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found below {input_root}")

    stats = {"files": 0, "regions": 0, "empty_files": 0, "bytes": 0}
    for input_path in input_files:
        output_path = output_root / input_path.relative_to(input_root)
        region_count, byte_count = convert_sam_file(input_path, output_path, overwrite=overwrite)
        stats["files"] += 1
        stats["regions"] += region_count
        stats["empty_files"] += int(region_count == 0)
        stats["bytes"] += byte_count

    return stats


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bbox fields from SAM object-array files as numeric N x 4 float32 XYWH arrays."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = convert_sam_tree(args.input_root, args.output_root, overwrite=args.overwrite)
    print(
        "Converted {files} files and {regions} regions "
        "({empty_files} empty files, {bytes} output bytes).".format(**stats)
    )


if __name__ == "__main__":
    main()
