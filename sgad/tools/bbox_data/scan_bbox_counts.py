import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger


def collect_npy_files(bbox_root: Path) -> List[Path]:
    npy_files: List[Path] = []
    for root, _, files in os.walk(bbox_root, followlinks=True):
        for fname in files:
            if fname.lower().endswith(".npy"):
                npy_files.append(Path(root) / fname)
    npy_files.sort()
    return npy_files


def count_file(idx: int, path: Path) -> tuple[int, int]:
    arr = np.load(path)
    if arr.ndim <= 1:
        count = 1 if arr.size >= 4 else 0
    else:
        count = int(arr.shape[0])
    return idx, count


def main():
    parser = argparse.ArgumentParser(description="Scan bbox npy files and record bbox counts.")
    parser.add_argument("--bbox_root", type=Path, required=True, help="Root directory containing generated bbox npy files.")
    parser.add_argument("--output", type=Path, default=None, help="Output npz path (default: <bbox_root>_bbox_counts.npz).")
    parser.add_argument("--relative_to", type=Path, default=None, help="Base directory for relative paths (default: bbox_root.parent).")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool size for counting npy files.")
    args = parser.parse_args()

    bbox_root = args.bbox_root.resolve()
    rel_base = args.relative_to.resolve() if args.relative_to else bbox_root.parent
    out_path = args.output or (bbox_root.parent / f"{bbox_root.name}_bbox_counts.npz")

    logger.info(f"Scanning bbox npy files under {bbox_root}")
    npy_files = collect_npy_files(bbox_root)
    logger.info(f"Total bbox files: {len(npy_files)}")

    rel_paths: List[str] = []
    counts: List[int] = [0] * len(npy_files)

    for path in npy_files:
        try:
            rel_paths.append(path.relative_to(rel_base).as_posix())
        except ValueError:
            logger.error(f"File {path} is not under base {rel_base}")
            rel_paths.append(path.as_posix())

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(count_file, idx, path) for idx, path in enumerate(npy_files)]
        for fut in as_completed(futures):
            idx, cnt = fut.result()
            counts[idx] = cnt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(npy_files) == 0:
        logger.warning(f"No .npy files found under {bbox_root}. Saved empty listing.")
    np.savez_compressed(out_path, paths=np.array(rel_paths, dtype=object), counts=np.array(counts, dtype=np.int32))
    logger.info(f"Saved counts to {out_path}")


if __name__ == "__main__":
    main()
