import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np
from loguru import logger


def load_counts(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    paths = data["paths"].tolist()
    counts = data["counts"].tolist()
    return list(zip(paths, counts))


def dump_threshold(threshold, pairs, out_dir: Path, fmt: str, stem: str):
    filtered = [path for path, cnt in pairs if cnt < threshold]
    suffix = f"lt{threshold}"
    out_path = out_dir / f"{stem}_{suffix}.{fmt}"
    if fmt == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
    else:
        out_path.write_text("\n".join(filtered), encoding="utf-8")
    logger.info(f"Threshold <{threshold}: {len(filtered)} entries saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze bbox counts and list files below thresholds.")
    parser.add_argument("--counts_npz", type=Path, required=True, help="Output of scan_bbox_counts.py")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save threshold files (default: <counts_npz.parent>/<counts_npz.stem>_analysis)",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=[1, 3, 5], help="Thresholds for bbox counts")
    parser.add_argument("--format", type=str, default="txt", choices=["txt", "json"], help="Output format")
    parser.add_argument("--workers", type=int, default=4, help="Thread pool size when exporting multiple thresholds.")
    args = parser.parse_args()

    counts_npz = args.counts_npz.resolve()
    thresholds = sorted(set(args.thresholds))
    out_dir = (args.out_dir.resolve() if args.out_dir else counts_npz.parent / f"{counts_npz.stem}_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_counts(counts_npz)
    logger.info(f"Loaded {len(pairs)} entries from {counts_npz}")

    stem = counts_npz.stem
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(dump_threshold, thr, pairs, out_dir, args.format, stem) for thr in thresholds]
        for fut in futures:
            fut.result()


if __name__ == "__main__":
    main()
