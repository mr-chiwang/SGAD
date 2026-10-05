import argparse
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from sgad.data.bbox_io import (
    BBoxCountLookup,
    infer_bbox_path,
    load_bbox_counts,
    resolve_bbox_dataset_root,
)


def build_lookup(counts_path: Path, count_rel_base: Path | None, bbox_root: Path) -> BBoxCountLookup:
    counts = load_bbox_counts(counts_path)
    bases = [bbox_root, bbox_root.parent]
    if count_rel_base is not None:
        bases.insert(0, count_rel_base)
    return BBoxCountLookup(counts, bases=bases)


def load_scene_list(list_path: Path) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as file:
        return [line.split()[0] for line in file if line.strip()]


def build_scannet_counts(
    npz_path: Path,
    data_root: Path,
    bbox_root: Path,
    lookup: BBoxCountLookup,
) -> np.ndarray:
    with np.load(npz_path) as data:
        names = data["name"]

    if len(names) == 0:
        return np.empty((0, 2), dtype=np.int32)

    frame_keys = np.concatenate([names[:, [0, 1, 2]], names[:, [0, 1, 3]]], axis=0)
    unique_frames, inverse = np.unique(frame_keys, axis=0, return_inverse=True)
    frame_counts = np.full(len(unique_frames), -1, dtype=np.int32)
    for index, (scene_id, scene_sub_id, frame) in enumerate(unique_frames):
        scene_name = f"scene{scene_id:04d}_{scene_sub_id:02d}"
        image = data_root / scene_name / "color" / f"{frame}.jpg"
        bbox = infer_bbox_path(image, data_root, bbox_root).with_suffix(".npy")
        count = lookup(bbox)
        if count is not None:
            frame_counts[index] = count

    pair_frame_indices = inverse.reshape(2, len(names)).T
    return frame_counts[pair_frame_indices]


def build_megadepth_counts(
    npz_path: Path,
    data_root: Path,
    bbox_root: Path,
    lookup: BBoxCountLookup,
) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    try:
        image_paths = data["image_paths"]
        pair_infos = data["pair_infos"]
        if len(pair_infos) == 0:
            return np.empty((0, 2), dtype=np.int32)

        if isinstance(pair_infos, np.ndarray) and pair_infos.ndim == 2:
            pair_indices = pair_infos[:, 0]
        else:
            pair_indices = [pair_info[0] for pair_info in pair_infos]
        pair_image_indices = np.stack(pair_indices).astype(np.int64, copy=False)
        unique_images, inverse = np.unique(pair_image_indices, return_inverse=True)
        image_counts = np.full(len(unique_images), -1, dtype=np.int32)
        for index, image_index in enumerate(unique_images):
            image = Path(image_paths[image_index])
            if not image.is_absolute():
                image = data_root / image
            bbox = infer_bbox_path(image, data_root, bbox_root).with_suffix(".npy")
            count = lookup(bbox)
            if count is not None:
                image_counts[index] = count
    finally:
        if hasattr(data, "close"):
            data.close()

    pair_image_indices = inverse.reshape(-1, 2)
    return image_counts[pair_image_indices]


def resolve_scene_names(npz_root: Path, list_paths: List[Path]) -> List[str]:
    if list_paths:
        names = []
        for list_path in list_paths:
            names.extend(load_scene_list(list_path))
    else:
        names = [path.name for path in npz_root.glob("*.npz")]

    names = [name if name.endswith(".npz") else f"{name}.npz" for name in names]
    return sorted(set(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one bbox-count sidecar aligned with every original LoFTR scene index."
    )
    parser.add_argument("--dataset", required=True, choices=["scannet", "megadepth"])
    parser.add_argument("--data_root", type=Path, required=True, help="Dataset image root.")
    parser.add_argument("--npz_root", type=Path, required=True, help="Unmodified LoFTR scene-index root.")
    parser.add_argument(
        "--list_path",
        type=Path,
        nargs="*",
        default=None,
        help="Optional original scene-list files used to select scene indices.",
    )
    parser.add_argument("--bbox_root", type=Path, required=True, help="Precomputed bbox root.")
    parser.add_argument("--counts_npz", type=Path, required=True, help="Per-image bbox count index.")
    parser.add_argument(
        "--count_rel_base",
        type=Path,
        default=None,
        help="Base used for paths in the per-image bbox count index.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output directory for per-scene .npy sidecars.",
    )
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    npz_root = args.npz_root.resolve()
    bbox_root = resolve_bbox_dataset_root(data_root, args.bbox_root.resolve())
    list_paths = [path.resolve() for path in args.list_path] if args.list_path else []
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    lookup = build_lookup(
        args.counts_npz.resolve(),
        args.count_rel_base.resolve() if args.count_rel_base else None,
        bbox_root,
    )
    scene_names = resolve_scene_names(npz_root, list_paths)
    if not scene_names:
        raise RuntimeError(f"No scene indices found under {npz_root}")

    total_pairs = 0
    missing_pairs = 0
    iterator = tqdm(scene_names, desc="pair bbox counts", disable=args.disable_progress)
    for scene_name in iterator:
        npz_path = npz_root / scene_name
        if not npz_path.exists():
            logger.warning(f"Skipping missing scene index: {npz_path}")
            continue

        if args.dataset == "scannet":
            counts = build_scannet_counts(npz_path, data_root, bbox_root, lookup)
        else:
            counts = build_megadepth_counts(npz_path, data_root, bbox_root, lookup)

        np.save(out_root / f"{npz_path.stem}.npy", counts, allow_pickle=False)
        total_pairs += len(counts)
        missing_pairs += int((counts < 0).any(axis=1).sum())

    logger.info(
        f"Saved {len(scene_names)} sidecars under {out_root}; "
        f"pairs={total_pairs}, missing_bbox_counts={missing_pairs}"
    )


if __name__ == "__main__":
    main()
