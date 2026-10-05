import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from loguru import logger


def resolve_bbox_dataset_root(image_root, bbox_root):
    image_root = Path(image_root)
    if bbox_root is None:
        return image_root

    bbox_root = Path(bbox_root)
    split_name = image_root.name
    candidate = bbox_root / split_name
    if candidate.exists():
        return candidate
    return bbox_root


def infer_bbox_path(image_path, root_dir, bbox_root=None):
    image_path = Path(image_path)
    root_dir = Path(root_dir)
    target_root = Path(bbox_root) if bbox_root is not None else root_dir

    try:
        rel = image_path.relative_to(root_dir)
    except ValueError:
        rel = image_path

    parent = rel.parent
    if parent == Path("."):
        bboxes_parent = Path("bboxes")
    else:
        bboxes_parent = parent.parent / "bboxes"

    return target_root / bboxes_parent / rel.name


def read_bboxes_from_file(base_path):
    base_path = Path(base_path)
    candidates = [base_path.with_suffix(".npy"), base_path.with_suffix(".json")]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".npy":
                bboxes_array = np.load(candidate)
            else:
                with open(candidate, "r", encoding="utf-8") as f:
                    bboxes_array = np.array(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load bbox file {candidate}: {e}")
            continue

        bboxes_array = np.asarray(bboxes_array, dtype=np.float32)
        if bboxes_array.ndim == 1:
            if bboxes_array.size == 0:
                return torch.empty((0, 4), dtype=torch.float32)
            bboxes_array = bboxes_array.reshape(-1, 4)
        return torch.from_numpy(bboxes_array)

    return torch.empty((0, 4), dtype=torch.float32)


def resize_bboxes_to_image(bboxes, src_shape, dst_shape):
    if bboxes.numel() == 0 or src_shape is None or dst_shape is None:
        return bboxes

    src_h, src_w = src_shape.tolist()
    dst_h, dst_w = dst_shape.tolist()
    if src_h == 0 or src_w == 0 or (src_h == dst_h and src_w == dst_w):
        return bboxes

    scale_w = dst_w / src_w
    scale_h = dst_h / src_h

    scaled = bboxes.to(torch.float32).clone()
    scaled[:, 0] = scaled[:, 0] * scale_w
    scaled[:, 2] = scaled[:, 2] * scale_w
    scaled[:, 1] = scaled[:, 1] * scale_h
    scaled[:, 3] = scaled[:, 3] * scale_h
    return scaled


def load_bbox_counts(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    try:
        if hasattr(data, "files"):
            paths = data["paths"].tolist()
            counts = data["counts"].tolist()
        elif isinstance(data, dict):
            paths = data["paths"].tolist()
            counts = data["counts"].tolist()
        else:
            raise TypeError(f"Unsupported bbox count container: {type(data)}")
    finally:
        if hasattr(data, "close"):
            data.close()
    return list(zip(paths, counts))


def load_pair_bbox_counts(path, expected_length):
    """Load per-image bbox counts for pairs in an unchanged LoFTR scene index."""
    counts = np.load(path, allow_pickle=False)
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 2:
        raise ValueError(f"Pair bbox counts must have shape [N, 2]: {path}; got {counts.shape}")
    if len(counts) != expected_length:
        raise ValueError(
            f"Pair bbox count length mismatch for {path}: "
            f"expected {expected_length}, got {len(counts)}"
        )
    if not np.issubdtype(counts.dtype, np.integer):
        raise TypeError(f"Pair bbox counts must use an integer dtype: {path}")
    return counts


@dataclass(frozen=True)
class BBoxCountLookup:
    counts: Mapping[str, int] | Sequence[tuple[str, int]]
    bases: Sequence[Path] = ()

    def __post_init__(self):
        object.__setattr__(self, "_absolute_bases", tuple(Path(base).absolute() for base in self.bases))
        object.__setattr__(self, "_resolved_bases", tuple(Path(base).resolve() for base in self.bases))
        object.__setattr__(self, "_path_to_count", self._build_index())

    def _build_index(self):
        mapping = {}
        basename_mapping = {}
        ambiguous_basenames = set()
        if isinstance(self.counts, Mapping):
            items = self.counts.items()
        else:
            items = self.counts

        for raw_path, count in items:
            path = Path(raw_path)
            count = int(count)
            mapping[path.as_posix()] = count
            if path.is_absolute():
                mapping[path.resolve().as_posix()] = count

            previous = basename_mapping.get(path.name)
            if previous is None:
                basename_mapping[path.name] = count
            elif previous != count:
                ambiguous_basenames.add(path.name)

        for basename in ambiguous_basenames:
            basename_mapping.pop(basename, None)
        object.__setattr__(self, "_basename_to_count", basename_mapping)
        return mapping

    def __call__(self, path):
        path = Path(path)
        candidates = [path.as_posix()]
        absolute_path = path.absolute()
        candidates.append(absolute_path.as_posix())
        for base in self._absolute_bases:
            try:
                rel = absolute_path.relative_to(base)
            except Exception:
                continue
            candidates.append(rel.as_posix())

        resolved_path = path.resolve()
        candidates.append(resolved_path.as_posix())
        for base in self._resolved_bases:
            try:
                rel = resolved_path.relative_to(base)
            except Exception:
                continue
            candidates.append(rel.as_posix())
            candidates.append((base / rel).as_posix())

        for candidate in candidates:
            if candidate in self._path_to_count:
                return self._path_to_count[candidate]
        return self._basename_to_count.get(path.name)
