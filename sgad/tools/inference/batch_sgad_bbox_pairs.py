import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("A2PM_SGAD_ROOT", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from loguru import logger

from area_matchers.sgad_inference import SgadInferenceAdapter
from sgad.models.region_matcher.assignment import dual_matching
from sgad.data.bbox_filtering import process_bboxes_for_training
from sgad.data.bbox_io import read_bboxes_from_file


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch bbox-pair matching with SGAD on input1/input2 image pairs."
    )
    parser.add_argument(
        "--data_roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset roots that contain input1/input2 folders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to save per-pair matched bbox results (.npz).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.environ.get("A2PM_SGAD_WEIGHTS", ""),
        help="Path to an SGAD-family checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--pred_score_threshold",
        type=float,
        default=None,
        help="Matching threshold for dual_matching. If not set, uses checkpoint/default.",
    )
    parser.add_argument(
        "--bbox1_candidates",
        type=str,
        default="bboxes0,bboxes1,bboxes",
        help="Comma-separated candidate folder names for input1 bboxes.",
    )
    parser.add_argument(
        "--bbox2_candidates",
        type=str,
        default="bboxes1,bboxes2,bboxes",
        help="Comma-separated candidate folder names for input2 bboxes.",
    )
    parser.add_argument("--min_bbox_area", type=float, default=4000)
    parser.add_argument("--min_bbox_wh", type=float, default=28)
    parser.add_argument("--min_bbox_count", type=int, default=3)
    parser.add_argument("--enable_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_expand", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_random_gen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_pairs", type=int, default=-1, help="Maximum number of image pairs; -1 processes all pairs.")
    return parser.parse_args()


def collect_input_pair_dirs(root: Path) -> List[Tuple[Path, Path]]:
    pair_dirs = []
    for input1_dir in sorted(root.rglob("input1")):
        if not input1_dir.is_dir():
            continue
        input2_dir = input1_dir.parent / "input2"
        if input2_dir.is_dir():
            pair_dirs.append((input1_dir, input2_dir))
    return pair_dirs


def build_stem_to_path_map(folder: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix not in IMG_EXTS:
            continue
        if p.name.startswith("."):
            continue
        key = p.stem.lower()
        if key not in mapping:
            mapping[key] = p
    return mapping


def infer_bbox_base(image_path: Path, candidates: Iterable[str]) -> Path | None:
    pair_dir = image_path.parent.parent
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        base = pair_dir / cand / image_path.stem
        if base.with_suffix(".npy").exists() or base.with_suffix(".json").exists():
            return base
    return None


def preprocess_megadepth_style(image_bgr: np.ndarray, matcher: SgadInferenceAdapter) -> torch.Tensor:
    """Convert a BGR image to the SGAD model's RGB ``[0, 1]`` input format."""
    device = next(matcher.matcher.parameters()).device
    image_rgb = image_bgr[..., ::-1].copy()
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return image


def match_pair_with_scores(
    matcher: SgadInferenceAdapter,
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    bboxes0_xywh: torch.Tensor,
    bboxes1_xywh: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(matcher.matcher.parameters()).device
    image0 = preprocess_megadepth_style(image0_bgr, matcher)
    image1 = preprocess_megadepth_style(image1_bgr, matcher)

    bboxes0_tensor = bboxes0_xywh.to(device=device, dtype=torch.float32)
    bboxes1_tensor = bboxes1_xywh.to(device=device, dtype=torch.float32)

    if bboxes0_tensor.numel() == 0 or bboxes1_tensor.numel() == 0:
        empty_idx = np.empty((0,), dtype=np.int64)
        empty_bbox = np.empty((0, 4), dtype=np.float32)
        empty_score = np.empty((0,), dtype=np.float32)
        return empty_bbox, empty_bbox, empty_score, empty_idx, empty_idx

    h0, w0 = image0_bgr.shape[:2]
    h1, w1 = image1_bgr.shape[:2]
    data = {
        "image0": image0,
        "image1": image1,
        "bboxes0": [bboxes0_tensor],
        "bboxes1": [bboxes1_tensor],
        "bboxes0_format": "xywh",
        "bboxes1_format": "xywh",
        "image_shape0": [(h0, w0)],
        "image_shape1": [(h1, w1)],
        "overlap_matrix": [torch.zeros((bboxes0_tensor.shape[0], bboxes1_tensor.shape[0]), device=device)],
    }

    with torch.no_grad():
        matcher.matcher(data)

    conf_0to1 = data["conf_matrix_0to1"]
    conf_1to0 = data["conf_matrix_1to0"]
    batch_indices0, batch_indices1 = dual_matching(
        conf_0to1,
        conf_1to0,
        threshold=matcher.pred_score_threshold,
    )

    idx0 = np.asarray(batch_indices0[0], dtype=np.int64)
    idx1 = np.asarray(batch_indices1[0], dtype=np.int64)

    if idx0.size == 0:
        empty_bbox = np.empty((0, 4), dtype=np.float32)
        empty_score = np.empty((0,), dtype=np.float32)
        return empty_bbox, empty_bbox, empty_score, idx0, idx1

    idx0_t = torch.from_numpy(idx0).to(device)
    idx1_t = torch.from_numpy(idx1).to(device)
    scores = conf_0to1[0, idx0_t, idx1_t].detach().cpu().numpy().astype(np.float32)

    matched0 = bboxes0_tensor[idx0_t].detach().cpu().numpy().astype(np.float32)
    matched1 = bboxes1_tensor[idx1_t].detach().cpu().numpy().astype(np.float32)
    return matched0, matched1, scores, idx0, idx1


def main() -> None:
    args = parse_args()

    if not args.weights:
        raise ValueError("--weights is required (or set A2PM_SGAD_WEIGHTS).")

    bbox1_candidates = [x.strip() for x in args.bbox1_candidates.split(",") if x.strip()]
    bbox2_candidates = [x.strip() for x in args.bbox2_candidates.split(",") if x.strip()]

    matcher = SgadInferenceAdapter(
        weights=args.weights,
        pred_score_threshold=args.pred_score_threshold,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    saved_count = 0
    skipped_count = 0

    for root_str in args.data_roots:
        root = Path(root_str)
        if not root.exists():
            logger.warning(f"Skip missing root: {root}")
            continue

        pair_dirs = collect_input_pair_dirs(root)
        logger.info(f"[{root}] found {len(pair_dirs)} input1/input2 folders")

        for input1_dir, input2_dir in pair_dirs:
            map1 = build_stem_to_path_map(input1_dir)
            map2 = build_stem_to_path_map(input2_dir)
            common_stems = sorted(set(map1.keys()) & set(map2.keys()))

            rel_parent = input1_dir.parent.relative_to(root)
            save_dir = output_root / root.name / rel_parent / "bbox_match"
            save_dir.mkdir(parents=True, exist_ok=True)

            for stem in common_stems:
                if args.max_pairs >= 0 and pair_count >= args.max_pairs:
                    logger.info(f"Reached max_pairs={args.max_pairs}, stop.")
                    logger.info(f"saved={saved_count}, skipped={skipped_count}, total_seen={pair_count}")
                    return

                img0_path = map1[stem]
                img1_path = map2[stem]
                out_file = save_dir / f"{stem}.npz"
                pair_count += 1

                if args.resume and out_file.exists():
                    skipped_count += 1
                    continue

                bbox0_base = infer_bbox_base(img0_path, bbox1_candidates)
                bbox1_base = infer_bbox_base(img1_path, bbox2_candidates)
                if bbox0_base is None or bbox1_base is None:
                    logger.warning(f"Missing bbox file: {img0_path.name} / {img1_path.name}")
                    skipped_count += 1
                    continue

                image0 = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
                image1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
                if image0 is None or image1 is None:
                    logger.warning(f"Failed to read image: {img0_path} / {img1_path}")
                    skipped_count += 1
                    continue

                bboxes0 = read_bboxes_from_file(bbox0_base)
                bboxes1 = read_bboxes_from_file(bbox1_base)

                h0, w0 = image0.shape[:2]
                h1, w1 = image1.shape[:2]
                bboxes0, _ = process_bboxes_for_training(
                    bboxes0,
                    h0,
                    w0,
                    min_area=args.min_bbox_area,
                    min_wh=args.min_bbox_wh,
                    min_count=args.min_bbox_count,
                    enable_filter=args.enable_filter,
                    enable_expand=args.enable_expand,
                    enable_random_gen=args.enable_random_gen,
                )
                bboxes1, _ = process_bboxes_for_training(
                    bboxes1,
                    h1,
                    w1,
                    min_area=args.min_bbox_area,
                    min_wh=args.min_bbox_wh,
                    min_count=args.min_bbox_count,
                    enable_filter=args.enable_filter,
                    enable_expand=args.enable_expand,
                    enable_random_gen=args.enable_random_gen,
                )

                matched0, matched1, scores, idx0, idx1 = match_pair_with_scores(
                    matcher,
                    image0,
                    image1,
                    bboxes0,
                    bboxes1,
                )

                np.savez_compressed(
                    out_file,
                    bbox0_xywh=matched0,
                    bbox1_xywh=matched1,
                    confidence=scores,
                    match_idx0=idx0,
                    match_idx1=idx1,
                    image0_path=str(img0_path),
                    image1_path=str(img1_path),
                )
                saved_count += 1

    logger.info(f"Done. saved={saved_count}, skipped={skipped_count}, total_seen={pair_count}")


if __name__ == "__main__":
    main()
