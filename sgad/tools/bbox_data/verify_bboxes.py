import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from list_images import list_images


def infer_bbox_save_path(image_path: Path, root_dir: Path, output_root: Path) -> Path:
    rel_path = image_path.relative_to(root_dir)
    parent = rel_path.parent
    bboxes_parent = parent.parent / "bboxes" if len(parent.parts) > 0 else Path("bboxes")
    rel_bbox = bboxes_parent / rel_path.name
    rel_bbox = rel_bbox.with_suffix(".npy")
    return output_root / rel_bbox


def check_bbox_file(
    image_path: Path, bbox_path: Path, root_dir: Path, output_root: Path
) -> Tuple[bool, str]:
    if not bbox_path.exists():
        return False, "missing bbox file"

    try:
        boxes = np.load(bbox_path)
    except Exception as e:
        return False, f"load failed: {e}"

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return False, f"bad shape {boxes.shape}"
    if not np.issubdtype(boxes.dtype, np.number):
        return False, f"bad dtype {boxes.dtype}"
    if not np.isfinite(boxes).all():
        return False, "nan or inf detected"

    img = Image.open(image_path)
    w, h = img.size

    xs, ys, ws, hs = boxes.T
    if (ws <= 0).any() or (hs <= 0).any():
        return False, "non-positive width/height"
    if (xs < 0).any() or (ys < 0).any():
        return False, "negative x or y"
    if ((xs + ws) > (w + 1)).any() or ((ys + hs) > (h + 1)).any():
        return False, "box exceeds image size"
    return True, ""


def resolve_output_root(root_dir: Path, output_root: Path = None) -> Path:
    if output_root:
        return output_root.resolve()
    parts = list(root_dir.resolve().parts)
    if len(parts) == 0:
        raise ValueError("root_dir is invalid.")
    parts[-1] = parts[-1] + "_bboxes"
    return Path(*parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated bbox npy files.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Original image root used when the bbox files were generated.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Bbox output root. Defaults to a sibling *_bboxes directory.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Check only the first N images. The default checks all images.",
    )
    parser.add_argument(
        "--scan_workers",
        type=int,
        default=32,
        help="Number of threads used to scan the image list; use 1 for a single thread.",
    )
    parser.add_argument(
        "--viz_dir",
        type=Path,
        default=None,
        help="Optional visualization directory for images with bbox overlays.",
    )
    parser.add_argument(
        "--viz_samples",
        type=int,
        default=20,
        help="Maximum number of visualizations to save when --viz_dir is set.",
    )
    return parser.parse_args()


def draw_bboxes(image_path: Path, boxes: np.ndarray, save_root: Path, root_dir: Path):
    if boxes.size == 0:
        return
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, y, w, h in boxes:
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=2)

    rel_path = image_path.relative_to(root_dir)
    save_path = (save_root / rel_path).with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def main():
    args = parse_args()
    root_dir = args.root_dir.resolve()
    output_root = resolve_output_root(root_dir, args.output_root)
    viz_root = args.viz_dir.resolve() if args.viz_dir is not None else None

    images = list_images(
        root_dir,
        max_images=args.max_files or 0,
        scan_workers=args.scan_workers,
        enable_tqdm=True,
    )
    total = len(images)
    print(f"found images: {total}")

    bad: List[Tuple[Path, str]] = []
    missing = 0
    viz_saved = 0
    try:
        from tqdm import tqdm

        iterator = tqdm(images, desc="validate")
    except Exception:
        iterator = images

    for img_path in iterator:
        bbox_path = infer_bbox_save_path(img_path, root_dir, output_root)
        ok, reason = check_bbox_file(img_path, bbox_path, root_dir, output_root)
        if not ok:
            if reason == "missing bbox file":
                missing += 1
            bad.append((bbox_path, reason))
        elif viz_root is not None and viz_saved < args.viz_samples:
            boxes = np.load(bbox_path)
            draw_bboxes(img_path, boxes, viz_root, root_dir)
            viz_saved += 1

    print(f"checked: {total}, missing: {missing}, invalid: {len(bad)}")
    if viz_root is not None:
        print(f"visualizations saved: {viz_saved} -> {viz_root}")
    if bad:
        print("examples of issues (up to 10):")
        for p, reason in bad[:10]:
            print(f"- {p}: {reason}")
        exit(1)


if __name__ == "__main__":
    main()
