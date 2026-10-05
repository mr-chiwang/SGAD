import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_DIR_TOKENS = ("color", "imgs", "images")


def _scan_single_dir(dir_path: Path, stop_event: threading.Event | None, max_images: int) -> List[Path]:
    results: List[Path] = []
    for root, _, files in os.walk(dir_path, followlinks=False):
        if stop_event is not None and stop_event.is_set():
            break
        parts = root.split(os.sep)
        if not any(token in parts for token in TARGET_DIR_TOKENS):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            results.append(Path(root) / fname)
            if max_images > 0 and len(results) >= max_images:
                if stop_event is not None:
                    stop_event.set()
                return results
    return results


def list_images(
    root_dir: Path,
    *,
    enable_tqdm: bool = True,
    max_images: int = 0,
    scan_workers: int = 1,
) -> List[Path]:
    max_images = max_images or 0
    scan_workers = max(1, scan_workers)
    stop_event = threading.Event() if max_images > 0 else None

    image_paths: List[Path] = []
    for entry in root_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS and any(t in entry.parts for t in TARGET_DIR_TOKENS):
            image_paths.append(entry)
            if max_images > 0 and len(image_paths) >= max_images:
                return sorted(image_paths)

    if scan_workers == 1:
        image_paths.extend(_scan_single_dir(root_dir, stop_event, max_images))
        if max_images > 0:
            image_paths = image_paths[:max_images]
        return sorted(image_paths)

    sub_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(sub_dirs), desc="scan dirs", disable=not enable_tqdm)
    except Exception:
        pbar = None

    with ThreadPoolExecutor(max_workers=scan_workers) as ex:
        futures = {ex.submit(_scan_single_dir, d, stop_event, max_images): d for d in sub_dirs}
        for future in as_completed(futures):
            chunk = future.result()
            image_paths.extend(chunk)
            if pbar is not None:
                pbar.update(1)
            if max_images > 0 and len(image_paths) >= max_images:
                if stop_event is not None:
                    stop_event.set()
                break

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    if max_images > 0:
        image_paths = image_paths[:max_images]
    return sorted(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List images under a root directory with filters.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Image root directory to scan recursively.",
    )
    parser.add_argument(
        "--scan_workers",
        type=int,
        default=32,
        help="Number of scan threads; use 1 for a single thread.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Maximum number of images to return; 0 means no limit.",
    )
    parser.add_argument("--no_tqdm", action="store_true", help="Disable the tqdm progress bar.")
    parser.add_argument(
        "--output_list",
        type=Path,
        default=None,
        help="Optional file in which to save the image list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir: Path = args.root_dir.resolve()
    images = list_images(
        root_dir,
        enable_tqdm=not args.no_tqdm,
        max_images=args.max_images,
        scan_workers=args.scan_workers,
    )
    print(f"Total images: {len(images)}")
    if args.output_list:
        args.output_list.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_list, "w") as f:
            for p in images:
                f.write(str(p) + "\n")


if __name__ == "__main__":
    main()
