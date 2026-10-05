"""Batch wrapper for scene-organized SAM preprocessing datasets."""

import os
import subprocess
import sys
from pathlib import Path
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch segment images with ImgSAMSeg.py")
    parser.add_argument("--input_folder", type=str, required=True, help="Dataset root containing scene folders")
    parser.add_argument("--output_base_folder", type=str, required=True, help="Output base folder")
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument(
        "--imgsamseg_py",
        type=str,
        default=str(Path(__file__).resolve().parent / "ImgSAMSeg.py"),
        help="Path to ImgSAMSeg.py",
    )
    parser.add_argument(
        "--img_subdir",
        type=str,
        default="images",
        help="Subfolder name containing images under each scene folder",
    )
    args = parser.parse_args()
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("--resize-width and --resize-height must be provided together")
    if args.resize_width is not None and (args.resize_width <= 0 or args.resize_height <= 0):
        parser.error("resize width and height must be positive")
    return args


def main() -> None:
    args = _parse_args()

    for scene_folder in os.listdir(args.input_folder):
        scene_path = os.path.join(args.input_folder, scene_folder, args.img_subdir)

        if not os.path.isdir(scene_path):
            continue

        for file in os.listdir(scene_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue

            img_path = os.path.join(scene_path, file)

            save_folder = os.path.join(args.output_base_folder, scene_folder)
            os.makedirs(save_folder, exist_ok=True)

            file_name, _ = os.path.splitext(file)
            save_name = f"{file_name}.npy"
            save_path = os.path.join(save_folder, save_name)

            cmd = [
                sys.executable,
                args.imgsamseg_py,
                "--img_path",
                img_path,
                "--save_folder",
                save_folder,
                "--save_name",
                save_name,
            ]
            if args.resize_width is not None:
                cmd.extend([
                    "--resize-width",
                    str(args.resize_width),
                    "--resize-height",
                    str(args.resize_height),
                ])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
            subprocess.run(cmd, env=env, check=False)

            print(f"Processed {img_path} -> {save_path}")


if __name__ == "__main__":
    main()
