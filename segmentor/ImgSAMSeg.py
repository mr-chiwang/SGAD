
"""Command-line entry point for automatic SAM region generation."""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentor.SAMSeger import SAMSeger

os.environ.setdefault("A2PM_SGAD_ROOT", str(REPO_ROOT))

PRETRAINED_ROOT = Path(os.environ.get("A2PM_PRETRAINED_ROOT", Path(os.environ["A2PM_SGAD_ROOT"]) / "pretrained"))
DEFAULT_SAM_CKPT = os.environ.get("A2PM_SAM_CKPT", str(PRETRAINED_ROOT / "sam" / "sam_vit_h_4b8939.pth"))
SAM_CONFIG = {
    "SAM_name": "SAM",
    "sam_model_type": "vit_h",
    "sam_model_path": DEFAULT_SAM_CKPT,
    "save_folder": "",
}


def run_segmentation(args):
    """Configure SAM and generate regions for one image."""
    seg_configs = dict(SAM_CONFIG)

    if args.model_path is not None:
        seg_configs["sam_model_path"] = args.model_path

    seg_configs["save_folder"] = args.save_folder
    if args.resize_width is not None:
        seg_configs["resize"] = (args.resize_width, args.resize_height)

    os.makedirs(seg_configs["save_folder"], exist_ok=True)

    sam_seger = SAMSeger(configs=seg_configs)
    sam_seger.segment(img_path=args.img_path, save_name=args.save_name or "", save_img_flag=False)


def parse_args():
    """Parse arguments and validate the optional output-coordinate size."""
    parser = argparse.ArgumentParser(description="A script to segment the image using SAM")
    parser.add_argument("--img_path", type=str, required=True, help="The path of the image to be segmented")
    parser.add_argument("--save_folder", type=str, required=True, help="The folder to save the segmented result")
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Output basename; defaults to the input image stem",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Resize width before SAM inference; must be used with --resize-height",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Resize height before SAM inference; must be used with --resize-width",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Override checkpoint path for selected model")
    args = parser.parse_args()
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("--resize-width and --resize-height must be provided together")
    if args.resize_width is not None and (args.resize_width <= 0 or args.resize_height <= 0):
        parser.error("resize width and height must be positive")
    return args


if __name__ == "__main__":
    run_segmentation(parse_args())
