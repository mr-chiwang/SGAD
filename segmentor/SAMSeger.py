"""Automatic SAM bounding-box generation and serialization utilities."""

import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from segmentor.extract_bboxes import sam_records_to_bboxes

SAM_ROOT = Path(__file__).resolve().parent / "SAM"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SAMSeger(object):
    """Generate and save automatic SAM region boxes for one image at a time.

    An optional ``resize`` tuple controls the image coordinate space exposed to
    SAM and therefore the coordinate space of saved bounding boxes.
    SAM's own encoder preprocessing remains internal to the upstream model.
    """

    def __init__(self, configs) -> None:
        self.resize = configs.get("resize")
        if self.resize is not None:
            width, height = self.resize
            if width <= 0 or height <= 0:
                raise ValueError("resize width and height must be positive")

        self.W = self.resize[0] if self.resize is not None else None
        self.H = self.resize[1] if self.resize is not None else None
        self.SAM_name = configs["SAM_name"]
        if self.SAM_name != "SAM":
            raise ValueError("SAMSeger only supports the original SAM implementation.")
        self.sam_model_type = configs["sam_model_type"]
        self.sam_model_path = configs["sam_model_path"]
        self.save_folder = configs["save_folder"]
        self.sam_model = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
        self.sam_model.to(device='cuda')
        self.sam_mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
        )

    def img_loader_SAM(self, path):
        """Load an image and optionally resize its external coordinate space."""
        image = cv2.imread(path, -1)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        original_shape = image.shape
        if self.resize is not None:
            image = cv2.resize(image, self.resize)

        self.H, self.W = image.shape[:2]
        logger.info(
            f"Loaded image {path}: original_shape={original_shape}, "
            f"inference_shape={image.shape}, dtype={image.dtype}"
        )
        return image

    def _extract_bbox_info(self, masks):
        """Extract the numeric bbox format consumed by SGAD and A2PM.

        Args:
            masks: SAM automatic-mask records containing ``segmentation``.

        Returns:
            A C-contiguous ``(N, 4)`` float32 array in XYWH format.
        """
        return sam_records_to_bboxes(masks)

    def segment(self, img_path, sort_flag=True, save_flag=True, save_img_flag=False, save_name=""):
        """Run automatic mask generation and optionally save the result.

        Args:
            img_path: Input image path.
            sort_flag: Sort regions by descending pixel area when true.
            save_flag: Save the generated records as a NumPy file when true.
            save_img_flag: Request visualization output. Visualization is
                currently disabled and produces a warning instead.
            save_name: Output basename. The input stem is used when omitted.

        Returns:
            An ``(N, 4)`` float32 array containing XYWH boxes in the image
            space produced by :meth:`img_loader_SAM`.
        """
        img = self.img_loader_SAM(img_path)

        import time
        start_time = time.time()
        masks = self.sam_mask_generator.generate(img)
        end_time = time.time()
        logger.info(f"SAM segmentation time: {end_time - start_time:.4f} seconds")

        if sort_flag:
            masks.sort(key=lambda region: region["area"], reverse=True)
        bboxes = self._extract_bbox_info(masks)

        if save_flag:
            if save_name == "":
                save_name = osp.splitext(osp.basename(img_path))[0]
            save_full_name = osp.join(self.save_folder, save_name)
            if not osp.exists(self.save_folder):
                logger.info(f"create folder {self.save_folder}")
                os.makedirs(self.save_folder)

            np.save(save_full_name, bboxes, allow_pickle=False)
            logger.info(f"Saved bounding-box regions to {save_full_name}")
        
        logger.info(f"Generated {len(bboxes)} bounding-box regions for {img_path}")

        if save_img_flag:
            logger.warning("save_img_flag is enabled but region visualization is disabled")

        return bboxes

    def draw_bbox(self, bbox_masks, save_name="bbox_visualization"):
        """Draw bounding boxes on a blank canvas matching the inference image.

        Args:
            bbox_masks: Numeric ``(N, 4)`` array whose rows use XYWH format.
            save_name: Basename for the saved visualization.
        """
        if self.W is None or self.H is None:
            raise RuntimeError("An image must be loaded before drawing bounding boxes.")

        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        for bbox in bbox_masks:
            x, y, w, h = bbox

            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

        save_path = osp.join(self.save_folder, f"{save_name}.jpg")
        cv2.imwrite(save_path, img)
        logger.info(f"Saved bbox visualization to {save_path}")
