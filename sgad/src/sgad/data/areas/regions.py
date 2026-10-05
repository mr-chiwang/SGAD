from typing import List, Tuple

import torch
import torch.nn.functional as F

from .geometry import corners_to_xyxy, xywh_to_xyxy
from .normalization import normalize_imagenet_images


def bbox_transforms(
        regions: torch.Tensor,
        image: torch.Tensor,
        target_width: int,
        target_height: int,
        region_format: str,
        is_normalize=True):
    """Extract, resize, pad, and optionally normalize image regions.

    Args:
        regions: Bounding boxes or polygon corners.
        image: Image tensor with shape ``[C, H, W]``.
        target_width: Output crop width.
        target_height: Output crop height.
        region_format: One of ``xywh``, ``xyxy``, or ``corners``.
        is_normalize: Apply ImageNet mean and standard-deviation normalization.

    Returns:
        A tensor of resized and padded region crops.
    """
    # Handle empty regions
    if regions.numel() == 0 or len(regions) == 0:
        C = image.shape[0]
        return torch.empty(
            (0, C, target_height, target_width),
            device=image.device,
            dtype=image.dtype,
        )

    image = image.float()
    target_size = (target_height, target_width)

    # Convert the requested regions into rectangular image crops.
    extracted_regions = extract_regions(image, regions, region_format)

    return torch.stack([
        resize_and_pad_region(region, target_size, is_normalize)
        for region in extracted_regions
    ])


def extract_regions(image: torch.Tensor,
                    regions: torch.Tensor,
                    region_format: str) -> List[torch.Tensor]:
    """Extract rectangular image crops for the requested regions.

    Args:
        image: Image tensor with shape ``[C, H, W]``.
        regions: Bounding boxes or polygon corners in ``region_format``.
        region_format: One of ``xywh``, ``xyxy``, or ``corners``.

    Returns:
        A list of rectangular image crops.
    """
    C, H, W = image.shape
    if region_format == 'xywh':
        boxes = xywh_to_xyxy(regions)
    elif region_format == 'xyxy':
        boxes = regions
    elif region_format == 'corners':
        boxes = corners_to_xyxy(regions)
    else:
        raise ValueError("Unsupported region format. Use 'xywh', 'xyxy', or 'corners'.")

    # Convert and clamp all region coordinates before slicing image crops.
    x1, y1, x2, y2 = boxes.t().int()
    x1, x2 = x1.clamp(0, W), x2.clamp(0, W)
    y1, y2 = y1.clamp(0, H), y2.clamp(0, H)

    extracted_regions = []

    for i in range(len(boxes)):
        cropped_region = image[:, y1[i]:y2[i], x1[i]:x2[i]]
        extracted_regions.append(cropped_region)

    return extracted_regions


def resize_and_pad_region(
        region: torch.Tensor,
        target_size: Tuple[int, int],
        is_normalize=True) -> torch.Tensor:
    """Resize one crop with preserved aspect ratio and bottom-right padding.

    Args:
        region: Extracted crop with shape ``[C, H, W]``.
        target_size: Output ``(height, width)``.
        is_normalize: Apply ImageNet mean and standard-deviation normalization.

    Returns:
        The resized and padded crop.
    """
    c, h, w = region.shape
    target_h, target_w = target_size

    # Preserve aspect ratio while fitting within the target size.
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the crop.
    resized_region = F.interpolate(region.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

    # Pad the right and bottom edges to the target size.
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom

    # Apply zero padding.
    padded_region = F.pad(resized_region, padding)

    if is_normalize:
        # Apply the normalization used by the pretrained image backbone.
        padded_region = normalize_imagenet_images(padded_region)

    return padded_region
