from __future__ import annotations

from dataclasses import dataclass

import torch

from sgad.data.areas.preparation import prepare_bboxes_images
from .padding import pad_aligned_features


@dataclass
class AreaInputs:
    bbox_images0: torch.Tensor | None
    bbox_images1: torch.Tensor | None
    centers0: list[torch.Tensor]
    centers1: list[torch.Tensor]


@dataclass
class AreaDescriptorBatch:
    features0: torch.Tensor
    features1: torch.Tensor
    centers0: torch.Tensor
    centers1: torch.Tensor
    mask0: torch.Tensor
    mask1: torch.Tensor


def prepare_area_inputs(
    data: dict,
    *,
    from_img: bool,
    img_resize: int,
) -> AreaInputs:
    """Prepare area crops or full-image metadata for descriptor extraction."""
    (
        bbox_images0,
        bbox_images1,
        centers0,
        centers1,
    ) = prepare_bboxes_images(
        data,
        from_img=from_img,
        target_width=img_resize,
        target_height=img_resize,
    )
    return AreaInputs(bbox_images0, bbox_images1, centers0, centers1)


def split_area_features(
    features: torch.Tensor,
    num_patches_per_batch: list[int],
) -> list[torch.Tensor]:
    """Split concatenated bbox features back to per-image lists."""
    split_features = []
    start_idx = 0
    for num_patches in num_patches_per_batch:
        end_idx = start_idx + num_patches
        split_features.append(features[start_idx:end_idx])
        start_idx = end_idx
    return split_features


def pad_area_descriptor_batch(
    features0: torch.Tensor,
    features1: torch.Tensor,
    centers0: list[torch.Tensor],
    centers1: list[torch.Tensor],
) -> AreaDescriptorBatch:
    """Pad per-image bbox features/metadata for transformer and matching heads."""
    num_patches_per_batch0 = [centers.shape[0] for centers in centers0]
    num_patches_per_batch1 = [centers.shape[0] for centers in centers1]
    features0_list = split_area_features(features0, num_patches_per_batch0)
    features1_list = split_area_features(features1, num_patches_per_batch1)

    features0, centers0, mask0 = pad_aligned_features(features0_list, centers0)
    features1, centers1, mask1 = pad_aligned_features(features1_list, centers1)

    return AreaDescriptorBatch(
        features0=features0,
        features1=features1,
        centers0=centers0,
        centers1=centers1,
        mask0=mask0,
        mask1=mask1,
    )
