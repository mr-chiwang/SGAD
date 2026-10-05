from __future__ import annotations

from typing import Any

import torch
from torchvision.ops import roi_align

from sgad.data.areas.normalization import normalize_imagenet_images


def pool_spatial_features(features: torch.Tensor) -> torch.Tensor:
    """Pool DINO token features to one descriptor per bbox crop."""
    if features.dim() != 3:
        raise ValueError(f"Expected DINO token features [B, N, D], got {features.shape}")
    return features.mean(dim=1)


def roi_pool_features(
    spatial_features: torch.Tensor,
    bboxes,
    image_size,
) -> torch.Tensor:
    """Vectorized ROI pooling from DINOv3 patch tokens using roi_align."""
    orig_H, orig_W = image_size
    device = spatial_features.device
    dtype = spatial_features.dtype

    if spatial_features.dim() != 3:
        raise ValueError(
            f"Expected DINOv3 patch tokens [B, N, D], got {spatial_features.shape}"
        )

    B, _num_patches, D = spatial_features.shape
    H_feat = orig_H // 16
    W_feat = orig_W // 16
    patch_features = spatial_features.reshape(B, H_feat, W_feat, D).permute(0, 3, 1, 2)

    scale_x = W_feat / orig_W
    scale_y = H_feat / orig_H

    all_boxes_list = []
    total_boxes = 0

    for batch_idx in range(B):
        batch_bboxes = bboxes[batch_idx]
        num_boxes = len(batch_bboxes) if batch_bboxes is not None and len(batch_bboxes) > 0 else 0

        if num_boxes > 0:
            if isinstance(batch_bboxes, torch.Tensor):
                boxes_tensor = batch_bboxes.clone().to(device=device, dtype=dtype)
            else:
                boxes_tensor = torch.tensor(batch_bboxes, device=device, dtype=dtype)

            x1 = boxes_tensor[:, 0]
            y1 = boxes_tensor[:, 1]
            x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2]
            y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3]

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            x1_scaled = x1_scaled.clamp(min=0, max=W_feat)
            y1_scaled = y1_scaled.clamp(min=0, max=H_feat)
            x2_scaled = x2_scaled.clamp(min=0, max=W_feat)
            y2_scaled = y2_scaled.clamp(min=0, max=H_feat)

            x2_scaled = torch.max(x2_scaled, x1_scaled + 1)
            y2_scaled = torch.max(y2_scaled, y1_scaled + 1)

            batch_indices = torch.full((num_boxes, 1), batch_idx, device=device, dtype=dtype)
            boxes_with_idx = torch.stack(
                [
                    batch_indices.squeeze(1),
                    x1_scaled,
                    y1_scaled,
                    x2_scaled,
                    y2_scaled,
                ],
                dim=1,
            )

            all_boxes_list.append(boxes_with_idx)
            total_boxes += num_boxes

    if total_boxes == 0:
        return torch.empty(0, D, device=device, dtype=dtype)

    all_boxes = torch.cat(all_boxes_list, dim=0)
    pooled = roi_align(
        patch_features,
        all_boxes,
        output_size=1,
        spatial_scale=1.0,
        aligned=True,
    )
    return pooled.squeeze(-1).squeeze(-1)


def extract_bbox_features(backbone: Any, bbox_images: torch.Tensor, *, batch_size: int = 1500) -> torch.Tensor:
    num_bboxes = bbox_images.size(0)

    if num_bboxes <= batch_size:
        spatial_features = backbone(bbox_images)
        return pool_spatial_features(spatial_features)

    feature_list = []
    for i in range(0, num_bboxes, batch_size):
        batch = bbox_images[i : i + batch_size]
        spatial_features = backbone(batch)
        pooled_features = pool_spatial_features(spatial_features)
        feature_list.append(pooled_features)

    return torch.cat(feature_list, dim=0)


def extract_features_from_image(
    backbone: Any,
    images: torch.Tensor,
    bboxes,
) -> torch.Tensor:
    normalized_images = normalize_imagenet_images(images)
    spatial_features = backbone(normalized_images)
    image_size = (images.shape[2], images.shape[3])
    return roi_pool_features(spatial_features, bboxes, image_size)
