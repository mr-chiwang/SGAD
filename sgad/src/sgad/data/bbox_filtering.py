import math

import numpy as np
import torch


def filter_bboxes_by_area_and_wh(bboxes, min_area, min_wh):
    if bboxes.numel() == 0:
        empty = bboxes.new_empty((0, 4))
        return empty, empty

    areas = bboxes[:, 2] * bboxes[:, 3]
    valid_mask = (areas >= min_area) & (bboxes[:, 2] >= min_wh) & (bboxes[:, 3] >= min_wh)
    return bboxes[valid_mask], bboxes[~valid_mask]


def expand_bbox(bbox, min_area, min_wh, image_h, image_w, max_scale=3.0):
    x, y, w, h = bbox.tolist()

    if w <= 0 or h <= 0:
        return None

    current_area = w * h
    area_scale = math.sqrt(min_area / current_area) if current_area < min_area else 1.0
    w_scale = min_wh / w if w < min_wh else 1.0
    h_scale = min_wh / h if h < min_wh else 1.0
    scale = max(area_scale, w_scale, h_scale)

    if scale <= 1.0:
        return bbox.clone()
    if scale > max_scale:
        return None

    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    new_x = max(0.0, cx - new_w / 2)
    new_y = max(0.0, cy - new_h / 2)
    new_w = min(new_w, image_w - new_x)
    new_h = min(new_h, image_h - new_y)

    if new_w < min_wh or new_h < min_wh or new_w * new_h < min_area:
        return None

    return bbox.new_tensor([new_x, new_y, new_w, new_h])


def compute_max_iou(bbox, bboxes):
    if bboxes.numel() == 0:
        return 0.0

    x1, y1, w1, h1 = bbox.tolist()
    box1 = bbox.new_tensor([x1, y1, x1 + w1, y1 + h1])

    x2 = bboxes[:, 0]
    y2 = bboxes[:, 1]
    x2_max = x2 + bboxes[:, 2]
    y2_max = y2 + bboxes[:, 3]

    inter_x1 = torch.maximum(box1[0], x2)
    inter_y1 = torch.maximum(box1[1], y2)
    inter_x2 = torch.minimum(box1[2], x2_max)
    inter_y2 = torch.minimum(box1[3], y2_max)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = bboxes[:, 2] * bboxes[:, 3]
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou.max().item()


def generate_random_bbox(image_h, image_w, min_area, min_wh, existing_bboxes=None, max_iou=0.5):
    min_size = max(min_wh, math.sqrt(min_area))
    max_size = min(image_h, image_w) * 0.5

    for _ in range(50):
        w = np.random.uniform(min_size, max(min_size + 1, max_size))
        h = np.random.uniform(min_size, max(min_size + 1, max_size))

        if w * h < min_area:
            scale = math.sqrt(min_area / (w * h))
            w *= scale
            h *= scale

        max_x = max(0, image_w - w)
        max_y = max(0, image_h - h)
        x = np.random.uniform(0, max_x) if max_x > 0 else 0
        y = np.random.uniform(0, max_y) if max_y > 0 else 0

        w = min(w, image_w - x)
        h = min(h, image_h - y)
        if w < min_wh or h < min_wh or w * h < min_area:
            continue

        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
        if existing_bboxes is not None and existing_bboxes.numel() > 0:
            if compute_max_iou(bbox, existing_bboxes) > max_iou:
                continue
        return bbox

    w = min(min_size * 1.5, image_w)
    h = min(min_size * 1.5, image_h)
    x = (image_w - w) / 2
    y = (image_h - h) / 2
    return torch.tensor([x, y, w, h], dtype=torch.float32)


def process_bboxes_for_training(
    bboxes,
    image_h,
    image_w,
    min_area,
    min_wh,
    min_count=3,
    max_expand_scale=3.0,
    enable_filter=True,
    enable_expand=True,
    enable_random_gen=True,
):
    if bboxes.numel() == 0:
        bboxes = torch.empty((0, 4), dtype=torch.float32)

    if enable_filter:
        filtered_bboxes, rejected_bboxes = filter_bboxes_by_area_and_wh(bboxes, min_area, min_wh)
    else:
        filtered_bboxes = bboxes
        rejected_bboxes = torch.empty((0, 4), dtype=bboxes.dtype, device=bboxes.device)

    result_bboxes = list(filtered_bboxes)
    bbox_sources = ["original"] * len(result_bboxes)

    if enable_expand and len(result_bboxes) < min_count and rejected_bboxes.numel() > 0:
        areas = rejected_bboxes[:, 2] * rejected_bboxes[:, 3]
        sorted_indices = torch.argsort(areas, descending=True)
        sorted_rejected = rejected_bboxes[sorted_indices]

        for bbox in sorted_rejected:
            if len(result_bboxes) >= min_count:
                break
            expanded = expand_bbox(bbox, min_area, min_wh, image_h, image_w, max_expand_scale)
            if expanded is not None:
                result_bboxes.append(expanded)
                bbox_sources.append("expanded")

    if enable_random_gen:
        existing_bboxes = torch.stack(result_bboxes) if result_bboxes else None
        while len(result_bboxes) < min_count:
            random_bbox = generate_random_bbox(
                image_h,
                image_w,
                min_area,
                min_wh,
                existing_bboxes,
                max_iou=0.5,
            )
            result_bboxes.append(random_bbox)
            bbox_sources.append("random")
            existing_bboxes = torch.stack(result_bboxes)

    if result_bboxes:
        processed_bboxes = torch.stack(result_bboxes)
    else:
        processed_bboxes = torch.empty((0, 4), dtype=torch.float32)

    return processed_bboxes, bbox_sources
