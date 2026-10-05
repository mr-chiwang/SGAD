import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from ``(x, y, width, height)`` to corner coordinates."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def corners_to_xyxy(polygons: torch.Tensor) -> torch.Tensor:
    """Convert polygon vertices to axis-aligned corner coordinates."""
    min_xy = torch.min(polygons, dim=1)[0]
    max_xy = torch.max(polygons, dim=1)[0]
    return torch.cat([min_xy, max_xy], dim=-1)


def calculate_polygon_centroid(points):
    """Calculate polygon centroids from vertex coordinates.

    Args:
        points: Polygon vertices with shape ``[N, V, 2]``.

    Returns:
        Polygon centroids with shape ``[N, 2]``.
    """
    shifted = points - points[:, 0:1, :]

    areas = torch.zeros(points.shape[0], device=points.device)
    for i in range(1, points.shape[1] - 1):
        areas += 0.5 * (shifted[:, i, 0] * shifted[:, i+1, 1] - shifted[:, i+1, 0] * shifted[:, i, 1])

    centroid_x = torch.zeros(points.shape[0], device=points.device)
    centroid_y = torch.zeros(points.shape[0], device=points.device)

    for i in range(1, points.shape[1]):
        factor = (shifted[:, i-1, 0] * shifted[:, i, 1] - shifted[:, i, 0] * shifted[:, i-1, 1])
        centroid_x += (shifted[:, i-1, 0] + shifted[:, i, 0]) * factor
        centroid_y += (shifted[:, i-1, 1] + shifted[:, i, 1]) * factor

    centroid_x = centroid_x / (6 * areas)
    centroid_y = centroid_y / (6 * areas)

    centroid_x += points[:, 0, 0]
    centroid_y += points[:, 0, 1]

    return torch.stack([centroid_x, centroid_y], dim=-1)


def calculate_patch_centers(bboxes, image_size, format='xywh'):
    """Return region centroids normalized by image width and height."""
    # Preserve the expected output shape for an empty region set.
    if bboxes.numel() == 0 or len(bboxes) == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=bboxes.device)

    if format == 'xywh':
        centers = torch.stack([
            (bboxes[:, 0] + bboxes[:, 2] / 2) / image_size[1],
            (bboxes[:, 1] + bboxes[:, 3] / 2) / image_size[0]
        ], dim=-1)
    elif format == 'corners':
        centers = calculate_polygon_centroid(bboxes)
        centers[:, 0] /= image_size[1]
        centers[:, 1] /= image_size[0]
    else:
        raise ValueError("Unsupported bbox format. Use 'xywh' or 'corners'.")

    return centers
