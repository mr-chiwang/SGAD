from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def compute_point_bbox_membership(
    points: Optional[torch.Tensor],
    bboxes: Optional[torch.Tensor],
    bboxes_format: str = "xywh",
) -> torch.Tensor:
    """Compute point membership for axis-aligned boxes or polygons.

    Args:
        points: Point coordinates with shape ``[M, 2]``.
        bboxes: Regions with shape ``[L, 4]`` for ``xywh`` or ``[L, V, 2]``
            for ``corners``.
        bboxes_format: Either ``xywh`` or ``corners``.

    Returns:
        A Boolean matrix with shape ``[L, M]`` whose entries indicate whether
        each point lies inside each region.
    """
    device = None
    if isinstance(bboxes, torch.Tensor):
        device = bboxes.device
    if device is None and isinstance(points, torch.Tensor):
        device = points.device
    if device is None:
        device = torch.device("cpu")

    if bboxes is None or not isinstance(bboxes, torch.Tensor) or bboxes.numel() == 0:
        m = points.shape[0] if isinstance(points, torch.Tensor) and points.numel() > 0 else 0
        return torch.zeros((0, m), dtype=torch.bool, device=device)

    if points is None or not isinstance(points, torch.Tensor) or points.numel() == 0:
        return torch.zeros((int(bboxes.shape[0]), 0), dtype=torch.bool, device=device)

    if points.dim() != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [M, 2]")

    if bboxes_format == "xywh":
        if bboxes.dim() != 2 or bboxes.shape[-1] != 4:
            raise ValueError("bboxes with format='xywh' must have shape [L, 4]")

        if points.device != device:
            points = points.to(device=device)
        if bboxes.device != device:
            bboxes = bboxes.to(device=device)

        x, y, w, h = bboxes.unsqueeze(1).chunk(4, dim=-1)
        px, py = points.unsqueeze(0).chunk(2, dim=-1)
        return ((px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)).squeeze(-1)

    if bboxes_format == "corners":
        if bboxes.dim() != 3 or bboxes.shape[-1] != 2:
            raise ValueError("bboxes with format='corners' must have shape [L, V, 2]")

        points = points.to(device=device, dtype=torch.float32)
        bboxes = bboxes.to(device=device, dtype=torch.float32)

        polygon_x = bboxes[..., 0]  # [L, V]
        polygon_y = bboxes[..., 1]  # [L, V]

        x = points[:, 0][None, :, None]  # [1, M, 1]
        y = points[:, 1][None, :, None]  # [1, M, 1]

        polygon_x = polygon_x[:, None, :]  # [L, 1, V]
        polygon_y = polygon_y[:, None, :]  # [L, 1, V]
        next_x = torch.roll(polygon_x, shifts=-1, dims=-1)
        next_y = torch.roll(polygon_y, shifts=-1, dims=-1)

        eps = torch.finfo(points.dtype).eps
        denom = next_y - polygon_y
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
        x_intersect = (next_x - polygon_x) * (y - polygon_y) / denom + polygon_x

        intersect = ((polygon_y > y) != (next_y > y)) & (x < x_intersect)
        return intersect.sum(dim=-1) % 2 == 1  # [L, M]

    raise ValueError(f"Unsupported bboxes_format: {bboxes_format}")
