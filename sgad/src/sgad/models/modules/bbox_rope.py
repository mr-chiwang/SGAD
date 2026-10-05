# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement. See THIRD_PARTY_LICENSES at the repository root.
# Adapted from RoMa V2's DINOv3 RoPE implementation for continuous bbox-center coordinates.

import math
from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class BBoxRoPE(nn.Module):
    """Rotary position embedding for continuous bbox-center coordinates.

    Differences from standard RoPE:
    1. Supports continuous floating-point coordinates (not discrete grid positions)
    2. Computes embeddings per bbox (not pre-computed for entire grid)
    3. Supports dynamic number of bboxes

    Based on RoMa V2's RopePositionEmbedding but adapted for bbox centers.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """
        Args:
            embed_dim: Total embedding dimension (must be divisible by 4 * num_heads)
            num_heads: Number of attention heads
            base: Base for frequency calculation.
            min_period, max_period: Alternative to base for specifying frequency range
            shift_coords: Training augmentation - shift range for coordinates
            jitter_coords: Training augmentation - per-axis jitter factor
            rescale_coords: Training augmentation - global rescale factor
            dtype: Data type for computations
            device: Device for buffers
        """
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0, \
            f"embed_dim ({embed_dim}) must be divisible by 4 * num_heads ({4 * num_heads})"

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, bbox_centers: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute RoPE sin/cos for bbox center coordinates.

        Args:
            bbox_centers: [B, N, 2] - Normalized bbox center coordinates (x, y)
                         Expected range: [0, 1] (will be converted to [-1, +1])

        Returns:
            sin, cos: Tuple of [B, N, D_head] tensors for applying RoPE
        """
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else bbox_centers.dtype
        dd = {"device": device, "dtype": dtype}

        B, N, _ = bbox_centers.shape

        # Map normalized coordinates from [0, 1] to [-1, 1].
        coords = bbox_centers.to(**dd)
        coords = 2.0 * coords - 1.0  # [B, N, 2]

        # Apply coordinate augmentation during training.
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords = coords + shift_hw[None, None, :]  # [B, N, 2]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords = coords * jitter_hw[None, None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords = coords * rescale_hw

        # Compute angles: angle = 2π * coord / period
        # coords: [B, N, 2], periods: [D_head//4]
        # angles: [B, N, 2, D_head//4]
        angles = (
            2 * math.pi * coords[:, :, :, None] / self.periods[None, None, None, :]
        )  # [B, N, 2, D_head//4]

        # Flatten and tile: [B, N, D_head//2] -> [B, N, D_head]
        angles = angles.flatten(2, 3)  # [B, N, D_head//2]
        angles = angles.tile(1, 1, 2)  # [B, N, D_head]

        cos = torch.cos(angles)  # [B, N, D_head]
        sin = torch.sin(angles)  # [B, N, D_head]

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # periods = base^(2i / (D_head/2)) for i in [0, D_head/4)
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods


def rope_rotate_half(x: Tensor) -> Tensor:
    """
    Rotate half of the dimensions.
    x:   [x0, x1, x2, x3, x4, x5]
    out: [-x3, -x4, -x5, x0, x1, x2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    Apply rotary position embedding.

    Args:
        x: [B, H, N, D_head] - Query or Key tensor
        sin, cos: [B, N, D_head] - Position embeddings

    Returns:
        Rotated tensor with same shape as x
    """
    # Reshape sin/cos from [B, N, D_head] to [B, 1, N, D_head] for broadcasting
    sin = sin.unsqueeze(1)  # [B, 1, N, D_head]
    cos = cos.unsqueeze(1)  # [B, 1, N, D_head]

    return (x * cos) + (rope_rotate_half(x) * sin)


def apply_rope_to_qk(
    q: Tensor,
    k: Tensor,
    rope: Tuple[Tensor, Tensor] | None,
    prefix_len: int = 0
) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE to query and key tensors, optionally skipping prefix tokens.

    Args:
        q: [B, H, N, D_head] - Query tensor
        k: [B, H, N, D_head] - Key tensor
        rope: Tuple of (sin, cos) each [B, N, D_head], or None
        prefix_len: Number of prefix tokens to skip (e.g., CLS token)

    Returns:
        Tuple of rotated (q, k) tensors
    """
    if rope is None:
        return q, k

    sin, cos = rope
    q_dtype = q.dtype
    k_dtype = k.dtype
    rope_dtype = sin.dtype

    # Use the RoPE dtype for the rotation calculation.
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)

    if prefix_len > 0:
        # Preserve prefix tokens such as CLS.
        q_prefix = q[:, :, :prefix_len, :]
        k_prefix = k[:, :, :prefix_len, :]

        q_pos = rope_apply(q[:, :, prefix_len:, :], sin, cos)
        k_pos = rope_apply(k[:, :, prefix_len:, :], sin, cos)

        q = torch.cat((q_prefix, q_pos), dim=2)
        k = torch.cat((k_prefix, k_pos), dim=2)
    else:
        q = rope_apply(q, sin, cos)
        k = rope_apply(k, sin, cos)

    # Restore the original query and key dtypes.
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)

    return q, k
