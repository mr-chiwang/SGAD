"""Alternating self/cross-attention blocks used by SGAD++."""

from typing import Callable, Optional, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .bbox_rope import BBoxRoPE, apply_rope_to_qk


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention with RoPE support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        rope: Tuple[Tensor, Tensor],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: [B, N, D] input tensor
            rope: Tuple of (sin, cos) each [B, N, D_head] for RoPE
            attn_mask: [B, N, N] or [B, 1, N, N] attention mask (True = keep, False = mask)

        Returns:
            [B, N, D] output tensor
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, N, H, D_head]
        q = q.transpose(1, 2)  # [B, H, N, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope_to_qk(q, k, rope)

        # Scaled dot-product attention
        if attn_mask is not None:
            # Ensure attn_mask has correct shape for scaled_dot_product_attention
            # Expected: [B, 1, N, N] or [B, H, N, N]
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, N, N] -> [B, 1, N, N]

            # Convert boolean mask to float mask for scaled_dot_product_attention
            # True -> 0, False -> -inf
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float().masked_fill(~attn_mask, float('-inf')).masked_fill(attn_mask, 0.0)

            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with Pre-LN and RoPE support.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            device=device,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
    def forward(
        self,
        x: Tensor,
        rope: Tuple[Tensor, Tensor],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: [B, N, D] input tensor
            rope: Tuple of (sin, cos) for RoPE
            attn_mask: Attention mask, or None

        Returns:
            [B, N, D] output tensor
        """
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)

        x = x + self.mlp(self.norm2(x))

        return x


class SGADPlusPlusDescriptorTransformer(nn.Module):
    """
    SGAD++ alternating self/cross-attention stack with RoPE.

    Architecture:
    - All layers alternate between:
      - Self-attention within each view (with RoPE)
      - Cross-attention between views (no RoPE)
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,  # Number of (self, cross) pairs
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        rope_base: float = 100.0,
        rope_shift_coords: float | None = None,
        rope_jitter_coords: float | None = None,
        rope_rescale_coords: float | None = None,
        norm_layer: str = "layernorm",
        device=None,
    ):
        super().__init__()

        self.num_layers = num_layers

        if norm_layer == "layernorm":
            norm_layer_cls = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        # RoPE for position encoding
        self.rope = BBoxRoPE(
            embed_dim=d_model,
            num_heads=num_heads,
            base=rope_base,
            shift_coords=rope_shift_coords,
            jitter_coords=rope_jitter_coords,
            rescale_coords=rope_rescale_coords,
            device=device,
        )

        # Self-attention blocks (with RoPE)
        self.self_attn_blocks = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=nn.GELU,
                norm_layer=norm_layer_cls,
                device=device,
            )
            for _ in range(num_layers)
        ])

        # Cross-attention blocks (no RoPE)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=nn.GELU,
                norm_layer=norm_layer_cls,
                device=device,
            )
            for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_self_attn_mask(self, mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return mask[:, :, None] & mask[:, None, :]

    def _create_cross_attn_mask(
        self,
        q_mask: Tensor | None,
        kv_mask: Tensor | None,
        Q_len: int,
        KV_len: int,
        device: torch.device,
    ) -> Tensor | None:
        """Create cross-attention mask where q attends to kv."""
        if q_mask is None and kv_mask is None:
            return None

        B = q_mask.shape[0] if q_mask is not None else kv_mask.shape[0]

        if q_mask is None:
            q_mask = torch.ones(B, Q_len, dtype=torch.bool, device=device)
        if kv_mask is None:
            kv_mask = torch.ones(B, KV_len, dtype=torch.bool, device=device)

        # mask[i, j] = True means query i can attend to key j
        attn_mask = q_mask[:, :, None] & kv_mask[:, None, :]
        return attn_mask

    def forward(
        self,
        feat0: Tensor,
        feat1: Tensor,
        coords0: Tensor,
        coords1: Tensor,
        mask0: Tensor | None = None,
        mask1: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        L = feat0.shape[1]
        S = feat1.shape[1]
        device = feat0.device

        for i in range(self.num_layers):
            # Self-attention within each view (with RoPE)
            rope0 = self.rope(coords0)
            rope1 = self.rope(coords1)

            self_mask0 = self._create_self_attn_mask(mask0)
            self_mask1 = self._create_self_attn_mask(mask1)

            feat0 = self.self_attn_blocks[i](feat0, rope=rope0, attn_mask=self_mask0)
            feat1 = self.self_attn_blocks[i](feat1, rope=rope1, attn_mask=self_mask1)

            # Cross-attention between views (no RoPE)
            cross_mask_0to1 = self._create_cross_attn_mask(mask0, mask1, L, S, device)
            cross_mask_1to0 = self._create_cross_attn_mask(mask1, mask0, S, L, device)

            feat0_new = self.cross_attn_blocks[i](feat0, feat1, attn_mask=cross_mask_0to1)
            feat1_new = self.cross_attn_blocks[i](feat1, feat0, attn_mask=cross_mask_1to0)

            feat0 = feat0_new
            feat1 = feat1_new

        return feat0, feat1


class CrossAttention(nn.Module):
    """Cross-attention module where query attends to key-value from different source."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q_input: Tensor,
        kv_input: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            q_input: [B, N_q, D] query input
            kv_input: [B, N_kv, D] key-value input
            attn_mask: [B, N_q, N_kv] attention mask (True = keep)

        Returns:
            [B, N_q, D] output
        """
        B, N_q, D = q_input.shape
        _, N_kv, _ = kv_input.shape

        q = self.q_proj(q_input).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, N_q, N_kv] -> [B, 1, N_q, N_kv]

            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float().masked_fill(~attn_mask, float('-inf')).masked_fill(attn_mask, 0.0)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        x = x.transpose(1, 2).reshape(B, N_q, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
    ) -> None:
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            device=device,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
    def forward(
        self,
        q_input: Tensor,
        kv_input: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            q_input: [B, N_q, D] query input
            kv_input: [B, N_kv, D] key-value input
            attn_mask: [B, N_q, N_kv] attention mask

        Returns:
            [B, N_q, D] output (only updates query)
        """
        x = q_input + self.attn(
            self.norm1_q(q_input), self.norm1_kv(kv_input), attn_mask=attn_mask
        )

        x = x + self.mlp(self.norm2(x))

        return x
