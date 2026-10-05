from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sgad.config.model_variants import (
    ALTERNATING_ATTENTION_MLP_PE,
    ALTERNATING_ATTENTION_ROPE,
    normalize_attention_name,
)

from ..modules.sgad_plus_plus_transformer import SGADPlusPlusDescriptorTransformer
from ..modules.sgad_transformer import SGADDescriptorTransformer
from .position_encoding import PositionalEncoding


@dataclass
class DescriptorTransformerModules:
    attention_type: str
    position_encoding: nn.Module | None
    pos_scale: nn.Parameter | float | None
    region_transformer: nn.Module


def build_descriptor_transformer(config, input_dim: int) -> DescriptorTransformerModules:
    """Build the configured SGAD attention stack."""
    bbox_cfg = config["bbox"]
    attention_type = normalize_attention_name(
        bbox_cfg.get("attention_type", bbox_cfg.get("transformer_type", "loftr"))
    )
    if attention_type == ALTERNATING_ATTENTION_ROPE:
        num_layers = len(bbox_cfg.get("layer_names", ["self", "cross"] * 4)) // 2
        rope_config = bbox_cfg.get("rope", {})

        return DescriptorTransformerModules(
            attention_type=attention_type,
            position_encoding=None,
            pos_scale=None,
            region_transformer=SGADPlusPlusDescriptorTransformer(
                d_model=input_dim,
                num_heads=bbox_cfg.get("nhead", 8),
                num_layers=num_layers,
                ffn_ratio=bbox_cfg.get("ffn_ratio", 4.0),
                qkv_bias=bbox_cfg.get("qkv_bias", True),
                drop=bbox_cfg.get("dropout", 0.0),
                attn_drop=bbox_cfg.get("attn_dropout", 0.0),
                rope_base=rope_config.get("base", 100.0),
                rope_shift_coords=rope_config.get("shift_coords", None),
                rope_jitter_coords=rope_config.get("jitter_coords", None),
                rope_rescale_coords=rope_config.get("rescale_coords", None),
            ),
        )

    if attention_type != ALTERNATING_ATTENTION_MLP_PE:
        raise ValueError(f"Unsupported attention_type: {attention_type!r}")

    pos_scale = (
        nn.Parameter(torch.ones(1))
        if bbox_cfg.get("pos_scale_learnable", True)
        else 1.0
    )

    return DescriptorTransformerModules(
        attention_type=attention_type,
        position_encoding=PositionalEncoding(embed_dim=input_dim),
        pos_scale=pos_scale,
        region_transformer=SGADDescriptorTransformer(
            bbox_cfg,
            d_model=input_dim,
            nhead=8,
        ),
    )


def apply_descriptor_transformer(
    *,
    attention_type: str,
    position_encoding: nn.Module | None,
    pos_scale,
    region_transformer: nn.Module,
    bbox_features0,
    bbox_features1,
    bboxes_img_centers0,
    bboxes_img_centers1,
    mask_features0,
    mask_features1,
):
    """Apply the configured SGAD or SGAD++ descriptor transformer."""
    if attention_type == ALTERNATING_ATTENTION_ROPE:
        return region_transformer(
            bbox_features0,
            bbox_features1,
            bboxes_img_centers0,
            bboxes_img_centers1,
            mask_features0,
            mask_features1,
        )

    position_encoding0 = position_encoding(bboxes_img_centers0, mask_features0)
    position_encoding1 = position_encoding(bboxes_img_centers1, mask_features1)
    bbox_features0 = bbox_features0 + pos_scale * position_encoding0
    bbox_features1 = bbox_features1 + pos_scale * position_encoding1

    return region_transformer(
        bbox_features0,
        bbox_features1,
        mask_features0,
        mask_features1,
    )
