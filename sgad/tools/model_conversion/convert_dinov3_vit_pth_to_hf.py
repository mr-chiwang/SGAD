#!/usr/bin/env python3
"""Convert the official DINOv3 ViT-L/16 .pth checkpoint to HF format.

Create a local Transformers model directory from the official .pth checkpoint
for SGAD++'s AutoModel.from_pretrained(..., local_files_only=True) loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.image_processing_dinov3_vit_fast import (
    DINOv3ViTImageProcessorFast,
)
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel


def build_vitl16_config() -> DINOv3ViTConfig:
    return DINOv3ViTConfig(
        image_size=224,
        patch_size=16,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layerscale_value=1.0,
        use_gated_mlp=False,
    )


def convert_state_dict(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target: dict[str, torch.Tensor] = {
        "embeddings.cls_token": source["cls_token"],
        "embeddings.mask_token": source["mask_token"].unsqueeze(0),
        "embeddings.register_tokens": source["storage_tokens"],
        "embeddings.patch_embeddings.weight": source["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": source["patch_embed.proj.bias"],
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
    }

    for idx in range(24):
        src_prefix = f"blocks.{idx}"
        dst_prefix = f"layer.{idx}"

        for norm_name in ("norm1", "norm2"):
            target[f"{dst_prefix}.{norm_name}.weight"] = source[f"{src_prefix}.{norm_name}.weight"]
            target[f"{dst_prefix}.{norm_name}.bias"] = source[f"{src_prefix}.{norm_name}.bias"]

        qkv_weight = source[f"{src_prefix}.attn.qkv.weight"]
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        target[f"{dst_prefix}.attention.q_proj.weight"] = q_weight
        target[f"{dst_prefix}.attention.k_proj.weight"] = k_weight
        target[f"{dst_prefix}.attention.v_proj.weight"] = v_weight

        qkv_bias = source[f"{src_prefix}.attn.qkv.bias"]
        q_bias, _k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        target[f"{dst_prefix}.attention.q_proj.bias"] = q_bias
        target[f"{dst_prefix}.attention.v_proj.bias"] = v_bias

        target[f"{dst_prefix}.attention.o_proj.weight"] = source[f"{src_prefix}.attn.proj.weight"]
        target[f"{dst_prefix}.attention.o_proj.bias"] = source[f"{src_prefix}.attn.proj.bias"]

        target[f"{dst_prefix}.layer_scale1.lambda1"] = source[f"{src_prefix}.ls1.gamma"]
        target[f"{dst_prefix}.layer_scale2.lambda1"] = source[f"{src_prefix}.ls2.gamma"]

        target[f"{dst_prefix}.mlp.up_proj.weight"] = source[f"{src_prefix}.mlp.fc1.weight"]
        target[f"{dst_prefix}.mlp.up_proj.bias"] = source[f"{src_prefix}.mlp.fc1.bias"]
        target[f"{dst_prefix}.mlp.down_proj.weight"] = source[f"{src_prefix}.mlp.fc2.weight"]
        target[f"{dst_prefix}.mlp.down_proj.bias"] = source[f"{src_prefix}.mlp.fc2.bias"]

    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to dinov3_vitl16_pretrain_lvd1689m-*.pth")
    parser.add_argument("--dst", required=True, help="Output Hugging Face local model directory")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    source = torch.load(src, map_location="cpu")
    if not isinstance(source, dict):
        raise TypeError(f"Expected dict-like checkpoint, got {type(source)!r}")

    config = build_vitl16_config()
    model = DINOv3ViTModel(config)
    converted = convert_state_dict(source)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}")

    model.save_pretrained(dst, safe_serialization=True)
    DINOv3ViTImageProcessorFast().save_pretrained(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
