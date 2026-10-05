from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ModelVariant(str, Enum):
    SGAD = "sgad"
    SGAD_PLUS_PLUS_BCE = "sgad_plus_plus_bce"
    SGAD_PLUS_PLUS_FOCAL = "sgad_plus_plus_focal"


DEFAULT_MODEL_VARIANT = ModelVariant.SGAD_PLUS_PLUS_BCE

ALTERNATING_ATTENTION_MLP_PE = "alternating_attention_mlp_pe"
ALTERNATING_ATTENTION_ROPE = "alternating_attention_rope"

_LEGACY_ATTENTION_NAMES = {
    "loftr": ALTERNATING_ATTENTION_MLP_PE,
    "loftr_rope": ALTERNATING_ATTENTION_ROPE,
}


@dataclass(frozen=True)
class ModelVariantSpec:
    variant: ModelVariant
    model_family: str
    backbone: str
    attention: str
    matching_objective: str
    ranking_objective: str


MODEL_VARIANT_SPECS = {
    ModelVariant.SGAD: ModelVariantSpec(
        variant=ModelVariant.SGAD,
        model_family="sgad",
        backbone="dinov2",
        attention=ALTERNATING_ATTENTION_MLP_PE,
        matching_objective="hard_focal",
        ranking_objective="listmle",
    ),
    ModelVariant.SGAD_PLUS_PLUS_BCE: ModelVariantSpec(
        variant=ModelVariant.SGAD_PLUS_PLUS_BCE,
        model_family="sgad_plus_plus",
        backbone="dinov3",
        attention=ALTERNATING_ATTENTION_ROPE,
        matching_objective="soft_bce",
        ranking_objective="listmle",
    ),
    ModelVariant.SGAD_PLUS_PLUS_FOCAL: ModelVariantSpec(
        variant=ModelVariant.SGAD_PLUS_PLUS_FOCAL,
        model_family="sgad_plus_plus",
        backbone="dinov3",
        attention=ALTERNATING_ATTENTION_ROPE,
        matching_objective="soft_focal",
        ranking_objective="listmle",
    ),
}


def normalize_attention_name(value: str) -> str:
    return _LEGACY_ATTENTION_NAMES.get(value, value)


def coerce_model_variant(value: str | ModelVariant | None) -> ModelVariant:
    if value in (None, "", "sgad_plus_plus"):
        return DEFAULT_MODEL_VARIANT
    if isinstance(value, ModelVariant):
        return value
    return ModelVariant(value)


def get_model_variant_spec(value: str | ModelVariant | None) -> ModelVariantSpec:
    return MODEL_VARIANT_SPECS[coerce_model_variant(value)]


def apply_model_variant(cfg, value: str | ModelVariant | None):
    """Apply one complete public SGAD model identity to a YACS config."""
    spec = get_model_variant_spec(value)
    was_frozen = bool(hasattr(cfg, "is_frozen") and cfg.is_frozen())
    allowed_new_keys = bool(hasattr(cfg, "is_new_allowed") and cfg.is_new_allowed())
    if was_frozen:
        cfg.defrost()
    if hasattr(cfg, "set_new_allowed"):
        cfg.set_new_allowed(True)

    try:
        cfg.MODEL.NAME = spec.model_family
        cfg.MODEL.VARIANT = spec.variant.value
        cfg.MODEL.BACKBONE_TYPE = spec.backbone
        cfg.MODEL.BBOX.ATTENTION_TYPE = spec.attention
        if spec.variant is ModelVariant.SGAD:
            cfg.MODEL.BBOX.FROM_IMG = False
            cfg.MODEL.LOSS.USE_SOFT_OVERLAP_LOSS = False
        else:
            cfg.MODEL.BBOX.FROM_IMG = True
            cfg.MODEL.LOSS.USE_SOFT_OVERLAP_LOSS = True
            cfg.MODEL.LOSS.SOFT_OVERLAP_LOSS_TYPE = (
                "bce" if spec.variant is ModelVariant.SGAD_PLUS_PLUS_BCE else "focal_soft"
            )
    finally:
        if hasattr(cfg, "set_new_allowed"):
            cfg.set_new_allowed(allowed_new_keys)
        if was_frozen:
            cfg.freeze()

    return cfg


def _get(mapping: Mapping[str, Any], key: str, default=None):
    return mapping.get(key, mapping.get(key.upper(), default))


def resolve_model_variant(config: Mapping[str, Any]) -> ModelVariant:
    """Resolve explicit model metadata, falling back to legacy checkpoint fields."""
    model_cfg = _get(config, "model", config)
    explicit = _get(model_cfg, "variant")
    if explicit:
        return coerce_model_variant(explicit)

    backbone = _get(model_cfg, "backbone_type", "dinov2")
    bbox_cfg = _get(model_cfg, "bbox", {})
    attention = normalize_attention_name(
        _get(bbox_cfg, "attention_type", _get(bbox_cfg, "transformer_type", "loftr"))
    )
    loss_cfg = _get(model_cfg, "loss", {})
    use_soft = bool(_get(loss_cfg, "use_soft_overlap_loss", False))
    soft_type = _get(loss_cfg, "soft_overlap_loss_type", "bce")

    if backbone == "dinov3" and attention == ALTERNATING_ATTENTION_ROPE and use_soft:
        if soft_type == "focal_soft":
            return ModelVariant.SGAD_PLUS_PLUS_FOCAL
        if soft_type == "bce":
            return ModelVariant.SGAD_PLUS_PLUS_BCE

    return ModelVariant.SGAD
