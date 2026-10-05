from __future__ import annotations

from sgad.config.model_variants import ModelVariant, resolve_model_variant

from .region_matcher import AreaDescriptorMatcher


class SGAD(AreaDescriptorMatcher):
    """Published SGAD region matcher."""

    def __init__(self, config, *args, **kwargs):
        variant = resolve_model_variant(config)
        if variant is not ModelVariant.SGAD:
            raise ValueError(f"SGAD requires variant='sgad', got {variant.value!r}")
        super().__init__(config, *args, **kwargs)


class SGADPlusPlus(AreaDescriptorMatcher):
    """SGAD++ region matcher shared by the BCE and soft-focal objectives."""

    def __init__(self, config, *args, **kwargs):
        variant = resolve_model_variant(config)
        if variant not in {
            ModelVariant.SGAD_PLUS_PLUS_BCE,
            ModelVariant.SGAD_PLUS_PLUS_FOCAL,
        }:
            raise ValueError(f"SGADPlusPlus requires an SGAD++ variant, got {variant.value!r}")
        super().__init__(config, *args, **kwargs)


def build_sgad_model(config, *args, **kwargs) -> AreaDescriptorMatcher:
    variant = resolve_model_variant(config)
    model_cls = SGAD if variant is ModelVariant.SGAD else SGADPlusPlus
    return model_cls(config, *args, **kwargs)
