from .dinov2 import Dinov2Extractor
from .dinov3 import Dinov3Extractor


def build_backbone(config):
    """Build the configured visual feature extractor.

    Args:
        config: Model configuration containing ``backbone_type`` and the
            corresponding backbone options.

    Returns:
        The configured DINO feature extractor.

    Supported backbone types:
        - dinov2: DINOv2 feature extractor
        - dinov3: DINOv3 feature extractor (HuggingFace)
    """

    if config["backbone_type"] == "dinov2":
        return Dinov2Extractor(config["dinov2"])

    if config["backbone_type"] == "dinov3":
        return Dinov3Extractor(config["dinov3"])

    raise NotImplementedError(f"Unsupported backbone_type: {config['backbone_type']}")
