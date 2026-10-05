from sgad.config.families import make_scannet_cfg
from sgad.config.model_variants import ModelVariant


cfg = make_scannet_cfg(ModelVariant.SGAD)
cfg.MODEL.BBOX.POS_SCALE_LEARNABLE = False
