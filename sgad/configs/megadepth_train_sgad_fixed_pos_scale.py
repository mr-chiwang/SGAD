from sgad.config.families import make_megadepth_cfg
from sgad.config.model_variants import ModelVariant


cfg = make_megadepth_cfg(ModelVariant.SGAD)
cfg.MODEL.BBOX.POS_SCALE_LEARNABLE = False
