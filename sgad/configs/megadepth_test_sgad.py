from sgad.config.families import make_megadepth_cfg
from sgad.config.model_variants import ModelVariant


cfg = make_megadepth_cfg(ModelVariant.SGAD, test=True)
