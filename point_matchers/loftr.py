
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any, List, Optional
from loguru import logger
import random

import inspect

from .abstract_point_matcher import AbstractPointMatcher
from .LoFTR.src.utils.plotting import make_matching_figure
from .LoFTR.src.loftr import LoFTR, default_cfg
from .LoFTR.src.config.default import get_cfg_defaults
from .LoFTR.src.utils.misc import lower_config


class LoFTRMatcher(AbstractPointMatcher):
    """Adapt LoFTR to the common point-matcher interface."""

    def __init__(
        self, 
        config_path: str,
        dataset_name: str,
        weights: str,
        cross_domain: bool = False,
        ):
        super().__init__()
        self._name = "LoFTRMatcher"
        _default_cfg = get_cfg_defaults()

        if dataset_name == "ScanNet" or dataset_name == "Matterport3D" or dataset_name == "KITTI" or dataset_name == "ETH3D":
            if cross_domain:
                main_cfg_path = f"{config_path}/loftr/outdoor/buggy_pos_enc/loftr_ds.py"
            else:
                main_cfg_path = f"{config_path}/loftr/indoor/scannet/loftr_ds_eval_new.py"

            data_config = f"{config_path}/data/scannet_test_1500.py"
        elif dataset_name == "MegaDepth" or dataset_name == "YFCC":
            if cross_domain:
                main_cfg_path = f"{config_path}/loftr/indoor/scannet/loftr_ds_eval_new.py"
            else:
                main_cfg_path = f"{config_path}/loftr/outdoor/buggy_pos_enc/loftr_ds.py"

            data_config = f"{config_path}/data/megadepth_test_1500.py"
        else:
            raise NotImplementedError(f"dataset_name {dataset_name} not implemented")
        

        # Official LoFTR configurations import the top-level src and configs packages.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "LoFTR"))
        _default_cfg.merge_from_file(main_cfg_path)
        _default_cfg.merge_from_file(data_config)
        _default_cfg = lower_config(_default_cfg)

        # _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
        
        self._default_cfg = _default_cfg['loftr']

        torch.cuda.set_device(0)
        matcher = LoFTR(config=_default_cfg['loftr'])
        torch_load_sig = inspect.signature(torch.load)
        if "weights_only" in torch_load_sig.parameters:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        else:
            ckpt = torch.load(weights, map_location="cpu")

        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        matcher.load_state_dict(state_dict)
        self.matcher = matcher.eval().cuda()

        # LoFTR's third-party configuration imports a top-level package named
        # ``src``. Remove that module after initialization so it cannot shadow
        # another project's package with the same name.
        self._cleanup_third_party_src_namespace()

    @staticmethod
    def _cleanup_third_party_src_namespace() -> None:
        src_mod = sys.modules.get("src", None)
        if src_mod is None:
            return

        src_file = getattr(src_mod, "__file__", None)
        if not src_file:
            return

        try:
            src_file_path = Path(src_file).resolve()
        except Exception:
            return

        loftr_root = (Path(__file__).resolve().parent / "LoFTR").resolve()
        loftr_src_init = (loftr_root / "src" / "__init__.py").resolve()

        if src_file_path != loftr_src_init:
            return

        for name in list(sys.modules.keys()):
            if name == "src" or name.startswith("src."):
                sys.modules.pop(name, None)

        loftr_root_str = str(loftr_root)
        sys.path[:] = [p for p in sys.path if p != loftr_root_str]

    def match(self, img0, img1, mask0=None, mask1=None):
        """Match a pair of grayscale or BGR images."""
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img_tensor0 = torch.from_numpy(img0)[None][None].cuda() / 255.
        img_tensor1 = torch.from_numpy(img1)[None][None].cuda() / 255.

        batch = {"image0": img_tensor0, "image1": img_tensor1}

        if mask0 is not None and mask1 is not None:
            mask0 = torch.from_numpy(mask0).cuda()
            mask1 = torch.from_numpy(mask1).cuda()
            [ts_mask_0, ts_mask_1] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=0.125,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool().to("cuda")
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})

        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy() # Nx2
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            m_bids = batch['m_bids'].cpu().numpy()
            mask = m_bids == 0 # only one batch
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
        
        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        
        self.corrs = self.matched_corrs

        logger.info(f"matched corrs num is {len(self.matched_corrs)}")

        return self.matched_corrs

    def get_coarse_mkpts_c(self, area0, area1):
        """ match region and only use coarse level to get coarse mkpts
        """
        assert area0.shape == area1.shape

        if len(area0.shape) == 3:
            area0 = cv2.cvtColor(area0, cv2.COLOR_BGR2GRAY)
            area1 = cv2.cvtColor(area1, cv2.COLOR_BGR2GRAY)
        
        area0_tensor = torch.from_numpy(area0 / 255.)[None][None].cuda().float()
        area1_tensor = torch.from_numpy(area1 / 255.)[None][None].cuda().float()

        batch = {"image0": area0_tensor, "image1": area1_tensor}

        with torch.no_grad():
            self.matcher.coarse_match_mkpts_c(batch)

        conf_matrix = batch["conf_matrix"]
        mkpts0_c = batch["mkpts0_c"]
        mkpts1_c = batch["mkpts1_c"]
        mconf = batch["mconf"]

        return mkpts0_c, mkpts1_c, mconf, conf_matrix
