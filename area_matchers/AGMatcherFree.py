"""
AGMatcherF: area graph builder + learned area matcher loader.

Build `areagraph0/areagraph1` from image and bbox paths and initialize
`SgadInferenceAdapter` for learned region matching in `area_matchers/sgad.py`.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Optional

from utils.common import test_dir_if_not_create

from .AGConfig import areagraph_configs
from .AreaGrapher import AreaGraph


def _to_plain_container(value: Any) -> Any:
    """Convert OmegaConf containers (DictConfig/ListConfig) to plain Python types."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively update dict values (Hydra-style overrides)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)  # type: ignore[index]
        else:
            dst[key] = value
    return dst


class AGMatcherF:
    """Build two AreaGraphs and load the learned area matcher."""

    def __init__(self, configs: dict[str, Any]) -> None:
        self.datasetName = configs["datasetName"]
        self.level_num = configs["level_num"]
        self.level_step = configs["level_step"]
        self.W = configs["W"]
        self.H = configs["H"]
        self.show_flag = int(configs.get("show_flag", 0))

        self.hcrf = bool(configs.get("hcrf", False))
        self.hcrf_overlap_threshold = float(configs.get("hcrf_overlap_threshold", 0.9))
        self.hcrf_coverage_threshold = float(configs.get("hcrf_coverage_threshold", 0.4))
        self.weights = configs.get("weights", None)
        self.model_variant = configs.get("model_variant", None)
        self.pretrained_root = configs.get("pretrained_root", None)
        self.pred_score_threshold = configs.get("pred_score_threshold", None)

        # Optional AreaGraph/AreaPreprocessor overrides, deep-merged into defaults.
        overrides = _to_plain_container(configs.get("areagraph_overrides")) or {}
        if not isinstance(overrides, dict):
            raise TypeError(f"areagraph_overrides must be a dict, got {type(overrides)!r}")
        self.areagraph_overrides: dict[str, Any] = overrides
        self.base_out_path = configs["out_path"]
        self.out_path = self.base_out_path

        # Pair-specific state populated by path_loader().
        self.name0: Optional[str] = None
        self.name1: Optional[str] = None
        self.img_path0: Optional[str] = None
        self.img_path1: Optional[str] = None
        self.bbox_path0: Optional[str] = None
        self.bbox_path1: Optional[str] = None

        self.ag_config0: Optional[dict[str, Any]] = None
        self.ag_config1: Optional[dict[str, Any]] = None

        self.areagraph0: Optional[AreaGraph] = None
        self.areagraph1: Optional[AreaGraph] = None

        # SgadInferenceAdapter must provide `area_match(...)`.
        self.area_matcher: Optional[Any] = None

    def path_loader(
        self,
        img_path0: str,
        bbox_path0: str,
        img_path1: str,
        bbox_path1: str,
        name0: str,
        name1: str,
    ) -> None:
        self.name0 = name0
        self.name1 = name1
        self.img_path0 = img_path0
        self.img_path1 = img_path1
        self.bbox_path0 = bbox_path0
        self.bbox_path1 = bbox_path1

        self.out_path = os.path.join(self.base_out_path, f"{self.name0}_{self.name1}")
        # Avoid creating per-pair output directories when visualizations are disabled.
        # This prevents accumulating many empty folders under `res/...`.
        if self.show_flag == 1:
            test_dir_if_not_create(self.out_path)
        self._set_ag_config()

    def init_area_matcher(self) -> None:
        if self.area_matcher is not None:
            return

        from area_matchers.sgad_inference import SgadInferenceAdapter

        if self.weights is None:
            raise ValueError("The SGAD area matcher requires `weights` to be set in the area_matcher config.")

        self.area_matcher = SgadInferenceAdapter(
            weights=self.weights,
            model_variant=self.model_variant,
            hcrf=self.hcrf,
            hcrf_overlap_threshold=self.hcrf_overlap_threshold,
            hcrf_coverage_threshold=self.hcrf_coverage_threshold,
            pretrained_root=self.pretrained_root,
            pred_score_threshold=self.pred_score_threshold,
        )

    def img_areagraph_construct(self, efficient: bool = False) -> tuple[AreaGraph, AreaGraph]:
        if self.ag_config0 is None or self.ag_config1 is None:
            raise RuntimeError("Call path_loader(...) before img_areagraph_construct(...).")

        # AreaGraph builds ScanNet graphs during initialization; ``efficient``
        # selects the same construction path requested by the caller.
        self.areagraph0 = AreaGraph(self.ag_config0, efficient=efficient, dataset_name=self.datasetName)
        self.areagraph1 = AreaGraph(self.ag_config1, efficient=efficient, dataset_name=self.datasetName)
        return self.areagraph0, self.areagraph1

    def _set_ag_config(self) -> None:
        assert self.img_path0 is not None
        assert self.img_path1 is not None
        assert self.bbox_path0 is not None
        assert self.bbox_path1 is not None

        save_path0 = os.path.join(self.out_path, "area_graph0")
        save_path1 = os.path.join(self.out_path, "area_graph1")

        def _build_cfg(img_path: str, bbox_path: str, save_path: str) -> dict[str, Any]:
            cfg = deepcopy(areagraph_configs)
            if self.areagraph_overrides:
                _deep_update(cfg, deepcopy(self.areagraph_overrides))

            cfg["ori_img_path"] = img_path
            cfg["level_num"] = self.level_num
            cfg["level_step"] = self.level_step
            cfg["save_path"] = save_path
            cfg["show_flag"] = self.show_flag

            # Force resolution from hydra area_matcher.{W,H} instead of AGConfig.py defaults.
            cfg["W"] = self.W
            cfg["H"] = self.H

            preprocess = cfg.setdefault("preprocesser_config", {})
            preprocess = _to_plain_container(preprocess)
            cfg["preprocesser_config"] = preprocess
            if not isinstance(preprocess, dict):
                raise TypeError(f"areagraph_configs.preprocesser_config must be a dict, got {type(preprocess)!r}")
            cfg["bbox_path"] = bbox_path

            if self.show_flag == 1:
                test_dir_if_not_create(save_path)

            return cfg

        self.ag_config0 = _build_cfg(self.img_path0, self.bbox_path0, save_path0)
        self.ag_config1 = _build_cfg(self.img_path1, self.bbox_path1, save_path1)
