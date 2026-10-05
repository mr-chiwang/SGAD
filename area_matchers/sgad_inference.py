"""SGAD-family region-matcher adapter for the A2PM pipeline.

The adapter loads models trained by ``sgad`` and converts bounding boxes
between the A2PM ``[x_min, x_max, y_min, y_max]`` convention and the model's
``[x, y, width, height]`` convention.
"""

import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Keep the in-repository ``src`` package importable when A2PM is run directly
# from a clone without installing the SGAD subproject in editable mode.
SGAD_SRC = Path(__file__).resolve().parents[1] / "sgad" / "src"
if str(SGAD_SRC) not in sys.path:
    sys.path.insert(0, str(SGAD_SRC))

from sgad.config.default import get_cfg_defaults
from sgad.config.model_variants import apply_model_variant
from sgad.utils.misc import lower_config
from sgad.models import build_sgad_model
from sgad.models.checkpoints import infer_checkpoint_model_variant, load_sgad_checkpoint
from sgad.models.region_matcher.assignment import dual_matching

from area_matchers.filte_contain import filter_matched_regions


class SgadInferenceAdapter:
    """Expose an SGAD-family model through the A2PM region-matcher interface."""

    def __init__(
        self,
        weights: str,
        model_variant: str | None = None,
        config_path: str = None,
        hcrf: bool = False,
        hcrf_overlap_threshold: float = 0.9,
        hcrf_coverage_threshold: float = 0.4,
        pretrained_root: str | None = None,
        pred_score_threshold: float | None = None,
        img_resize: int = 224,
    ) -> None:
        """Initialize the matcher and load its checkpoint.

        Args:
            weights: Path to a model checkpoint.
            model_variant: Explicit SGAD-family model variant. When omitted, the
                variant is inferred from checkpoint metadata when available.
            config_path: Optional YACS configuration file. Checkpoint metadata or
                the package defaults are used when this is omitted.
            hcrf: Whether to apply HCRF containment filtering to matched regions.
            hcrf_overlap_threshold: Minimum child coverage in both images for containment.
            hcrf_coverage_threshold: Minimum final coverage for replacing a parent.
            pretrained_root: Optional root directory containing local backbones.
            pred_score_threshold: Minimum confidence accepted by dual matching.
            img_resize: Spatial input size used by the region-feature backbone.
        """
        super().__init__()
        self._name = "SgadInferenceAdapter"

        ckpt = None
        if weights:
            # PyTorch 2.6 defaults to weights_only=True, which rejects objects such
            # as YACS CfgNode stored in historical Lightning checkpoints.
            torch_load_sig = inspect.signature(torch.load)
            if "weights_only" in torch_load_sig.parameters:
                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            else:
                ckpt = torch.load(weights, map_location="cpu")

        # Prefer an explicit config, then checkpoint metadata, then package defaults.
        if config_path:
            config_node = get_cfg_defaults()
            config_node.merge_from_file(config_path)
        elif isinstance(ckpt, dict) and "hyper_parameters" in ckpt and "config" in ckpt["hyper_parameters"]:
            config_node = ckpt["hyper_parameters"]["config"]
        else:
            config_node = get_cfg_defaults()

        declared_variant = model_variant
        if declared_variant is None and isinstance(ckpt, dict):
            inferred_variant = infer_checkpoint_model_variant(ckpt)
            declared_variant = inferred_variant.value if inferred_variant is not None else None
        if declared_variant is not None:
            apply_model_variant(config_node, declared_variant)

        self._override_backbone_paths_from_env(config_node, pretrained_root=pretrained_root)
        _config = lower_config(config_node)

        self.hcrf = hcrf
        self.hcrf_overlap_threshold = hcrf_overlap_threshold
        self.hcrf_coverage_threshold = hcrf_coverage_threshold
        self.img_resize = img_resize
        if pred_score_threshold is None:
            pred_score_threshold = (
                ckpt.get("hyper_parameters", {}).get("pred_score_threshold", 0.2)
                if isinstance(ckpt, dict)
                else 0.2
            )
        self.pred_score_threshold = pred_score_threshold

        # Build the SGAD family model declared by checkpoint/config metadata.
        matcher = build_sgad_model(
            config=_config['model'],
            img_resize=self.img_resize,
        )

        # Portable checkpoints load strictly; historical Lightning checkpoints
        # use the isolated compatibility path in sgad.models.checkpoints.
        if weights and ckpt is not None:
            report = load_sgad_checkpoint(matcher, ckpt)
            if report.mismatched_keys:
                logger.warning(
                    f"Skipped {len(report.mismatched_keys)} mismatched keys when loading "
                    f"legacy checkpoint '{weights}': {', '.join(report.mismatched_keys[:5])}"
                )
            logger.success(
                f"Loaded {report.model_variant.value} weights: "
                f"{report.loaded_keys}/{report.model_keys} keys from '{weights}' "
                f"(strict={report.strict}, skipped={report.skipped_keys}, "
                f"mismatched={len(report.mismatched_keys)})"
            )

        self.matcher = matcher.eval().cuda()

    def _override_backbone_paths_from_env(self, config_node, pretrained_root: str | None = None) -> None:
        """Override checkpoint backbone directories using runtime path settings.

        Checkpoints may include absolute paths in their YACS configuration.
        Update those paths for local model files while preserving the model
        architecture and other configuration values.
        """
        pretrained_root = pretrained_root or os.environ.get("A2PM_PRETRAINED_ROOT")
        overrides = {
            "DINOV2": os.environ.get("A2PM_DINOV2_PATH")
            or (str(Path(pretrained_root) / "dinov2_large") if pretrained_root else None),
            "DINOV3": os.environ.get("A2PM_DINOV3_PATH")
            or (
                str(Path(pretrained_root) / "dinov3-vitl16-pretrain-lvd1689m")
                if pretrained_root
                else None
            ),
        }
        overrides = {name: path for name, path in overrides.items() if path}
        if not overrides:
            return

        was_frozen = False
        if hasattr(config_node, "is_frozen") and config_node.is_frozen():
            was_frozen = True
            config_node.defrost()

        try:
            for backbone_name, override in overrides.items():
                try:
                    backbone_cfg = getattr(config_node.MODEL, backbone_name)
                except Exception:
                    continue

                old_path = str(getattr(backbone_cfg, "PATH", ""))
                if old_path == override:
                    continue
                backbone_cfg.PATH = override
                logger.info(f"Override {backbone_name} path: {old_path} -> {override}")
        finally:
            if was_frozen:
                config_node.freeze()

    def _convert_a2pm_bbox_to_xywh(self, area_bbox):
        """Convert an A2PM bounding box to ``[x, y, width, height]``.

        Args:
            area_bbox: Bounding box in ``[x_min, x_max, y_min, y_max]`` format.

        Returns:
            Bounding box in ``[x, y, width, height]`` format.
        """
        x_min, x_max, y_min, y_max = area_bbox
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def _convert_xywh_to_a2pm_bbox(self, xywh):
        """Convert an ``[x, y, width, height]`` box to A2PM format.

        Args:
            xywh: Bounding box in ``[x, y, width, height]`` format.

        Returns:
            Bounding box in ``[x_min, x_max, y_min, y_max]`` format.
        """
        x, y, w, h = xywh
        return [x, x + w, y, y + h]

    def area_match(self, area_info_list0, area_info_list1, ori_img0, ori_img1):
        """Match regions using the interface expected by the A2PM pipeline.

        Args:
            area_info_list0: Regions from the first image. Each item contains
                ``area_bbox`` and ``area_center``.
            area_info_list1: Regions from the second image in the same format.
            ori_img0: First BGR image as an ``[height, width, channels]`` array.
            ori_img1: Second BGR image as an ``[height, width, channels]`` array.

        Returns:
            A tuple containing matched A2PM boxes for both images, model matching
            time, and HCRF post-processing time.
        """

        # Extract bounding boxes in the A2PM coordinate convention.
        area_bboxes0_a2pm = np.array([area['area_bbox'] for area in area_info_list0])
        area_bboxes1_a2pm = np.array([area['area_bbox'] for area in area_info_list1])

        # Convert boxes to the format expected by the SGAD model.
        bboxes0_xywh = np.array([self._convert_a2pm_bbox_to_xywh(bbox) for bbox in area_bboxes0_a2pm])
        bboxes1_xywh = np.array([self._convert_a2pm_bbox_to_xywh(bbox) for bbox in area_bboxes1_a2pm])

        # Extract region centers and normalize them by image dimensions.
        area_centers0 = np.array([area['area_center'] for area in area_info_list0])
        area_centers1 = np.array([area['area_center'] for area in area_info_list1])

        h0, w0 = ori_img0.shape[:2]
        h1, w1 = ori_img1.shape[:2]

        # Normalize region-center coordinates by image dimensions.
        area_centers0_norm = area_centers0 / np.array([w0, h0])
        area_centers1_norm = area_centers1 / np.array([w1, h1])

        # Prepare model inputs on the active inference device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert BGR images to RGB tensors in the model's [0, 1] input range.
        ori_img0_rgb = ori_img0[..., ::-1].copy()
        ori_img1_rgb = ori_img1[..., ::-1].copy()

        image0 = torch.from_numpy(ori_img0_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image1 = torch.from_numpy(ori_img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image0 = image0.to(device)
        image1 = image1.to(device)

        bboxes0_tensor = torch.tensor(bboxes0_xywh, dtype=torch.float32, device=device)
        bboxes1_tensor = torch.tensor(bboxes1_xywh, dtype=torch.float32, device=device)

        data = {
            'image0': image0,
            'image1': image1,
            'bboxes0': [bboxes0_tensor],
            'bboxes1': [bboxes1_tensor],
            'bboxes0_format': 'xywh',
            'bboxes1_format': 'xywh',
            'image_shape0': [(h0, w0)],
            'image_shape1': [(h1, w1)],
            # The model still expects the training-only overlap matrix during
            # inference, so provide a zero placeholder with the required shape.
            'overlap_matrix': [torch.zeros((bboxes0_tensor.shape[0], bboxes1_tensor.shape[0]), device=device)],
        }

        # Run region matching without constructing an autograd graph.
        time_matcher_start = time.time()
        with torch.no_grad():
            self.matcher(data)

        # Apply bidirectional assignment to the two confidence matrices.
        conf_matrix_0to1 = data['conf_matrix_0to1']
        conf_matrix_1to0 = data['conf_matrix_1to0']

        # Convert confidence matrices into paired region indices.
        batch_indices_area0, batch_indices_area1 = dual_matching(
            conf_matrix_0to1, conf_matrix_1to0,
            threshold=self.pred_score_threshold
        )

        time_matcher_end = time.time()
        area_match_time = time_matcher_end - time_matcher_start

        # The A2PM adapter processes one image pair per call.
        matched_indices0 = batch_indices_area0[0]
        matched_indices1 = batch_indices_area1[0]

        # Return the original boxes, which are already in A2PM format.
        matched_area_bboxes0 = area_bboxes0_a2pm[matched_indices0]
        matched_area_bboxes1 = area_bboxes1_a2pm[matched_indices1]

        merge_contain_time = 0

        # Use one shared selection with final-coverage checks in both images.
        if self.hcrf and len(matched_area_bboxes0) > 0:
            time_graph_start = time.time()

            keep_indices = filter_matched_regions(
                matched_area_bboxes0,
                matched_area_bboxes1,
                overlap_threshold=self.hcrf_overlap_threshold,
                coverage_threshold=self.hcrf_coverage_threshold,
                image_shape0=ori_img0.shape,
                image_shape1=ori_img1.shape,
            )
            time_merge_end = time.time()

            logger.success(f"hcrf before matched area nums: {matched_area_bboxes0.shape[0]}")

            merge_contain_time = time_merge_end - time_graph_start
            matched_area_bboxes0 = matched_area_bboxes0[keep_indices]
            matched_area_bboxes1 = matched_area_bboxes1[keep_indices]

            logger.success(f"hcrf after matched area nums: {matched_area_bboxes0.shape[0]}")

        return matched_area_bboxes0, matched_area_bboxes1, area_match_time, merge_contain_time
