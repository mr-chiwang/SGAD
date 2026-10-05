import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("A2PM_SGAD_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import numpy as np
import torch
import random
import cv2
import time
from point_matchers.abstract_point_matcher import AbstractPointMatcher
from dataloader.abstract_dataloader import AbstractDataloader
from area_matchers.abstract_am import AbstractAreaMatcher
from geo_area_matchers.abstract_gam import AbstractGeoAreaMatcher
from utils.common import validate_type
from utils.geo import list_of_corrs2corr_list
from utils.plotting import _make_evaluation_figure
from point_matchers.roma import RoMaMatcher
# Fix random seeds
random.seed(66)
torch.manual_seed(66)

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def _reset_random_state(seed) -> None:
    """Reset all RNGs used by matching, sampling, and pose estimation."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)

def write_to_file(content: str, file_path: str, file_name: str) -> None:
    """Write content to a file."""
    with open(os.path.join(file_path, file_name), 'a') as f:
        f.write(content + '\n')


def _alpha_list_from_cfg(cfg: DictConfig) -> list:
    alpha_list = OmegaConf.select(cfg, "geo_area_matcher.alpha_list")
    if alpha_list is None:
        return []
    values = OmegaConf.to_container(alpha_list)
    if not isinstance(values, list):
        return []
    return values


def _flatten_corrs(corrs):
    """Flatten grouped regional matches without changing point order or values."""
    if corrs is None or len(corrs) == 0:
        return []
    first = corrs[0]
    if len(first) == 4 and np.isscalar(first[0]):
        return corrs
    return list_of_corrs2corr_list(corrs)


def _load_processed_pairs(results_dir: str, alpha_list):
    """Resume only pairs with PM and every requested alpha fully evaluated."""
    if not os.path.isdir(results_dir) or not alpha_list:
        return set()

    processed = None
    prefixes = ["pm"] + [f"a2pm-{alpha}" for alpha in alpha_list]
    for prefix in prefixes:
        for suffix in ("pose_err_names.txt", "precision_names.txt"):
            file_path = os.path.join(results_dir, f"{prefix}_{suffix}")
            try:
                with open(file_path, "r") as f:
                    names = {line.strip() for line in f if line.strip()}
            except OSError:
                return set()
            processed = names if processed is None else processed.intersection(names)
            if not processed:
                return set()
    return processed


def _iter_batch_pairs(pair_file_path: str, dataset_name: str):
    """Yield (scene_name, pair0, pair1) for supported datasets."""
    with open(pair_file_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if dataset_name == "ScanNet":
                # Support two formats:
                # 1) scene0720_00_615_660
                # 2) scene0720_00 615 660
                tokens = line.split()
                if len(tokens) == 3:
                    scene_name, pair0, pair1 = tokens[0], tokens[1], tokens[2]
                    yield scene_name, pair0, pair1
                    continue

                parts = line.split("_")
                if len(parts) < 4:
                    logger.error(f"Invalid ScanNet line format: {raw_line}")
                    continue

                scene_name = f"{parts[0]}_{parts[1]}"
                pair0, pair1 = parts[2], parts[3]
                yield scene_name, pair0, pair1
                continue

            # Default: whitespace-separated pairs (e.g., MegaDepth1500)
            tokens = line.split()
            if len(tokens) < 2:
                logger.error(f"Invalid line format: {raw_line}")
                continue
            yield None, tokens[0], tokens[1]


def _resolve_pair_file(cfg: DictConfig, dataset_name: str) -> str:
    """Use a custom pair list or the dataset's complete 1500-pair list."""
    pair_file_path = OmegaConf.select(cfg, "pair_file_path")
    if pair_file_path:
        return str(pair_file_path)
    if dataset_name == "ScanNet":
        return str(REPO_ROOT / "assets" / "pairs" / "scannet1500_pairs.txt")
    if dataset_name == "MegaDepth":
        return str(REPO_ROOT / "assets" / "pairs" / "megadepth1500_pairs.txt")
    raise NotImplementedError(f"dataset {dataset_name} not supported for batch testing")


class A2PMTester:
    def __init__(self, cfg: DictConfig):
        """Initialize models and configurations."""
        self.cfg = cfg
        
        # Initialize point matcher
        if cfg.point_matcher._target_ == 'point_matchers.roma.RoMaMatcher':
            self.pmer = RoMaMatcher(
                cfg.dataset_name, 
                cfg.point_matcher.weights,
                cfg.point_matcher.dinov2_weights,
                coarse_res=cfg.point_matcher.coarse_res,
                upsample_res=cfg.point_matcher.upsample_res
            )
        else:
            self.pmer = hydra.utils.instantiate(cfg.point_matcher)
        validate_type(self.pmer, AbstractPointMatcher)

        # Initialize area matcher
        self.amer = hydra.utils.instantiate(cfg.area_matcher)
        validate_type(self.amer, AbstractAreaMatcher)
        logger.info(f"amer: {self.amer.name()}")
        
        # Initialize geometric area matcher
        self.gamer = hydra.utils.instantiate(cfg.geo_area_matcher)
        validate_type(self.gamer, AbstractGeoAreaMatcher)
        logger.info(f"gamer: {self.gamer.name()}")
        
        # Initialize dataloader (will be set for each pair)
        self.dataloader = None
        
        # Initialize evaler (will be set for each pair)
        self.evaler = hydra.utils.instantiate(cfg.evaler)
        self.alpha_list = _alpha_list_from_cfg(cfg)

    def process_single_pair(self, pair0: str, pair1: str):
        """Process a single image pair."""
        _reset_random_state(OmegaConf.select(self.cfg, "deterministic_eval_seed"))
        # Update dataloader for current pair
        self.cfg.dataset.image_name0 = pair0
        self.cfg.dataset.image_name1 = pair1
        self.dataloader = hydra.utils.instantiate(self.cfg.dataset)
        validate_type(self.dataloader, AbstractDataloader)
        
        # Initialize evaler with current dataloader
        self.evaler.init_data_loader(
            self.dataloader,
            self.cfg.eval_from_size_W,
            self.cfg.eval_from_size_H
        )
        
        # Test point matching
        if self.cfg.dataset_name in ['ScanNet', 'KITTI']:
            img0, img1, ori_corrs = self._process_standard_dataset()
        elif self.cfg.dataset_name in ['MegaDepth', 'YFCC', 'ETH3D']:
            img0, img1, ori_corrs = self._process_padded_dataset()

            # Keep plotting/evaluation visuals in eval resolution.
            img0, img1, _, _ = self.dataloader.load_images(
                self.cfg.eval_from_size_W,
                self.cfg.eval_from_size_H,
            )
        else:
            raise NotImplementedError(f"dataset {self.cfg.dataset_name} not supported")
        
        # Test area matching
        area_matches0, area_matches1 = self.amer.area_matching(self.dataloader, self.cfg.out_path)
        logger.success(f"area matching done, area_matches len: {len(area_matches0)}")

        write_to_file(f"{len(area_matches0)}", self.cfg.out_path, "area_matches_len.txt")

        # Test geometric area matching
        self.gamer.init_gam(
            dataloader=self.dataloader,
            point_matcher=self.pmer,
            ori_corrs=ori_corrs,
            out_path=self.cfg.out_path
        )
        alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs_dict, time_areas_pm = \
            self.gamer.geo_area_matching_refine(area_matches0, area_matches1)

        if alpha_sampled_corrs_dict is None and alpha_corrs_dict is not None:
            alpha_corrs_dict = {alpha: _flatten_corrs(corrs) for alpha, corrs in alpha_corrs_dict.items()}

        logger.success(f"area PM time: {time_areas_pm:.4f}s")
        write_to_file(f"{time_areas_pm:.4f}", self.cfg.out_path, "area_pm_time.txt")

        # Run evaluations
        self._run_evaluations(
            ori_corrs,
            area_matches0,
            area_matches1,
            alpha_corrs_dict,
            alpha_inlier_idxs_dict,
            alpha_sampled_corrs_dict,
            img0,
            img1
        )

    def _process_standard_dataset(self):
        """Process datasets that don't need padding (ScanNet, KITTI)."""
        img0, img1, _, _ = self.dataloader.load_images(
            self.cfg.crop_size_W,
            self.cfg.crop_size_H
        )
        self.pmer.set_corr_num_init(self.cfg.match_num)
        time_pm_start = time.time()
        ori_corrs = self.pmer.match(img0, img1, None, None)

        ori_corrs = self.dataloader.tune_corrs_size_to_eval(
            ori_corrs,
            self.cfg.crop_size_W,
            self.cfg.crop_size_H,
            self.cfg.eval_from_size_W,
            self.cfg.eval_from_size_H
        )
        time_pm_end = time.time()
        logger.success(f"ori resolution point matching time: {time_pm_end - time_pm_start:.4f}s")

        write_to_file(f"{time_pm_end - time_pm_start:.4f}", self.cfg.out_path, "pm_time.txt")

        return img0, img1, ori_corrs

    def _process_padded_dataset(self):
        """Process datasets that need padding (MegaDepth, YFCC, ETH3D)."""
        img0, mask0, match_in_W0, match_in_H0, img1, mask1, match_in_W1, match_in_H1 = \
            self.dataloader.load_images(self.cfg.crop_size_W, self.cfg.crop_size_H, PMer=True)
        self.pmer.set_corr_num_init(self.cfg.match_num)
        time_pm_start = time.time()
        ori_corrs = self.pmer.match(img0, img1, mask0, mask1)

        ori_corrs = self.dataloader.tune_corrs_size_to_eval(
            ori_corrs,
            match_in_W0,
            match_in_H0,
            match_in_W1,
            match_in_H1
        )
        time_pm_end = time.time()
        logger.success(f"point matching time: {time_pm_end - time_pm_start:.4f}s")

        write_to_file(f"{time_pm_end - time_pm_start:.4f}", self.cfg.out_path, "pm_time.txt")

        return img0, img1, ori_corrs

    def _run_evaluations(
        self,
        ori_corrs,
        area_matches0,
        area_matches1,
        alpha_corrs_dict,
        alpha_inlier_idxs_dict,
        alpha_sampled_corrs_dict,
        img0,
        img1
    ):
        """Run all configured evaluations."""
        eval_seed = OmegaConf.select(self.cfg, "deterministic_eval_seed")
        # Test area accuracy if configured
        if self.cfg.test_area_acc:
            self.evaler.eval_area_overlap_ratio(area_matches0, area_matches1, 'am', corrs=ori_corrs)
            for alpha in alpha_inlier_idxs_dict.keys():
                inlier_areas0 = [area_matches0[i] for i in alpha_inlier_idxs_dict[alpha]]
                inlier_areas1 = [area_matches1[i] for i in alpha_inlier_idxs_dict[alpha]]
                alpha_corrs = (
                    alpha_corrs_dict[alpha]
                    if alpha_sampled_corrs_dict is None
                    else alpha_sampled_corrs_dict[alpha]
                )
                self.evaler.eval_area_overlap_ratio(
                    inlier_areas0,
                    inlier_areas1,
                    f'am+gam-{alpha}',
                    corrs=alpha_corrs,
                )

        # Test point matching accuracy if configured
        if self.cfg.test_pm_acc:
            logger.success(f"ori corrs matching accuracy from pmer {self.pmer.name()} are: ")
            _reset_random_state(eval_seed)
            self.evaler.eval_point_match(ori_corrs, 'pm', self.cfg.pm_acc_thds)
            
            if alpha_sampled_corrs_dict is None:
                for alpha, corrs in alpha_corrs_dict.items():
                    logger.success(f"for alpha: {alpha} of {self.cfg.name}, matching accuracies are: ")
                    _reset_random_state(eval_seed)
                    self.evaler.eval_point_match(corrs, f'a2pm-{alpha}', self.cfg.pm_acc_thds)
            else:
                for alpha, corrs in alpha_sampled_corrs_dict.items():
                    logger.success(f"for alpha: {alpha} of {self.cfg.name}, matching accuracies are: ")
                    _reset_random_state(eval_seed)
                    self.evaler.eval_point_match(corrs, f'a2pm-{alpha}', self.cfg.pm_acc_thds)

        # Test pose error if configured
        is_plot = False
        if self.cfg.verbose == 1:
            is_plot = True
        if self.cfg.test_pose_err:
            self._evaluate_pose_errors(
                ori_corrs,
                alpha_corrs_dict,
                alpha_sampled_corrs_dict,
                img0,
                img1,
                is_plot=is_plot
            )

    def _evaluate_pose_errors(
        self,
        ori_corrs,
        alpha_corrs_dict,
        alpha_sampled_corrs_dict,
        img0,
        img1,
        is_plot=False
    ):
        """Evaluate pose errors and generate visualizations."""
        eval_seed = OmegaConf.select(self.cfg, "deterministic_eval_seed")
        # Evaluate original correspondences
        _reset_random_state(eval_seed)
        pose_err = self.evaler.eval_pose_error(ori_corrs, 'pm')
        logger.success(f"ori corrs pose error from pmer {self.pmer.name()} are: t-{pose_err[1]:.4f}, R-{pose_err[0]:.4f}")

        _reset_random_state(eval_seed)
        epi_errs, precision = self.evaler.eval_epipolar_error_and_precision(ori_corrs, 'pm')
        
        # Save visualization
        if is_plot:
            folder = f"{self.cfg.dataset.scene_name}_{self.cfg.dataset.image_name0}_{self.cfg.dataset.image_name1}"
            save_path = os.path.join(self.cfg.out_path, folder, "baseline_corrs.png")
            figure = _make_evaluation_figure(
                self.cfg.dataset_name,
                ori_corrs,
                img0,
                img1,
                epi_errs,
                pose_err,
                path=save_path
            )

        # Evaluate refined correspondences
        if alpha_sampled_corrs_dict is None:
            for alpha, corrs in alpha_corrs_dict.items():
                self._evaluate_single_alpha(alpha, corrs, img0, img1, is_plot=is_plot)
        else:
            for alpha, corrs in alpha_sampled_corrs_dict.items():
                self._evaluate_single_alpha(alpha, corrs, img0, img1, is_plot=is_plot)

    def _evaluate_single_alpha(self, alpha, corrs, img0, img1, is_plot=False):
        """Evaluate pose errors for a single alpha value."""
        eval_seed = OmegaConf.select(self.cfg, "deterministic_eval_seed")
        _reset_random_state(eval_seed)
        pose_err = self.evaler.eval_pose_error(corrs, f'a2pm-{alpha}')
        logger.success(f"for alpha: {alpha} of {self.cfg.name}, pose errors are: t-{pose_err[1]:.4f}, R-{pose_err[0]:.4f}")

        _reset_random_state(eval_seed)
        epi_errs, precision = self.evaler.eval_epipolar_error_and_precision(corrs, f'a2pm-{alpha}')
        
        if is_plot:
            folder = f"{self.cfg.dataset.scene_name}_{self.cfg.dataset.image_name0}_{self.cfg.dataset.image_name1}"
            save_path = os.path.join(self.cfg.out_path, folder, f"corrs_{alpha}.png")
            figure = _make_evaluation_figure(
                self.cfg.dataset_name,
                corrs,
                img0,
                img1,
                epi_errs,
                pose_err,
                path=save_path
            )

@hydra.main(version_base=None, config_path="../conf")
def batch_test(cfg: DictConfig) -> None:
    """Batch evaluate pairs for ScanNet/MegaDepth without editing this file.

    Usage examples:
      - MegaDepth:
        python scripts/test.py +experiment=a2pm_sgad_egam_loftr_megadepth_640_area640_sgad verbose=0
      - ScanNet:
        python scripts/test.py +experiment=a2pm_sgad_egam_roma_scannet_640 verbose=0

    Optional overrides:
      - pair_file_path=/path/to/pairs.txt
      - max_pairs=0                 (run all pairs; evaluation configs default to 1500)
      - max_pairs=50                (quick sanity check)
      - skip_done=false
    """

    os.environ["HYDRA_FULL_ERROR"] = "1"

    dataset_name = OmegaConf.select(cfg, "dataset_name")
    if dataset_name is None:
        raise ValueError("Missing `dataset_name`. Provide an experiment config via `+experiment=...`.")

    # Default MegaDepth scene_name to avoid mandatory ??? in conf/dataset/megadepth.yaml.
    if dataset_name == "MegaDepth":
        cfg.dataset.scene_name = "md"

    if cfg.verbose == 0:
        logger.remove()
        logger.add(sys.stdout, level="SUCCESS")
    elif cfg.verbose == 1:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    else:
        raise NotImplementedError(f"verbose {cfg.verbose} not supported")

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    pair_file_path = _resolve_pair_file(cfg, dataset_name)

    max_pairs = OmegaConf.select(cfg, "max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else 0

    skip_done = OmegaConf.select(cfg, "skip_done")
    if skip_done is None:
        skip_done = True
    elif isinstance(skip_done, str):
        skip_done = skip_done.strip().lower() not in {"0", "false", "no", "off"}
    else:
        skip_done = bool(skip_done)

    tester = A2PMTester(cfg)

    results_dir = os.path.join(str(cfg.out_path), "ratios")
    os.makedirs(results_dir, exist_ok=True)

    processed_pairs = (
        _load_processed_pairs(results_dir, tester.alpha_list)
        if skip_done and cfg.test_pose_err else set()
    )
    attempted = 0

    for scene_name, pair0, pair1 in _iter_batch_pairs(pair_file_path, dataset_name):
        if scene_name is not None:
            cfg.dataset.scene_name = scene_name

        complete_pair_name = f"{cfg.dataset.scene_name}_{pair0}_{pair1}"
        logger.success(f"Processing pair: {complete_pair_name}")

        if skip_done and complete_pair_name in processed_pairs:
            logger.success(f"Pair {complete_pair_name} already processed, skipping...")
            continue

        try:
            tester.process_single_pair(pair0, pair1)

            processed_pairs.add(complete_pair_name)
            logger.success(f"Successfully processed {complete_pair_name}")
        except Exception as e:
            logger.error(f"Error processing {complete_pair_name}: {str(e)}")

        attempted += 1
        if max_pairs > 0 and attempted >= max_pairs:
            break

if __name__ == "__main__":
    batch_test()
