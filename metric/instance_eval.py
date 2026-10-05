import os
import numpy as np
from loguru import logger
import random
from typing import Dict, List, Optional, Any, Tuple

from utils.geo import (
    tune_corrs_size_diff,
    nms_for_corrs,
    compute_pose_error_simp,
    calc_area_match_performence_eff_MC,
    assert_match_reproj,
    assert_match_qFp,
    calc_areas_iou_eval,
    compute_epi_errs,
)

from utils.vis import plot_matches_with_mask_ud
from utils.plotting import _compute_conf_thresh
from utils.common import test_dir_if_not_create


class InstanceEval(object):
    """ params are loaded when using
    """

    def __init__(self,
        sample_mode,
        eval_corr_num,
        sac_mode,
        out_path,
        draw_verbose=False,
        ) -> None:
        self.eval_info = None
        self.sample_mode = sample_mode
        self.eval_corr_num = eval_corr_num
        self.sac_mode = sac_mode
        self.out_path = os.path.join(out_path, 'ratios')
        self.draw_verbose = draw_verbose
        test_dir_if_not_create(self.out_path)

    def init_data_loader(self, dataloader, eval_W, eval_H):
        """
        """
        try:
            self.eval_info = dataloader.get_eval_info(eval_W, eval_H)
            self.instance_name = f'{dataloader.scene_name}_{dataloader.image_name0}_{dataloader.image_name1}'
        except Exception as e:
            logger.error(f'Error: {e}')

    @staticmethod
    def _areas_to_xywh_np(areas: List[List[float]]) -> np.ndarray:
        """Convert A2PM area format [xmin, xmax, ymin, ymax] to xywh."""
        if areas is None or len(areas) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        areas_np = np.asarray(areas, dtype=np.float32)
        if areas_np.ndim != 2 or areas_np.shape[1] != 4:
            raise ValueError("areas should have shape [N, 4] in [xmin, xmax, ymin, ymax] format")

        xywh = np.zeros_like(areas_np, dtype=np.float32)
        xywh[:, 0] = areas_np[:, 0]
        xywh[:, 1] = areas_np[:, 2]
        xywh[:, 2] = np.maximum(areas_np[:, 1] - areas_np[:, 0], 0.0)
        xywh[:, 3] = np.maximum(areas_np[:, 3] - areas_np[:, 2], 0.0)
        return xywh

    @staticmethod
    def _compute_point_bbox_membership_np(points: np.ndarray, bboxes_xywh: np.ndarray) -> np.ndarray:
        """Compute point-in-bbox mask with shape [num_bboxes, num_points]."""
        num_boxes = bboxes_xywh.shape[0]
        num_points = points.shape[0]
        if num_boxes == 0 or num_points == 0:
            return np.zeros((num_boxes, num_points), dtype=np.float32)

        x = bboxes_xywh[:, 0:1]
        y = bboxes_xywh[:, 1:2]
        w = bboxes_xywh[:, 2:3]
        h = bboxes_xywh[:, 3:4]

        px = points[None, :, 0:1]
        py = points[None, :, 1:2]

        inside = (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)
        return inside.squeeze(-1).astype(np.float32)

    @staticmethod
    def _compute_overlap_matrix_from_point_iou(
        areas0: List[List[float]],
        areas1: List[List[float]],
        corrs: Optional[List[List[float]]],
    ) -> Optional[np.ndarray]:
        """
        Compute overlap matrix with the same point-set IoU idea used in sgad:
        overlap(i,j) = |P0(i) ∩ P1(j)| / |P0(i) ∪ P1(j)|
        where P0(i)/P1(j) are matched-point sets that fall into bbox i/j.
        """
        if corrs is None:
            return None

        bboxes0_xywh = InstanceEval._areas_to_xywh_np(areas0)
        bboxes1_xywh = InstanceEval._areas_to_xywh_np(areas1)

        corrs_np = np.asarray(corrs, dtype=np.float32)
        if corrs_np.size == 0:
            return np.zeros((len(areas0), len(areas1)), dtype=np.float32)

        if corrs_np.ndim != 2 or corrs_np.shape[1] != 4:
            raise ValueError("corrs should have shape [M, 4] in [x0, y0, x1, y1] format")

        points0 = corrs_np[:, :2]
        points1 = corrs_np[:, 2:]

        points_in_bbox0 = InstanceEval._compute_point_bbox_membership_np(points0, bboxes0_xywh)
        points_in_bbox1 = InstanceEval._compute_point_bbox_membership_np(points1, bboxes1_xywh)

        intersection = points_in_bbox0 @ points_in_bbox1.T
        union = (
            points_in_bbox0.sum(axis=1, keepdims=True)
            + points_in_bbox1.sum(axis=1, keepdims=True).T
            - intersection
        )

        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        return np.clip(overlap, 0.0, 1.0)

    @staticmethod
    def _compute_mqd_stats(match_overlaps: np.ndarray) -> Dict[str, float]:
        if len(match_overlaps) == 0:
            return {
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "Q25": float("nan"),
                "Q75": float("nan"),
                "IQR": float("nan"),
            }

        q25 = float(np.percentile(match_overlaps, 25))
        q75 = float(np.percentile(match_overlaps, 75))
        return {
            "median": float(np.median(match_overlaps)),
            "std": float(np.std(match_overlaps)) if len(match_overlaps) > 1 else 0.0,
            "min": float(np.min(match_overlaps)),
            "max": float(np.max(match_overlaps)),
            "Q25": q25,
            "Q75": q75,
            "IQR": q75 - q25,
        }

    @staticmethod
    def _format_tau_for_file(tau: float) -> str:
        return f"{tau:g}".replace(".", "p").replace("-", "m")

    def _compute_bbox_point_metrics(
        self,
        areas0: List[List[float]],
        areas1: List[List[float]],
        corrs: Optional[List[List[float]]],
        thresholds: List[float],
    ) -> Dict[str, Any]:
        num_matches = min(len(areas0), len(areas1))
        if len(areas0) != len(areas1):
            logger.warning(
                f"Area pair length mismatch for {self.instance_name}: "
                f"len(areas0)={len(areas0)}, len(areas1)={len(areas1)}. "
                f"Use first {num_matches} pairs."
            )

        overlap_matrix = self._compute_overlap_matrix_from_point_iou(areas0, areas1, corrs)
        if overlap_matrix is None:
            logger.warning(
                f"corrs is None for {self.instance_name} ({num_matches} matches), "
                "skip point-IoU metrics and write NaN."
            )
            mqd_stats = self._compute_mqd_stats(np.array([]))
            return {
                "MOM": float("nan"),
                "MP": {tau: float("nan") for tau in thresholds},
                "MQD": mqd_stats,
                "num_matches": int(num_matches),
            }

        if num_matches == 0:
            mqd_stats = self._compute_mqd_stats(np.array([]))
            return {
                "MOM": float("nan"),
                "MP": {tau: float("nan") for tau in thresholds},
                "MQD": mqd_stats,
                "num_matches": 0,
            }

        diag_idx = np.arange(num_matches)
        match_overlaps = overlap_matrix[diag_idx, diag_idx]
        mom = float(np.mean(match_overlaps))
        mp = {tau: float(np.mean(match_overlaps >= tau)) for tau in thresholds}
        mqd_stats = self._compute_mqd_stats(match_overlaps)

        return {
            "MOM": mom,
            "MP": mp,
            "MQD": mqd_stats,
            "num_matches": int(num_matches),
        }
    
    # def eval_area_overlap_ratio(self,
    #     areas0,
    #     areas1,
    #     pre_name,
    #     ):
    #     """
    #     Args:
    #         areas0/1: list of [u_min, u_max, v_min, v_max] 
    #     """
    #     assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
    #     assert self.eval_info['dataset_name'] in ['ScanNet'], 'Error: dataset not supported'

    #     if len(areas0) < 1 or len(areas1) < 1:
    #         logger.error(f'Error: areas0/1 is empty')
    #         return

    #     try:
    #         acr, aor = calc_area_match_performence_eff_MC(
    #                 areas0,
    #                 areas1,
    #                 self.eval_info['image0'],
    #                 self.eval_info['image1'],
    #                 self.eval_info['K0'],
    #                 self.eval_info['K1'],
    #                 self.eval_info['P0'],
    #                 self.eval_info['P1'],
    #                 self.eval_info['depth0'],
    #                 self.eval_info['depth1'],
    #                 self.eval_info['depth_factor'],
    #                 )
    #         mean_aor = np.mean(aor)
            
    #         iou = calc_areas_iou_eval(areas0, areas1, 
    #                                     self.eval_info['depth0'], self.eval_info['depth1'], 
    #                                     self.eval_info['K0'], self.eval_info['K1'], 
    #                                     self.eval_info['P0'], self.eval_info['P1'], 
    #                                     self.eval_info['depth_factor'])
    #         mean_iou = np.mean(iou)
    #     except Exception as e:
    #         logger.error(f'Error: {e}')
    #         raise e
        
    #     # write to file
    #     name_file = os.path.join(self.out_path, f'{pre_name}_ameval_names.txt')
    #     aor_file = os.path.join(self.out_path, f'{pre_name}_aor.txt')
    #     acr_file = os.path.join(self.out_path, f'{pre_name}_acr.txt')
    #     iou_file = os.path.join(self.out_path, f'{pre_name}_iou.txt')
        
    #     # read names
    #     if not os.path.exists(name_file):
    #         exist_names = []
    #     else:
    #         with open(name_file, 'r') as f:
    #             exist_names = f.readlines()
    #             exist_names = [name.strip() for name in exist_names]
        
    #     if self.instance_name in exist_names:
    #         pass
    #     else:
    #         with open(name_file, 'a') as f:
    #             f.write(f'{self.instance_name}\n')
    #         with open(aor_file, 'a') as f:
    #             f.write(f'{mean_aor}\n')
    #         with open(acr_file, 'a') as f:
    #             f.write(f'{acr}\n')
    #         with open(iou_file, 'a') as f:
    #             f.write(f'{mean_iou}\n')

    def eval_area_overlap_ratio(self,
        areas0,
        areas1,
        pre_name,
        corrs=None,
        mp_thresholds=None,
        ):
        """
        Args:
            areas0/1: list of [u_min, u_max, v_min, v_max] 
            corrs: list of [x0, y0, x1, y1], used for point-set IoU overlap
        Note:
            IoU is intersection over union; AOR is the directional area-overlap
            ratio used by the region-matching evaluation.
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.eval_info['dataset_name'] in ['ScanNet'], 'Error: dataset not supported'

        thresholds = mp_thresholds if mp_thresholds is not None else [0.2, 0.4, 0.6]
        
        # Initialize values to zero
        acr = 0
        aor = 0
        iou = 0
        
        if len(areas0) < 1 or len(areas1) < 1:
            logger.error(f'Error: areas0/1 is empty')
            # No return here, continue to write zeros to files
        else:
            try:
                acr, aor = calc_area_match_performence_eff_MC(
                        areas0,
                        areas1,
                        self.eval_info['image0'],
                        self.eval_info['image1'],
                        self.eval_info['K0'],
                        self.eval_info['K1'],
                        self.eval_info['P0'],
                        self.eval_info['P1'],
                        self.eval_info['depth0'],
                        self.eval_info['depth1'],
                        self.eval_info['depth_factor'],
                        )
                
                iou = calc_areas_iou_eval(areas0, areas1, 
                                        self.eval_info['depth0'], self.eval_info['depth1'], 
                                        self.eval_info['K0'], self.eval_info['K1'], 
                                        self.eval_info['P0'], self.eval_info['P1'], 
                                        self.eval_info['depth_factor'])
            except Exception as e:
                logger.error(f'Error: {e}')
                # No raise here, use initialized zero values
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_ameval_names.txt')
        aor_file = os.path.join(self.out_path, f'{pre_name}_aor.txt')
        acr_file = os.path.join(self.out_path, f'{pre_name}_acr.txt')
        iou_file = os.path.join(self.out_path, f'{pre_name}_iou.txt')
        mom_file = os.path.join(self.out_path, f'{pre_name}_mom.txt')
        num_matches_file = os.path.join(self.out_path, f'{pre_name}_num_matches.txt')
        mqd_median_file = os.path.join(self.out_path, f'{pre_name}_mqd_median.txt')
        mqd_std_file = os.path.join(self.out_path, f'{pre_name}_mqd_std.txt')
        mqd_min_file = os.path.join(self.out_path, f'{pre_name}_mqd_min.txt')
        mqd_max_file = os.path.join(self.out_path, f'{pre_name}_mqd_max.txt')
        mqd_q25_file = os.path.join(self.out_path, f'{pre_name}_mqd_q25.txt')
        mqd_q75_file = os.path.join(self.out_path, f'{pre_name}_mqd_q75.txt')
        mqd_iqr_file = os.path.join(self.out_path, f'{pre_name}_mqd_iqr.txt')
        mp_files = {
            tau: os.path.join(self.out_path, f'{pre_name}_mp_at_{self._format_tau_for_file(tau)}.txt')
            for tau in thresholds
        }

        bbox_metrics = self._compute_bbox_point_metrics(areas0, areas1, corrs, thresholds)
        
        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]
        
        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(acr_file, 'a') as f:
                f.write(f'{acr}\n')
            
            # Save all AOR values (comma-separated)
            with open(aor_file, 'a') as f:
                # Handle the case when aor is 0 (integer) instead of a list
                if isinstance(aor, (int, float)):
                    aor_str = str(aor)
                else:
                    aor_str = ','.join(map(str, aor))
                f.write(f'{aor_str}\n')
            
            # Save all IoU values (comma-separated)
            with open(iou_file, 'a') as f:
                # Handle the case when iou is 0 (integer) instead of a list
                if isinstance(iou, (int, float)):
                    iou_str = str(iou)
                else:
                    iou_str = ','.join(map(str, iou))
                f.write(f'{iou_str}\n')

            # Save point-IoU style bbox metrics (ported from sgad)
            with open(mom_file, 'a') as f:
                f.write(f"{bbox_metrics['MOM']}\n")
            with open(num_matches_file, 'a') as f:
                f.write(f"{bbox_metrics['num_matches']}\n")

            mqd = bbox_metrics["MQD"]
            with open(mqd_median_file, 'a') as f:
                f.write(f"{mqd['median']}\n")
            with open(mqd_std_file, 'a') as f:
                f.write(f"{mqd['std']}\n")
            with open(mqd_min_file, 'a') as f:
                f.write(f"{mqd['min']}\n")
            with open(mqd_max_file, 'a') as f:
                f.write(f"{mqd['max']}\n")
            with open(mqd_q25_file, 'a') as f:
                f.write(f"{mqd['Q25']}\n")
            with open(mqd_q75_file, 'a') as f:
                f.write(f"{mqd['Q75']}\n")
            with open(mqd_iqr_file, 'a') as f:
                f.write(f"{mqd['IQR']}\n")

            for tau in thresholds:
                with open(mp_files[tau], 'a') as f:
                    f.write(f"{bbox_metrics['MP'][tau]}\n")

    def eval_point_match(self,
        corrs,
        pre_name,
        thds=[1,3,5],
        ):
        """
        Args:
            corrs: all corrs should be in the eval size
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.sample_mode in ['random', 'grid'], f'Error: sample_mode {sample_mode} not supported'

        eval_num = self.eval_corr_num
        sample_mode = self.sample_mode

        corr_num = len(corrs)
        if corr_num < 10:
            logger.error(f'Error: not enough corrs for pose estimation')
            return []

        if corr_num > eval_num:
            if sample_mode == 'random':
                corrs = random.sample(corrs, eval_num)
            elif sample_mode == 'grid':
                corrs = np.array(corrs)
                corrs = nms_for_corrs(corrs, r=3)
                if len(corrs) > eval_num:
                    corrs = random.sample(corrs, eval_num)

        good_ratios = []
        masks = []

        try:
            dataset_name = self.eval_info['dataset_name']
            pose0 = self.eval_info['P0']
            pose1 = self.eval_info['P1']
            K0 = self.eval_info['K0']
            K1 = self.eval_info['K1']
            image0 = self.eval_info['image0'] # imgs are in eval size
            image1 = self.eval_info['image1']
            if dataset_name == "ScanNet":
                depth0 = self.eval_info['depth0']
                depth1 = self.eval_info['depth1']
                depth_factor = self.eval_info['depth_factor']
                for thd in thds:
                    mask, bad_ratio, gt_pts = assert_match_reproj(corrs,
                                        depth0, depth1, depth_factor,
                                        K0, K1,
                                        pose0, pose1,
                                        thd, 0)
                    good_ratio = (100 - bad_ratio)/100
                    good_ratios.append(good_ratio)
                    masks.append(mask)

            elif dataset_name in ['MegaDepth', 'YFCC', 'ETH3D']:
                for thd in thds:
                    mask, bad_ratio = assert_match_qFp(corrs, K0, K1, pose0, pose1, thd)
                    good_ratio = (1 - bad_ratio)
                    good_ratios.append(good_ratio)
                    masks.append(mask)
            else:
                raise NotImplementedError(f"dataset {dataset_name} not supported")

            # draw match images with masks for corrs
            if self.draw_verbose:
                # get the upper path of the out_path
                up_out_path = os.path.dirname(self.out_path)
                pm_out_path = os.path.join(up_out_path, self.instance_name, 'pm')
                test_dir_if_not_create(pm_out_path)
                for i, thd in enumerate(thds):
                    plot_matches_with_mask_ud(
                        image0, image1,
                        masks[i], corrs, pm_out_path, pre_name+f"_mma_{thd}")

        except Exception as e:
            logger.error(f'Error: {e}')
            raise e
            
        
        logger.success(f'point matches good ratios: {good_ratios}')
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_mma_names.txt')
        mma_file = os.path.join(self.out_path, f'{pre_name}_mmas.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]

        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(mma_file, 'a') as f:
                for i, _ in enumerate(thds):
                    f.write(f'{good_ratios[i]} ')
                f.write('\n')

        return good_ratios


    def eval_pose_error(self, 
        corrs,
        pre_name,
        ):
        """
        Args:
            corrs: list in eval size
        Returns:
            pose error: [R_err, t_err]
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.sample_mode in ['random', 'grid'], f'Error: sample_mode {sample_mode} not supported'

        corr_num = len(corrs)
        if corr_num < 10:
            logger.error(f'Error: not enough corrs for pose estimation')
            errs = [180, 180]

        eval_num = self.eval_corr_num
        sample_mode = self.sample_mode
        if corr_num > eval_num:
            if sample_mode == 'random':
                corrs = random.sample(corrs, eval_num)
            elif sample_mode == 'grid':
                corrs = np.array(corrs)
                corrs = nms_for_corrs(corrs, r=3)
                if len(corrs) > eval_num:
                    corrs = random.sample(corrs, eval_num)
        
        try:
            dataset_name = self.eval_info['dataset_name']
            pose0 = self.eval_info['P0']
            pose1 = self.eval_info['P1']
            K0 = self.eval_info['K0']
            K1 = self.eval_info['K1']
            if dataset_name == "MegaDepth":
                gt_pose = np.matmul(pose1, np.linalg.inv(pose0))
            else:
                gt_pose = pose1.I @ pose0

            errs = compute_pose_error_simp(corrs, K0, K1, gt_pose, pix_thd=0.5, conf=0.9999, sac_mode=self.sac_mode)

        except Exception as e:
            logger.error(f'Error: {e}')
            if e is KeyError:
                raise e
            else:
                errs = [180, 180]
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_pose_err_names.txt')
        pose_err_file = os.path.join(self.out_path, f'{pre_name}_pose_errs.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]

        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(pose_err_file, 'a') as f:
                f.write(f'{errs[0]} {errs[1]}\n')

        return errs


    @staticmethod
    def tune_corrs_size(corrs, 
        src_W0=None, 
        src_W1=None,
        src_H0=None,
        src_H1=None,
        dst_W0=None,
        dst_W1=None,
        dst_H0=None,
        dst_H1=None):
        """
        """
        return tune_corrs_size_diff(corrs, src_W0, src_W1, src_H0, src_H1, dst_W0, dst_W1, dst_H0, dst_H1)
    
    def eval_epipolar_error_and_precision(self, corrs, pre_name):
        """
        Args:
            corrs: list of correspondences for evaluation
        Returns:
            epi_errs: list of symmetric epipolar distances
            precision: fraction of matches below the epipolar error threshold;
                zero when fewer than 10 correspondences are available
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, initialize dataloader first'
        assert self.sample_mode in ['random', 'grid'], f'Error: sample_mode {self.sample_mode} not supported'

        corr_num = len(corrs)
        if corr_num < 10:
            logger.error('Error: not enough correspondences for epipolar error estimation')
            epi_errs = [np.inf]
            precision = 0
        else:
            eval_num = self.eval_corr_num
            sample_mode = self.sample_mode
            if corr_num > eval_num:
                if sample_mode == 'random':
                    corrs = random.sample(corrs, eval_num)
                elif sample_mode == 'grid':
                    corrs = np.array(corrs)
                    corrs = nms_for_corrs(corrs, r=3)
                    if len(corrs) > eval_num:
                        corrs = random.sample(corrs, eval_num)

            corr_num = len(corrs)

            try:
                K0 = self.eval_info['K0']
                K1 = self.eval_info['K1']

                epi_errs = compute_epi_errs(corrs, K0, K1)

                corrs = np.array(corrs) # N x 4
                dataset_name = self.eval_info['dataset_name']
                conf_thr = _compute_conf_thresh(dataset_name)
                correct_mask = epi_errs < conf_thr
                precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0

                # logger.success(f"point match precision is: {precision}")

            except Exception as e:
                logger.error(f'Error: {e}')
                epi_errs = [np.inf]
                precision = 0

        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_precision_names.txt')
        precision_file = os.path.join(self.out_path, f'{pre_name}_precision.txt')
        corr_num_file = os.path.join(self.out_path, f'{pre_name}_corr_num.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]

        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(precision_file, 'a') as f:
                f.write(f'{precision}\n')
            with open(corr_num_file, 'a') as f:  # Write corr_num to the new file
                f.write(f'{corr_num}\n')

        return epi_errs, precision
