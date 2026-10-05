import os
import numpy as np
import cv2
from loguru import logger
from collections import defaultdict
import copy
from omegaconf import OmegaConf

from utils.load import (
    load_cv_img_resize, 
)

from utils.geo import (
    recover_corrs_offset_scales, 
    calc_sampson_dist,
    calc_E_from_corrs,
    recover_F_from_E_K,
    list_of_corrs2corr_list,
    is_corr_inside_area,
    tune_corrs_size_diff,
    tune_corrs_size,
    
)

from utils.img_process import (
    img_crop_fix_aspect_ratio,
    img_crop_with_padding_improve_resolution,
    img_crop_with_padding_expand_square,
)


from utils.vis import (
    plot_matches_lists_lr,
    plot_matches_lists_ud,
    draw_matched_area_list,
)

from utils.common import test_dir_if_not_create

from .abstract_gam import AbstractGeoAreaMatcher

class EGeoAreaMatcher(AbstractGeoAreaMatcher):
    """ Simplified GAM
        - It estimates essential matrix to calc Sampson Distance to perform GR.
    Inputs:
        matched_area{0/1}s
    Outputs:
        refined_corrs
        inlier_area_idxs
    """

    def __init__(self, 
        area_from_size_W,
        area_from_size_H,
        crop_size_W,
        crop_size_H,
        area_crop_size_W,
        area_crop_size_H,
        crop_from_size_W,
        crop_from_size_H,
        eval_from_size_W,
        eval_from_size_H,
        std_match_num,
        alpha_list,
        valid_inside_area_match_num,
        reject_out_area_flag,
        crop_mode,
        sac_mode,
        occ_size,
        common_occ_flag,
        datasetName="ScanNet", 
        verbose=0,
        min_area_matches_for_geom=None,
        min_inliers_for_geom=None) -> None:
        """
        """
        self._name = "E-GeoAreaMatcher"
        self.datasetName = datasetName

        # Region boxes are expressed on the area-matcher input canvas.
        self.area_from_size_W, self.area_from_size_H = area_from_size_W, area_from_size_H
        ## to get point matches within areas, crop area with crop_size from original image with crop_from_size
        # self.crop_size_W, self.crop_size_H = crop_size_W, crop_size_H
        self.crop_size_W, self.crop_size_H = area_crop_size_W, area_crop_size_H   
        self.crop_from_size_W, self.crop_from_size_H = crop_from_size_W, crop_from_size_H
        # Point and pose metrics are evaluated on the evaluation canvas.
        self.eval_from_size_W, self.eval_from_size_H = eval_from_size_W, eval_from_size_H

        # other configs
        self.std_match_num = std_match_num

        # Sampson-distance thresholds.
        self.alpha_list = OmegaConf.to_container(alpha_list)
        assert type(self.alpha_list) == list, f"alpha_list should be a list, but {self.alpha_list} type is {type(self.alpha_list)}"
        self.valid_inside_area_match_num = int(valid_inside_area_match_num)
        # Split thresholds (default keeps original behavior):
        # - min_area_matches_for_geom: minimum in-area matches before estimating E/F (pre-MAGSAC)
        # - min_inliers_for_geom: minimum inliers after MAGSAC when accepting E/F
        self.min_area_matches_for_geom = (
            self.valid_inside_area_match_num if min_area_matches_for_geom is None else int(min_area_matches_for_geom)
        )
        self.min_inliers_for_geom = (
            self.valid_inside_area_match_num if min_inliers_for_geom is None else int(min_inliers_for_geom)
        )
        # flags
        self.draw_verbose = verbose
        # Reject crop matches that map outside their source regions when enabled.
        self.reject_out_area_flag = reject_out_area_flag
        # Region crop strategy: 0 aspect-ratio crop, 1 padded crop,
        # 2 square-expanded padded crop, or 3 masked crop.
        self.crop_mode = crop_mode
        ## estimator mode
        ### "MAGSAC": MAGSAC++ is used to estimate E
        ### "RANSAC": RANSAC is used to estimate E
        self.sac_mode = sac_mode

        # holders
        self.ori_img_coors = [] # point matches in the original (entire) images
        self.all_input_matched_area0s = None
        self.all_input_matched_area1s = None
        
        self.occ_size = occ_size
        self.common_occ_flag = common_occ_flag

        self.initialized = False
        self.time_pm = 0

    def name(self) -> str:
        return self._name

    def init_dataloader(self, dataloader):
        """
        """
        self.name0 = dataloader.image_name0
        self.name1 = dataloader.image_name1
        self.scene_name = dataloader.scene_name

        # load imgs
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            # Keep MegaDepth/YFCC/ETH3D on configured pipeline sizes instead of
            # silently falling back to original image size.
            self.ori_img0, self.ori_img1, self.img_scale0, self.img_scale1 = dataloader.load_images(
                self.crop_from_size_W, self.crop_from_size_H
            )
            self.crop_from_size_W0, self.crop_from_size_H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]
            self.crop_from_size_W1, self.crop_from_size_H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]

            # color{0,1} are used for eval and for K scaling.
            self.color0, self.color1, self.scale0, self.scale1 = dataloader.load_images(
                self.eval_from_size_W, self.eval_from_size_H
            )
            self.eval_from_size_W0, self.eval_from_size_H0 = self.color0.shape[1], self.color0.shape[0]
            self.eval_from_size_W1, self.eval_from_size_H1 = self.color1.shape[1], self.color1.shape[0]
            
        # ScanNet, KITTI, MatterPort3D
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            # ori_imgs are used for crop
            self.ori_img0, self.ori_img1, self.img_scale0, self.img_scale1 = dataloader.load_images(self.crop_from_size_W, self.crop_from_size_H)

            # color0, color1 are used for eval, load for their scale and resize the following Ks
            self.color0, self.color1, self.scale0, self.scale1 = dataloader.load_images(self.eval_from_size_W, self.eval_from_size_H)

            # fill the eval size, which is used in sampler
            self.eval_from_size_W0, self.eval_from_size_W1 = self.eval_from_size_W, self.eval_from_size_W
            self.eval_from_size_H0, self.eval_from_size_H1 = self.eval_from_size_H, self.eval_from_size_H
        else:
            raise NotImplementedError(f"datasetName '{self.datasetName}' not supported")

        # load geos
        self.depth0 = None
        self.depth1 = None
        self.depth_factor = None
        self.K0, self.K1 = dataloader.load_Ks(self.scale0, self.scale1)
        self.pose0, self.pose1 = dataloader.load_poses()
        
        if self.datasetName == "ScanNet" or self.datasetName == "MatterPort3D":
            self.depth0, self.depth1 = dataloader.load_depths()
            self.depth_factor = dataloader.get_depth_factor()
        elif self.datasetName == "YFCC":
            # turn ndarray to matrix, YFCC need to inverse
            self.pose0 = self.pose0.I
            self.pose1 = self.pose1.I
        elif self.datasetName == "KITTI" or self.datasetName == "ETH3D" or self.datasetName == "MegaDepth":
            pass
        else:
            raise NotImplementedError

    def load_point_matcher(self, point_matcher):
        """
        """
        self.PMatcher = point_matcher

    def load_ori_corrs(self, ori_corrs):
        """
        """
        self.ori_img_coors = ori_corrs

    def doubtful_area_match_predict(self, doubt_match_pairs):
        """
        """
        assert self.initialized, f"{self.name} not initialized"
        logger.error(f"{self.name} has no doubtful area match predicter")
        raise NotImplementedError

    def init_gam(self,
        dataloader,
        point_matcher,
        ori_corrs,
        out_path,
        ):
        """
        """
        self.load_point_matcher(point_matcher)
        self.init_dataloader(dataloader)
        self.load_ori_corrs(ori_corrs)
        self.set_outpath(out_path)
        self.initialized = True

        # draw
        if self.draw_verbose == 1:
            self.draw_ori_img()
            test_dir_if_not_create(self.out_path)

    def geo_area_matching_refine(self, 
        matched_areas0, 
        matched_areas1):
        """ 
        Returns:
            alpha_corrs_dict: {
                alpha0: [[inside-area-corrs], ...],
                alpha1: [[inside-area-corrs], ...],
                ... NOTE: corrs in eval-from-size
            }
            alpha_inlier_idxs_dict: {
                alpha0: [inlier-area-idxs, ...], # index corresponding to input area list
                alpha1: [inlier-area-idxs, ...],
                ...
            }
            [Optional] alpha_sampled_corrs: { 
                alpha0: [sampled-corrs, ...],
                alpha1: [sampled-corrs, ...],
                ...
            }
        """
        assert self.initialized, f"{self.name} not initialized, run .init_gam first"

        self.time_pm = 0

        alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs = self.geo_area_match_rejection_sampler(matched_areas0, matched_areas1)
        logger.success(f"area PM time: {self.time_pm:.4f}s")
        return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs, self.time_pm

    """=============Rejection Part=================="""

    # Main Function without AC
    def geo_area_match_rejection_sampler(self, matched_area0s, matched_area1s):
        """ Perform GAM
        Flow:
            1. for each area matches:
                - match points - put into eval-from-size
                - normalize points with K
                - estimate E using MAGSAC++
                - get mask to filter inlier points
                - recover inlier points to original images
                - recover F and calculate Sampson Distance
            2. calc all-areas-mean Sampson Distance and derive thd
            3. calc geometry consistency of every area matches
        Returns:
            alpha-corr-dict = {
                alpha0: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                alpha1: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                ... NOTE: corrs are masked inliers and in eval-from-size
            }
            alpha-inlier-idxs = {
                alpha0: [inlier-area0-idxs, inlier-area1-idxs, ...],
                alpha1: [inlier-area0-idxs, inlier-area1-idxs, ...],
                ...
            }
            if no area matches, return original matched points
        """
        logger.info(f"Start GAM with Sampler Rejection")
        # holder
        alpha_corrs_dict = defaultdict(list)
        alpha_inlier_idxs_dict = defaultdict(list)
        alpha_sampled_corrs_dict = defaultdict(list)
        # if no area matches, return directly the original matched points
        if len(matched_area0s) == 0 or len(matched_area1s) == 0:
            logger.info(f"No area matches, return directly")
            
            if len(self.ori_img_coors) == 0:
                logger.error(f"No matched points in the original images")
                return tuple({alpha: [] for alpha in self.alpha_list} for _ in range(3))

            # fill the original matched points
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = [self.ori_img_coors]
                alpha_inlier_idxs_dict[alpha] = []
                alpha_sampled_corrs_dict[alpha] = self.ori_img_coors

            return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs_dict

        # load all area matches: put into crop-from-size
        self.tune_all_area_matches_to_crop_from_size(matched_area0s, matched_area1s)

        # draw areas before GAM 
        if self.draw_verbose == 1:
            self.draw_match_areas(self.all_input_matched_area0s, self.all_input_matched_area1s, "before_GAM")

        # for each area matches, calc inside point matches and Sampson Distance
        inlier_mkpts_list = [] # list of corrs
        geometry_inlier_mkpts_list = []
        F_list = []
        self_sampson_dist_list = []
        for idx, (area0, area1) in enumerate(zip(self.all_input_matched_area0s, self.all_input_matched_area1s)):
            logger.info(f"Start SD calc for area {idx}")
            temp_inlier_corrs, temp_F, temp_self_sampson_dist, temp_geometry_inliers = self.inside_area_match_calculation(area0, area1, idx)
            
            if temp_inlier_corrs is None:
                inlier_mkpts_list.append(None)
                geometry_inlier_mkpts_list.append(None)
                F_list.append(None)
                self_sampson_dist_list.append(None)
                continue
            else:
                inlier_mkpts_list.append(temp_inlier_corrs)
                geometry_inlier_mkpts_list.append(temp_geometry_inliers)
                F_list.append(temp_F)
                self_sampson_dist_list.append(temp_self_sampson_dist)

        # derive thd by take the mean of all areas Sampson Distance
        # mind the None
        self_sampson_dist_list_unorder = [dist for dist in self_sampson_dist_list if dist is not None]
        if len(self_sampson_dist_list_unorder) == 0:
            logger.error(f"No valid Sampson Distance, return directly")
            # Preserve failed matching as empty outputs, without falling back to
            # PM, so every alpha receives an explicit failed evaluation record.
            return tuple({alpha: [] for alpha in self.alpha_list} for _ in range(3))
                
        mean_sd = np.mean(self_sampson_dist_list_unorder)

        # derive thd
        gam_thd_dict = {}
        for alpha in self.alpha_list:
            gam_thd_dict[alpha] = mean_sd * alpha
        
        # for each area matches, calc geometry consistency
        geo_cons_list = []
        for idx, temp_F in enumerate(F_list):
            if temp_F is None:
                geo_cons_list.append(None)
                continue

            # calc geometry consistency
            geo_cons = self.calc_geo_consistency(temp_F, geometry_inlier_mkpts_list)
            geo_cons_list.append(geo_cons)

        # for each alpha, filter inliers
        for alpha in self.alpha_list:
            logger.info(f"Start filtering inliers for alpha {alpha}")
            inlier_idxs = []
            for idx, geo_cons in enumerate(geo_cons_list):
                if geo_cons is None:
                    continue
                if geo_cons <= gam_thd_dict[alpha]:
                    inlier_idxs.append(idx)
                    
            alpha_inlier_idxs_dict[alpha] = inlier_idxs
            # fill inlier corrs
            alpha_corrs_dict[alpha] = [inlier_mkpts_list[idx] for idx in inlier_idxs]

        # draw areas after GAM
        if self.draw_verbose == 1:
            for alpha in self.alpha_list:
                temp_area0s = [self.all_input_matched_area0s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                temp_area1s = [self.all_input_matched_area1s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                self.draw_match_areas(temp_area0s, temp_area1s, f"after_GAM_alpha_{alpha}")

        # till now, we have got the inlier corrs for each alpha
        # next, we focus on how to sample the inliers to achieve spatially even distribution
        # we use sampler to achieve this

        for alpha in self.alpha_list:
            # Pass the selected per-region inliers and the unfiltered global
            # correspondences directly to the sampler. EGAM owns region-model
            # estimation and cross-region selection; global fusion belongs to
            # the sampler.
            temp_alpha_corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
            if len(temp_alpha_corrs) < 8:
                logger.warning(f"Too few corrs for alpha {alpha}, skip")
                alpha_corrs_dict[alpha] = self.ori_img_coors
                alpha_sampled_corrs_dict[alpha] = self.ori_img_coors
                continue
            original_corrs = self.ori_img_coors
            # Valid regional correspondences can be sampled without global PM.
            alpha_corrs_dict[alpha] = temp_alpha_corrs + original_corrs
                                        
            # sampler
            sampler = self._load_sampler()
            sampler.load_ori_imgs(self.color0, self.color1)
            sampler.name = f"{alpha}"
            
            cur_corrs_list = copy.deepcopy(temp_alpha_corrs)
            cur_corrs_list.append(original_corrs)
            sampler.load_corrs_from_GAM(cur_corrs_list)

            sampled_corrs, sampled_corrs_rand = sampler.sample()
            if sampled_corrs is None:
                alpha_sampled_corrs_dict[alpha] = self.ori_img_coors
            else:
                alpha_sampled_corrs_dict[alpha] = sampled_corrs_rand
        
        return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs_dict


    """=============Rejection Utils Part=================="""

    def _load_sampler(self):
        """
        """
        from .MatchSampler import GridFillSampler

        sampler_cfg = {
            "W0": self.eval_from_size_W0,
            "H0": self.eval_from_size_H0,
            "W1": self.eval_from_size_W1,
            "H1": self.eval_from_size_H1,
            "out_path": self.out_path,
            "sample_num": self.std_match_num,
            "draw_verbose": self.draw_verbose,
            "occ_size": self.occ_size,
            "common_occ_flag": self.common_occ_flag,
            "K0": self.K0,
            "K1": self.K1,
            "sac_mode": self.sac_mode,
            "datasetName": self.datasetName,
        }

        return GridFillSampler(sampler_cfg)

    def calc_geo_consistency(self, F, inlier_corrs_list):
        """ calc geometry consistency
        Funs:
            0. remove None
            1. turn list of corrs to corr list
            2. calc Sampson Distance
        """
        # remove None
        non_None_corrs_list = [corrs for corrs in inlier_corrs_list if corrs is not None]

        corrs = list_of_corrs2corr_list(non_None_corrs_list)
        geo_cons = calc_sampson_dist(F, corrs)
        
        return geo_cons

    def inside_area_match_calculation(self, area0, area1, idx):
        """ Calc point matches, Sampson Distance, and inlier mask for area matches
        Args:
            area0, area1: [u_min, u_max, v_min, v_max]
        Returns:
            inlier_corrs: in eval-from-size, if None, means no valid matches
            F: Fundamental Matrix
            self_sampson_dist: self Sampson Distance
        """

        # crop areas from original images and match, NOTE: output in eval-from-size
        if self.crop_mode == 0:
            temp_corrs = self.match_area_pair_mind_size(area0, area1, f"rejecting_{idx}") # corrs in eval_from_size
        elif self.crop_mode == 1:
            temp_corrs = self.match_area_pair_padding(area0, area1, f"rejecting_{idx}")
        elif self.crop_mode == 2:
            temp_corrs = self.match_area_pair_padding_expand_square(area0, area1, f"rejecting_{idx}")
        elif self.crop_mode == 3:
            temp_corrs = self.match_area_pair_mask(area0, area1, f"rejecting_{idx}")
        else:
            raise NotImplementedError

        if len(temp_corrs) <= self.min_area_matches_for_geom:
            logger.warning(
                f"{len(temp_corrs)} Less than {self.min_area_matches_for_geom} matches for area {idx}, skip"
            )
            return None, None, None, None

        if self.reject_out_area_flag:
            temp_corrs = self.reject_corrs_outside_area(temp_corrs, area0, area1)
            if len(temp_corrs) <= self.min_area_matches_for_geom:
                logger.warning(
                    f"{len(temp_corrs)} Less than {self.min_area_matches_for_geom} matches for area {idx}, skip"
                )
                return None, None, None, None

        # calc E
        estimate = calc_E_from_corrs(temp_corrs, self.K0, self.K1, sac_mode=self.sac_mode, datasetName=self.datasetName)
        if estimate is None:
            return None, None, None, None
        E, inlier_corrs = estimate
        F = recover_F_from_E_K(E, self.K0, self.K1)

        if len(inlier_corrs) <= self.min_inliers_for_geom:
            logger.warning(
                f"{len(inlier_corrs)} Less than {self.min_inliers_for_geom} inliers for area {idx}, skip"
            )
            return None, None, None, None

        # calc self Sampson Distance
        inlier_corrs_np = np.array(inlier_corrs)
        self_sampson_dist = calc_sampson_dist(F, inlier_corrs_np)

        return inlier_corrs, F, self_sampson_dist, inlier_corrs
    
    def reject_corrs_outside_area(self, corrs, area0, area1):
        """
        Args:
            corrs in eval_from_size
            areas in eval_from_size
        """
        refined_corrs = []
        for corr in corrs:
            if is_corr_inside_area(corr, area0, area1):
                refined_corrs.append(corr)
            else:
                pass
        
        return refined_corrs

    def match_area_pair_padding_expand_square(self, area0, area1, name=""):
        """use padding to match, but expand area to square first
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_with_padding_expand_square(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_with_padding_expand_square(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)
        import time
        time_pm_start = time.time()
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []
        time_pm_end = time.time()
        self.time_pm += time_pm_end - time_pm_start
        logger.success(f"area point matching time: {time_pm_end - time_pm_start:.4f}s")

        temp_corrs = self.PMatcher.return_matches()
        temp_corrs = np.array(temp_corrs)

        # draw the matches on area matches
        if self.draw_verbose == 1:
            # Render matches in the two region crops.
            plot_matches_lists_lr(crop0, crop1, temp_corrs, self.out_path, name=f"area_point_{name}")

        if temp_corrs.shape[0] == 0: return []

        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]] # [u0, v0, u1, v1]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size

        # recover to eval size
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            rt_corrs = tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)
        else:
            raise NotImplementedError

        return rt_corrs

    def match_area_pair_padding(self, area0, area1, name=""):
        """use padding to match
        Args:
            area0/1: [u_min, u_max, v_min, v_max] in crop_from_size
        Returns:
            rt_corrs: [[u0, v0, u1, v1]s] in eval_from_size
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_with_padding_improve_resolution(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_with_padding_improve_resolution(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)

        import time
        time_pm_start = time.time()
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []
        time_pm_end = time.time()
        self.time_pm += time_pm_end - time_pm_start
        logger.success(f"area point matching time: {time_pm_end - time_pm_start:.4f}s")

        temp_corrs = self.PMatcher.return_matches()
        temp_corrs = np.array(temp_corrs)

        # draw the matches on area matches
        if self.draw_verbose == 1:
            # Render matches in the two region crops.
            plot_matches_lists_lr(crop0, crop1, temp_corrs, self.out_path, name=f"area_point_{name}")
            
        if temp_corrs.shape[0] == 0: return []

        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size

        # recover to eval size
        rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)

        return rt_corrs

    def match_area_pair_mind_size(self, area0, area1, name=""):
        """Match points in two region crops and recover evaluation coordinates.

        Returns:
            rt_corrs: [[u0, v0, u1, v1]s] in eval_from_size
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_fix_aspect_ratio(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_fix_aspect_ratio(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)

        import time
        time_pm_start = time.time()
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []
        time_pm_end = time.time()
        self.time_pm += time_pm_end - time_pm_start
        logger.success(f"area point matching time: {time_pm_end - time_pm_start:.4f}s")

        temp_corrs = self.PMatcher.return_matches()
        temp_corrs = np.array(temp_corrs)
        if temp_corrs.shape[0] == 0: 
            logger.warning("No valid matches")
            return []

        # draw the matches on area matches
        if self.draw_verbose == 1:
            # Render matches in the two region crops.
            plot_matches_lists_lr(crop0, crop1, temp_corrs, self.out_path, name=f"area_point_{name}")
            
        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size
        
        # self.draw_area_match_res(rt_corrs, name)
        
        # recover to eval size
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
        # if self.datasetName == "MegaDepth":
            rt_corrs = tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)
        else:
            raise NotImplementedError

        logger.info(f"achieve {np.array(rt_corrs).shape} corrs for {name}")

        return rt_corrs

    def tune_all_area_matches_to_crop_from_size(self, matched_area0s, matched_area1s):
        """ Load all area matches to crop-from-size
        Args:
            self.all_area0_matches, self.all_area1_matches in the crop-from-size
        """
        assert matched_area0s is not None, f"all_area0_matches is None"
        assert matched_area1s is not None, f"all_area1_matches is None"
        assert len(matched_area0s) == len(matched_area1s), f"len(all_area0_matches) != len(all_area1_matches)"

        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W0, self.crop_from_size_H0)
            self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W1, self.crop_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
            self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
        else:
            raise NotImplementedError
        
        logger.info(f"Load all area matches to crop-from-size")

        return self.all_input_matched_area0s, self.all_input_matched_area1s


    """=============Utils Part=================="""

    def draw_match_areas(self, matched_area0s, matched_area1s, name=""):
        """ areas in crop from size
        """
        draw_matched_area_list(self.ori_img0, self.ori_img1, matched_area0s, matched_area1s, self.out_path, name, "")

    def tune_area_size(self, area, src_W, src_H, dst_W, dst_H):
        """
        Args:
            area: [u_min, u_max, v_min, v_max]
        Returns:
            area
        """
        W_ratio = dst_W / src_W
        H_ratio = dst_H / src_H

        u_min, u_max, v_min, v_max = area

        u_min_ = u_min * W_ratio
        u_max_ = u_max * W_ratio
        v_min_ = v_min * H_ratio
        v_max_ = v_max * H_ratio

        area_ = [u_min_, u_max_, v_min_, v_max_]

        return area_

    def tune_area_list_size(self, area_list, src_W, src_H, dst_W, dst_H):
        """
        """
        rt_area_list = []

        for area in area_list:
            area_ = self.tune_area_size(area, src_W, src_H, dst_W, dst_H)
            rt_area_list.append(area_)
        
        return rt_area_list

    def load_ori_img_corrs_in_eval_size(self, corrs):
        """
        """

    def draw_ori_img(self):
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori0.jpg"), self.ori_img0)
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori1.jpg"), self.ori_img1)
