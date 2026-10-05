import os
from loguru import logger
from utils.vis import draw_matched_area, draw_matched_area_list
from utils.common import validate_type

from dataloader.abstract_dataloader import AbstractDataloader
from .abstract_am import AbstractAreaMatcher


class SgadAreaMatcher(AbstractAreaMatcher):
    """A2PM adapter for the SGAD and SGAD++ learned region matchers."""
    def __init__(
        self,
        W,
        H,
        level_num,
        level_step,
        datasetName,
        draw_verbose=0,
        areagraph_overrides=None,
        hcrf=False,
        hcrf_overlap_threshold=0.9,
        hcrf_coverage_threshold=0.4,
        weights=None,
        model_variant=None,
        pretrained_root=None,
        pred_score_threshold=None,
    ):
        self.W, self.H = W, H
        # Keep area-matcher eval info on the same canvas as AM input size.
        self.eval_W, self.eval_H = W, H
        self.datasetName = datasetName
        self.draw_verbose = draw_verbose

        self.level_num = level_num
        self.level_step = level_step

        self.eval_info = None
        self.instance_name = None

        self.areagraph_overrides = areagraph_overrides

        self.hcrf = hcrf
        self.hcrf_overlap_threshold = hcrf_overlap_threshold
        self.hcrf_coverage_threshold = hcrf_coverage_threshold
        self.weights = weights
        self.model_variant = model_variant
        self.pretrained_root = pretrained_root
        self.pred_score_threshold = pred_score_threshold

    def init_dataloader(self, dataloader):
        """
        """
        validate_type(dataloader, AbstractDataloader)
        self.bbox_path0, self.bbox_path1 = dataloader.get_bbox_paths()

        self.name0 = dataloader.image_name0
        self.name1 = dataloader.image_name1

        self.image0_path, self.image1_path = dataloader.img0_path, dataloader.img1_path

        self.scene_name = dataloader.scene_name

        self.img0, self.img1, self.scale0, self.scale1 = dataloader.load_images(self.W, self.H)
        self.eval_info = dataloader.get_eval_info(self.eval_W, self.eval_H)
        self.instance_name = f'{dataloader.scene_name}_{dataloader.image_name0}_{dataloader.image_name1}'

    def name(self):
        return "SgadAreaMatcher"

    def init_area_matcher(self):
        """
        """
        AM_config = {
            "datasetName": self.datasetName,
            "out_path": self.out_path,
            "level_num": self.level_num,
            "level_step": self.level_step,
            "W": self.W,
            "H": self.H,
            "show_flag": self.draw_verbose,
            "hcrf": self.hcrf,
            "hcrf_overlap_threshold": self.hcrf_overlap_threshold,
            "hcrf_coverage_threshold": self.hcrf_coverage_threshold,
            "weights": self.weights,
            "model_variant": self.model_variant,
            "pretrained_root": self.pretrained_root,
            "pred_score_threshold": self.pred_score_threshold,
            "areagraph_overrides": self.areagraph_overrides,
        }

        from .AGMatcherFree import AGMatcherF
        if getattr(self, "area_matcher", None) is None or not isinstance(self.area_matcher, AGMatcherF):
            self.area_matcher = AGMatcherF(configs=AM_config)
        else:
            # Reuse the backend matcher (heavy model weights) across pairs.
            # Per-pair outputs are still separated via `base_out_path` updated here and `path_loader(...)`.
            self.area_matcher.datasetName = self.datasetName
            self.area_matcher.level_num = self.level_num
            self.area_matcher.level_step = self.level_step
            self.area_matcher.W = self.W
            self.area_matcher.H = self.H
            self.area_matcher.show_flag = int(self.draw_verbose)
            self.area_matcher.hcrf = self.hcrf
            self.area_matcher.hcrf_overlap_threshold = self.hcrf_overlap_threshold
            self.area_matcher.hcrf_coverage_threshold = self.hcrf_coverage_threshold
            self.area_matcher.weights = self.weights
            self.area_matcher.model_variant = self.model_variant
            self.area_matcher.pretrained_root = self.pretrained_root
            self.area_matcher.pred_score_threshold = self.pred_score_threshold
            self.area_matcher.areagraph_overrides = self.areagraph_overrides or {}
            self.area_matcher.base_out_path = self.out_path
            self.area_matcher.out_path = self.out_path

    def area_matching(self, dataloader, out_path):
        """
        """
        logger.info(f"start area matching")

        self.init_dataloader(dataloader)
        self.set_outpath(out_path)

        self.init_area_matcher()
        self.area_matcher.path_loader(
            self.image0_path,
            self.bbox_path0,
            self.image1_path,
            self.bbox_path1,
            self.name0,
            self.name1,
        )
        self.area_matcher.init_area_matcher()
        self.area_matcher.img_areagraph_construct(efficient=True)

        if self.datasetName == "ScanNet":
            ag0 = self.area_matcher.areagraph0
            ag1 = self.area_matcher.areagraph1
            
            area_info_list0 = []
            for node in ag0.AGNodes:
                area_info = {'area_bbox': node.area, 'area_center': node.center}
                area_info_list0.append(area_info)
            
            area_info_list1 = []
            for node in ag1.AGNodes:
                area_info = {'area_bbox': node.area, 'area_center': node.center}
                area_info_list1.append(area_info)
                
        elif self.datasetName == "MegaDepth":
            area_info_list0 = self.area_matcher.areagraph0.area_info_list
            area_info_list1 = self.area_matcher.areagraph1.area_info_list

        logger.info(
            "SGAD region input: dataset={}, graph_completion={}, preprocessed={}/{}, input={}/{}",
            self.datasetName, self.datasetName == "ScanNet",
            len(self.area_matcher.areagraph0.area_info_list),
            len(self.area_matcher.areagraph1.area_info_list),
            len(area_info_list0), len(area_info_list1),
        )
        
        ori_img0 = self.area_matcher.areagraph0.ori_img
        ori_img1 = self.area_matcher.areagraph1.ori_img

        import time
        area_match_srcs, area_match_dsts, area_matcher_time, merge_contain_time = self.area_matcher.area_matcher.area_match(area_info_list0, area_info_list1, ori_img0, ori_img1)
        with open(os.path.join(out_path, "area_matcher_time.txt"), 'a') as f:
            f.write(f"{area_matcher_time:.4f}\n")

        with open(os.path.join(out_path, "merge_contain_time.txt"), 'a') as f:
            f.write(f"{merge_contain_time:.4f}\n")

        self.area_match_srcs = area_match_srcs
        self.area_match_dsts = area_match_dsts

        if self.draw_verbose:
            flag = draw_matched_area_list(self.img0, self.img1, area_match_srcs, area_match_dsts, self.out_path, self.name0, self.name1, self.draw_verbose)

            if not flag:
                logger.critical(f"Something wrong with area matching, please check the code for {self.out_path.split('/')[-1]}")

            # draw each area's match
            for i, src_area in enumerate(area_match_srcs):
                dst_area = area_match_dsts[i]
                draw_matched_area(self.img0, self.img1, src_area, dst_area, (0,255,0), self.out_path, f"{i}_"+self.name0, self.name1, self.draw_verbose)
                
        logger.success(f"finish area matching")


        return area_match_srcs, area_match_dsts

                
def tune_all_area_matches_to_crop_from_size(matched_area0s, matched_area1s):
    """ Load all area matches to crop-from-size
    Args:
        self.all_area0_matches, self.all_area1_matches in the crop-from-size
    """
    assert matched_area0s is not None, f"all_area0_matches is None"
    assert matched_area1s is not None, f"all_area1_matches is None"
    # assert len(matched_area0s) == len(matched_area1s), f"len(all_area0_matches) != len(all_area1_matches)"

    # if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
    #     self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W0, self.crop_from_size_H0)
    #     self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W1, self.crop_from_size_H1)
    # elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
    #     self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
    #     self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
    # else:
    #     raise NotImplementedError
    

    all_input_matched_area0s = tune_area_list_size(matched_area0s, 1296, 968, 640, 480)
    all_input_matched_area1s = tune_area_list_size(matched_area1s, 1296, 968, 640, 480)

    return all_input_matched_area0s, all_input_matched_area1s

def tune_area_list_size(area_list, src_W, src_H, dst_W, dst_H):
    """
    """
    rt_area_list = []

    for area in area_list:
        area_ = tune_area_size(area, src_W, src_H, dst_W, dst_H)
        rt_area_list.append(area_)
    
    return rt_area_list

def tune_area_size(area, src_W, src_H, dst_W, dst_H):
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
