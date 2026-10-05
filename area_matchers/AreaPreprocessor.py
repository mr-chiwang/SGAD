
import sys
sys.path.append("..")

import numpy as np 
from loguru import logger
import copy

from .AGConfig import preprocess_configs

"""Convert precomputed region boxes into metadata for AreaGraph."""
class AreaPreprocesser(object):
    """Prepare numeric XYWH bounding boxes for area-graph construction.

    Inputs are converted to dictionaries containing ``area_bbox``, ``area_center``,
    and ``area_size``. The current ``refine_bbox`` step preserves loaded regions
    unchanged. The active filtering step fuses small or slender regions and removes
    near-duplicate boxes.
    """
    default_configs = preprocess_configs

    def __init__(self, configs) -> None:
        """
        Args:
            self.min_area_size
            self.WH_ratio_thd
        """
        configs = {**self.default_configs, **configs}

        self.min_area_size = configs["min_area_size"]
        self.max_wh_ratio= configs["max_wh_ratio"] # slim thd
        self.topK = configs["topK"]
        self.min_dist_thd = configs["min_dist_thd"]
        self.refined_areas = None

    def load(self, res_path="", res=None):
        """
        """
        self._load_sam(sam_results_path=res_path, sam_res=res)

    def _load_sam(self, sam_results_path="", sam_res=None):
        """Load XYWH boxes and derive the geometry used by AreaGraph.
        Args:
            sam_results_path
        Returns:
            self.sam_results: list of sam results
        """
        self.sam_results = []    
        if sam_results_path != "":
            sam_res = np.load(sam_results_path, allow_pickle=False)
        elif sam_res is None:
            logger.error("No sam results path or sam results!")
            return

        for bbox in sam_res:
            area_info = {}
            area_info["area_bbox"] = self._bbox2area(bbox)
            area_info["area_center"] = [
                int(bbox[0] + bbox[2] / 2),
                int(bbox[1] + bbox[3] / 2),
            ]
            area_info["area_size"] = bbox[2] * bbox[3]
            self.sam_results.append(area_info)

        logger.info(f"Loaded bounding boxes with size: {len(self.sam_results)}")

        return self.sam_results

    def _bbox2area(self, bbox):
        """
        Args:
            bbox: [uvwh]
        Returns:
            area: [umin,umax,vmin,vmax]
        """
        area = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]

        return area
    
    def refine_bbox(self):
        """Preserve loaded bounding-box regions."""
        # for i, area_info in enumerate(self.sam_results):
        #     area_bbox = area_info["area_bbox"]
        #     area_w = area_bbox[1] - area_bbox[0]
        #     area_h = area_bbox[3] - area_bbox[2]
        #     wh_ratio = max(area_w, area_h) / (min(area_w, area_h)+1e-6)
        #     area_size = area_w * area_h
        #     if area_size > self.min_area_size and wh_ratio < self.max_wh_ratio and area_w > 28 and area_h > 28:
        #         refined_areas.append(area_info)
            
        refined_areas = self.sam_results
        
        self.refined_areas = refined_areas

        # # filter repeat areas
        # self.refined_areas = self.filter_repeat_areas(self.refined_areas)

        return self.refined_areas

    def filter_repeat_areas(self, area_list, dist_thd=50):
        """ filter areas with similar bbox located point
        """
        area_list_filtered = copy.deepcopy(area_list)
        pop_list = []
        for i, area_info in enumerate(area_list):
            area_bbox = area_info["area_bbox"]
            pop_flag = False
            for j, area_info_j in enumerate(area_list):
                if i == j:
                    continue
                area_bbox_j = area_info_j["area_bbox"]
                # calc the mean distance between the points of bbox
                dist_ul = np.sqrt(np.sum(np.square(np.array([area_bbox[0], area_bbox[2]]) - np.array([area_bbox_j[0], area_bbox_j[2]]))))
                dist_rd = np.sqrt(np.sum(np.square(np.array([area_bbox[1], area_bbox[3]]) - np.array([area_bbox_j[1], area_bbox_j[3]]))))
                dist_mean = (dist_ul + dist_rd) / 2
                if dist_mean < dist_thd:
                    if area_info["area_size"] < area_info_j["area_size"]:
                        pop_flag = True
                    elif area_info["area_size"] == area_info_j["area_size"]:
                        if i > j:
                            pop_flag = True
            if pop_flag:
                pop_list.append(i)
        
        for i in pop_list[::-1]:
            area_list_filtered.pop(i)
        

        return area_list_filtered

    def filter_abnormal_areas(self):
        """ filter the abnormal areas
        Args:
            self.refined_areas
        Returns:
            self.filtered_areas
        """
        topK = self.topK
        min_dist_thd = self.min_dist_thd
        small_flag = True
        slim_flag = True
        while small_flag or slim_flag:
            if small_flag:
                small_idx_list = self._check_small_areas(self.refined_areas)

                self.refined_areas = self.fusion_areas(self.refined_areas, small_idx_list, topK, min_dist_thd)
            
            if slim_flag:
                slim_idx_list = self._check_slim_areas(self.refined_areas)

                self.refined_areas = self.fusion_areas(self.refined_areas, slim_idx_list, topK, min_dist_thd)

            if len(self._check_slim_areas(self.refined_areas)) == 0:
                slim_flag = False
            if len(self._check_small_areas(self.refined_areas)) == 0:
                small_flag = False

            # break
        
        # filter repeat areas
        self.refined_areas = self.filter_repeat_areas(self.refined_areas)

        self.filtered_areas = self.refined_areas

        return self.refined_areas
    
    def _check_slim_areas(self, area_infos):
        """
        Args:
            area_info
        Returns:
            idx_list    
        """
        idx_list = []
        for i, area_info in enumerate(area_infos):
            area_w = area_info["area_bbox"][1] - area_info["area_bbox"][0]
            area_h = area_info["area_bbox"][3] - area_info["area_bbox"][2]
            wh_ratio = max(area_w, area_h) / (min(area_w, area_h)+1e-6)
            if wh_ratio > self.max_wh_ratio or area_w < 28 or area_h < 28:
                idx_list.append(i)
                break
        
        return idx_list

    def _check_small_areas(self, area_infos):
        """ check the small areas, return small area idx list
        Args:
            area_infos
        Returns:
            idx_list
        """
        idx_list = []
        for i, area_info in enumerate(area_infos):
            if area_info["area_size"] < self.min_area_size:
                idx_list.append(i)
                break

        return idx_list

    def fusion_areas(self, src_area_info_list, await_area_idx_list, topK=3, min_dist_thd=100):
        """ fusion the areas, take the areas from list accroding to idx list
            fuse to area in left areas 
        """
        for idx in await_area_idx_list:
            await_fuse_area = copy.deepcopy(src_area_info_list[idx])
            # remove the await_fuse_area from src_area_info_list
            src_area_info_list.pop(idx)
            src_area_info_list = self._fuse_single_area_to_list(await_fuse_area, src_area_info_list)

        return src_area_info_list

    def _fuse_single_area_to_list(self, await_fuse_area, area_info_list, topK=3, min_dist_thd=100):
        """ find the target and fuse
        Args:

        Returns:
            modified area_info_list
        """
        fused_area_info_list = copy.deepcopy(area_info_list)
        await_fuse_center = await_fuse_area["area_center"]
        # calc the distance matrix between await_fuse_area and area_info_list
        dist_matrix = []
        for i, area_info in enumerate(area_info_list):
            dist = np.sqrt((await_fuse_center[0] - area_info["area_center"][0])**2 + (await_fuse_center[1] - area_info["area_center"][1])**2)
            dist_matrix.append({i: dist})

        if len(dist_matrix) == 0:
            logger.info("No fuse target")
            return fused_area_info_list

        max_sim_idx = -1
        max_sim = 0
        
        # sort the dist_matrix by dist
        dist_matrix.sort(key=lambda x: list(x.values())[0])
        if list(dist_matrix[0].values())[0] >= min_dist_thd: 
            logger.info("No fuse target")
            return fused_area_info_list

        # get the topK closest area size similarity
        topK = min(topK, len(dist_matrix))
        for i in range(topK):
            try:
                area_sim = (self._calc_area_similarity(await_fuse_area, area_info_list[list(dist_matrix[i].keys())[0]]))
            except Exception as e:
                logger.exception(e)

            if area_sim > max_sim and list(dist_matrix[i].values())[0] < 0.7*min_dist_thd:
                max_sim_idx = list(dist_matrix[i].keys())[0]
                max_sim = area_sim

        # Fuse the bounding-box geometry.
        if max_sim_idx == -1:
            logger.info("No fuse target")
            return fused_area_info_list
        fused_area_info = {}

        # fuse bbox
        fused_bbox = [min(await_fuse_area["area_bbox"][0], area_info_list[max_sim_idx]["area_bbox"][0]), max(await_fuse_area["area_bbox"][1], area_info_list[max_sim_idx]["area_bbox"][1]), min(await_fuse_area["area_bbox"][2], area_info_list[max_sim_idx]["area_bbox"][2]), max(await_fuse_area["area_bbox"][3], area_info_list[max_sim_idx]["area_bbox"][3])]
        fused_area_info["area_bbox"] = fused_bbox

        # fuse center
        fused_center = [(fused_bbox[0] + fused_bbox[1])//2, (fused_bbox[2] + fused_bbox[3])//2]
        fused_area_info["area_center"] = fused_center

        # fuse size
        fused_size = (fused_bbox[1] - fused_bbox[0]) * (fused_bbox[3] - fused_bbox[2])
        fused_area_info["area_size"] = fused_size
        
        fused_area_info_list[max_sim_idx] = fused_area_info

        return fused_area_info_list

    def _calc_area_similarity(self, area_info_1, area_info_2):
        """ calc the similarity of area size between two areas
        Returns:
            area_size_similarity: bigger is better
        """
        area_size_1 = area_info_1["area_size"]
        area_size_2 = area_info_2["area_size"]
        area_size_similarity = min(area_size_1, area_size_2) / (max(area_size_1, area_size_2) + 1e-6)
        return area_size_similarity
