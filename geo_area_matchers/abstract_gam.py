import numpy as np
import abc
import os

from utils.common import test_dir_if_not_create

class AbstractGeoAreaMatcher(abc.ABC):
    def __init__(self) -> None:
        self.initialized = False
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def init_dataloader(self, dataloader):
        raise NotImplementedError

    @abc.abstractmethod
    def load_point_matcher(self, point_matcher):
        raise NotImplementedError

    @abc.abstractmethod
    def load_ori_corrs(self, ori_corrs):
        raise NotImplementedError

    @abc.abstractmethod
    def init_gam(self):
        raise NotImplementedError

    @abc.abstractmethod
    def geo_area_matching_refine(self, matched_areas0, matched_areas1):
        """ Main Func
        Returns:
            alpha_corrs_dict: dict, inside-area corrs under each alpha
            alpha_inlier_idxs_dict: dict, inlier idxs of input areas under each alpha
            alpha_sampled_corrs: optional dict of sampled corrs under each alpha
            time_pm: seconds spent in point-matcher calls during this refinement
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doubtful_area_match_predict(self, doubt_match_pairs):
        """
        """
        raise NotImplementedError


    def set_outpath(self, outpath: str):
        """Run after init_dataloader 
        """
        self.out_path = os.path.join(outpath, f"{self.scene_name}_{self.name0}_{self.name1}", 'gam')
        if self.draw_verbose == 1:
            test_dir_if_not_create(self.out_path)
