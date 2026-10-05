import abc
import numpy as np
from typing import List, Optional, Any, Tuple
import cv2

from utils.load import load_cv_img_resize

class AbstractDataloader(abc.ABC):
    """
    Abstract class for dataloader
    """

    def __init__(self,
        root_path,
        scene_name,
        image_name0,
        image_name1,
        ) -> None:
        super().__init__()
        self._name = 'AbstractDataloader'
        self.root_path = root_path
        self.scene_name = scene_name
        self.image_name0 = str(image_name0)
        self.image_name1 = str(image_name1)

        # self.paths init
        self.img0_path = None
        self.img1_path = None

        # depth info
        self.depth0_path = None
        self.depth1_path = None

        # Precomputed region-box paths
        self.bbox0_path = None
        self.bbox1_path = None

        # geo info
        self.K0_path = None
        self.K1_path = None
        self.pose0_path = None
        self.pose1_path = None

    def name(self) -> str:
        """
        Return the name of the dataset
        """
        return self._name

    @abc.abstractmethod
    def _path_assemble(self):
        """Assemble the paths
        Returns:
            assembled path to self
        """
        pass

    def load_images(self, W=None, H=None, PMer=False):
        """ load images
        """
        assert not PMer, 'Error: PMer is not supported in this dataloader'
        if W is None or H is None:
            # load as original size
            img0 = cv2.imread(self.img0_path, cv2.IMREAD_COLOR)
            img1 = cv2.imread(self.img1_path, cv2.IMREAD_COLOR)
            scale0 = 1
            scale1 = 1
        else:
            img0, scale0 = load_cv_img_resize(self.img0_path, W, H, 1)
            img1, scale1 = load_cv_img_resize(self.img1_path, W, H, 1)

        return img0, img1, scale0, scale1


    @abc.abstractmethod
    def load_Ks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Ks from self path
        Returns:
            K0, K1: np.ndarray 3x3
        """
        pass

    @abc.abstractmethod
    def load_depths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load depth from self path
        """
        pass

    @abc.abstractmethod
    def load_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pose from self path
        """
        pass

    @abc.abstractmethod
    def get_eval_info(self):
        """Return eval info as dict
        Returns:
            eval_info: dict
                - dataset_name: str
                - image0, image1: np.ndarray HxWx3
                - K0, K1: np.ndarray 3x3
                - P0, P1: np.ndarray 4x4
                - optional: depth_factor, depth0, depth1
        """
        pass

    @abc.abstractmethod
    def get_bbox_paths(self) -> Tuple[str, str]:
        """
        Return precomputed region-box paths.
        """
        pass

    @abc.abstractmethod
    def tune_corrs_size_to_eval(self, corrs, match_W, match_H, eval_W, eval_H):
        """
        Tune the corrs size to eval size
        """
        pass
