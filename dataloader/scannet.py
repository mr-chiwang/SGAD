import os

from .abstract_dataloader import AbstractDataloader
from utils.load import (
    load_cv_depth,
    load_K_txt,
    load_pose_txt,
)

from utils.geo import tune_corrs_size

class ScanNetDataloader(AbstractDataloader):
    """ dataloader for ScanNet dataset 
    """

    def __init__(self, 
        root_path, 
        scene_name, 
        image_name0, 
        image_name1,
        color_folder,
        color_post,
        depth_folder,
        depth_post,
        depth_factor,
        K_folder,
        pose_folder,
        pose_post,
        bbox_root,
        bbox_post,
        ) -> None:
        super().__init__(root_path, scene_name, image_name0, image_name1)

        self._name = 'ScanNetDataLoader'

        self.color_folder = color_folder
        self.color_post = color_post
        self.depth_folder = depth_folder
        self.depth_post = depth_post
        self.depth_factor = depth_factor
        self.K_folder = K_folder
        self.pose_folder = pose_folder
        self.pose_post = pose_post
        self.bbox_root = bbox_root
        self.bbox_post = bbox_post

        self._path_assemble()

    def reset_imgs(self, scene_name, image_name0, image_name1):
        """
        """
        self.scene_name = scene_name
        self.image_name0 = image_name0
        self.image_name1 = image_name1
        self._path_assemble()
    
    def _path_assemble(self):
        """ assemble the paths
        """
        # color image
        self.img0_path = os.path.join(self.root_path, self.scene_name, self.color_folder, f"{self.image_name0}.{self.color_post}")
        self.img1_path = os.path.join(self.root_path, self.scene_name, self.color_folder, f"{self.image_name1}.{self.color_post}")

        # depth image
        self.depth0_path = os.path.join(self.root_path, self.scene_name, self.depth_folder, f"{self.image_name0}.{self.depth_post}")
        self.depth1_path = os.path.join(self.root_path, self.scene_name, self.depth_folder, f"{self.image_name1}.{self.depth_post}")

        # intrinsic
        self.K0_path = os.path.join(self.root_path, self.scene_name, self.K_folder, f"intrinsic_color.txt")
        self.K1_path = os.path.join(self.root_path, self.scene_name, self.K_folder, f"intrinsic_color.txt")

        # pose
        self.pose0_path = os.path.join(self.root_path, self.scene_name, self.pose_folder, f"{self.image_name0}.{self.pose_post}")
        self.pose1_path = os.path.join(self.root_path, self.scene_name, self.pose_folder, f"{self.image_name1}.{self.pose_post}")

        # Precomputed bounding boxes
        self.bbox0_path = os.path.join(self.bbox_root, self.scene_name, f"{self.image_name0}.{self.bbox_post}")
        self.bbox1_path = os.path.join(self.bbox_root, self.scene_name, f"{self.image_name1}.{self.bbox_post}")
             
    def load_Ks(self, scale0, scale1):
        """ load Ks
        Returns:
            K0, K1: np.mat, 3x3
        """
        K0 = load_K_txt(self.K0_path, scale0)
        K1 = load_K_txt(self.K1_path, scale1)
        return K0, K1

    def load_poses(self):
        """ load poses
        Returns:
            P0, P1: np.mat, 4x4
        """
        P0 = load_pose_txt(self.pose0_path)
        P1 = load_pose_txt(self.pose1_path)
        return P0, P1
    
    def get_depth_factor(self):
        return self.depth_factor
    
    def load_depths(self):
        """
        """
        depth0 = load_cv_depth(self.depth0_path)
        depth1 = load_cv_depth(self.depth1_path)
        return depth0, depth1
    
    def get_bbox_paths(self):
        """Return the precomputed bbox paths for the image pair."""
        return self.bbox0_path, self.bbox1_path
    
    def get_eval_info(self, eval_W, eval_H):
        """ for evaluation
        """
        eval_info = {}
        image0, image1, scale0, scale1 = self.load_images(eval_W, eval_H)
        K0, K1 = self.load_Ks(scale0, scale1)
        P0, P1 = self.load_poses()
        depth0, depth1 = self.load_depths()
        eval_info["dataset_name"] = 'ScanNet'
        eval_info["image0"] = image0
        eval_info["image1"] = image1
        eval_info["K0"] = K0
        eval_info["K1"] = K1
        eval_info["P0"] = P0
        eval_info["P1"] = P1
        eval_info["depth0"] = depth0
        eval_info["depth1"] = depth1
        eval_info["depth_factor"] = self.depth_factor
        return eval_info
        
    # specific for PMer
    def tune_corrs_size_to_eval(self, corrs, match_W, match_H, eval_W, eval_H):
        """
        """
        eval_corrs = tune_corrs_size(corrs, match_W, match_H, eval_W, eval_H)
        return eval_corrs
