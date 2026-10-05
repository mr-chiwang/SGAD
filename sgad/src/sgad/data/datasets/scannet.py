# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR ScanNet dataset loading for SGAD region matching.

from os import path as osp
from typing import Dict, Set
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
from loguru import logger
from sgad.data.bbox_filtering import process_bboxes_for_training
from sgad.data.bbox_io import (
    read_bboxes_from_file,
    infer_bbox_path,
    load_pair_bbox_counts,
    resize_bboxes_to_image,
    resolve_bbox_dataset_root,
)
from sgad.data.scannet_io import (
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_rgb,
)

# Known corrupted depth frames: {(scene_id, sub_id): {frame_id, ...}}
# Format derived from: scene0474_02/depth/938.png -> (474, 2): {938}
CORRUPTED_DEPTH_FRAMES = {
    (474, 2): {938},
}


class ScanNetDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 bbox_root=None,
                 min_bbox_area=2000,
                 min_bbox_wh=28,
                 min_bbox_count=3,
                 bbox_enable_filter=True,
                 bbox_enable_expand=True,
                 bbox_enable_random_gen=True,
                 pair_bbox_counts_path=None,
                 min_pair_bbox_count=0,
                 **kwargs):
        """Manage the image pairs and metadata for one ScanNet scene.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): Reserved training augmentation; ``__getitem__`` currently disables it.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
            bbox_root (str): Root directory for precomputed bboxes.
            min_bbox_area (float): Minimum bbox area threshold for filtering.
            min_bbox_wh (float): Minimum bbox width/height threshold for filtering.
            min_bbox_count (int): Minimum number of bboxes required per image.
            bbox_enable_filter (bool): Whether to filter bboxes by area/wh.
            bbox_enable_expand (bool): Whether to expand rejected bboxes.
            bbox_enable_random_gen (bool): Whether to randomly generate bboxes if count is insufficient.
            pair_bbox_counts_path (str): Per-pair bbox-count sidecar aligned with the LoFTR index.
            min_pair_bbox_count (int): Minimum raw bbox count required in both images of a pair.
        """
        super().__init__()
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.bbox_root = resolve_bbox_dataset_root(root_dir, bbox_root)
        self.mode = mode
        self.img_resize = (640, 480) if 'img_resize' not in kwargs else kwargs['img_resize']

        # Region-filtering parameters.
        self.min_bbox_area = min_bbox_area
        self.min_bbox_wh = min_bbox_wh
        self.min_bbox_count = min_bbox_count
        self.bbox_enable_filter = bbox_enable_filter
        self.bbox_enable_expand = bbox_enable_expand
        self.bbox_enable_random_gen = bbox_enable_random_gen

        # Load image-pair metadata and apply the training overlap threshold.
        self.scene_name = osp.splitext(osp.basename(npz_path))[0]
        with np.load(npz_path) as data:
            data_names = data['name']
            kept_mask = np.ones(len(data_names), dtype=bool)
            if pair_bbox_counts_path is not None:
                pair_bbox_counts = load_pair_bbox_counts(pair_bbox_counts_path, len(data_names))
                kept_mask &= pair_bbox_counts.min(axis=1) >= min_pair_bbox_count
            elif min_pair_bbox_count > 0:
                raise ValueError("pair_bbox_counts_path is required when min_pair_bbox_count is positive")
            if 'score' in data.keys() and mode == 'train':
                kept_mask &= data['score'] > min_overlap_score
            self.data_names = data_names[kept_mask]

        # Filter out pairs involving known corrupted depth frames.
        original_count = len(self.data_names)
        self.data_names = self._filter_corrupted_pairs(self.data_names)
        filtered_count = original_count - len(self.data_names)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} pairs with corrupted depth from {self.scene_name}")

        self.intrinsics = dict(np.load(intrinsic_path))

        self.augment_fn = augment_fn if mode == 'train' else None

    def _filter_corrupted_pairs(self, data_names):
        """Filter out image pairs that involve corrupted depth frames."""
        if not CORRUPTED_DEPTH_FRAMES:
            return data_names

        valid_mask = np.ones(len(data_names), dtype=bool)
        for i, entry in enumerate(data_names):
            scene_id, sub_id, frame0, frame1 = entry
            corrupted_frames = CORRUPTED_DEPTH_FRAMES.get((scene_id, sub_id), set())
            if frame0 in corrupted_frames or frame1 in corrupted_frames:
                valid_mask[i] = False

        return data_names[valid_mask]


    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.pose_dir,
                       scene_name,
                       'pose', f'{name}.txt')
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        
        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # Resolve image paths.
        img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')

        # Load precomputed regions.
        bbox_base0 = infer_bbox_path(img_name0, self.root_dir, self.bbox_root)
        bbox_base1 = infer_bbox_path(img_name1, self.root_dir, self.bbox_root)
        bboxes0 = read_bboxes_from_file(bbox_base0)
        bboxes1 = read_bboxes_from_file(bbox_base1)

        image0, mask0, scale0, image_shape0, orig_shape0 = read_scannet_rgb(img_name0, resize=self.img_resize, augment_fn=None)
        image1, mask1, scale1, image_shape1, orig_shape1 = read_scannet_rgb(img_name1, resize=self.img_resize, augment_fn=None)

        # Resize regions to match the resized images.
        bboxes0 = resize_bboxes_to_image(bboxes0, orig_shape0, image_shape0)
        bboxes1 = resize_bboxes_to_image(bboxes1, orig_shape1, image_shape1)

        # Filter regions and, when configured, expand or synthesize replacements.
        image_h, image_w = image_shape0[0].item(), image_shape0[1].item()
        bboxes0, _ = process_bboxes_for_training(
            bboxes0, image_h, image_w,
            self.min_bbox_area, self.min_bbox_wh, self.min_bbox_count,
            enable_filter=self.bbox_enable_filter,
            enable_expand=self.bbox_enable_expand,
            enable_random_gen=self.bbox_enable_random_gen
        )
        image_h, image_w = image_shape1[0].item(), image_shape1[1].item()
        bboxes1, _ = process_bboxes_for_training(
            bboxes1, image_h, image_w,
            self.min_bbox_area, self.min_bbox_wh, self.min_bbox_count,
            enable_filter=self.bbox_enable_filter,
            enable_expand=self.bbox_enable_expand,
            enable_random_gen=self.bbox_enable_random_gen
        )
                                
        
        depth_path0 = osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png')
        depth_path1 = osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png')
        depth0 = read_scannet_depth(depth_path0)
        depth1 = read_scannet_depth(depth_path1)

        # Read the depth-camera intrinsics.
        K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

        # Compute relative camera poses.
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,   # (3, h, w)
            'image_shape0': image_shape0,   # (h, w)
            'depth0': depth0,   # (h, w)
            'original_mask0': mask0,
            'bboxes0': bboxes0,
            'image1': image1,
            'image_shape1': image_shape1,   # (h, w)
            'depth1': depth1,   # (h, w)
            'original_mask1': mask1,
            'bboxes1': bboxes1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'ScanNet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg')),
            'root_dir': self.root_dir
        }

        return data
