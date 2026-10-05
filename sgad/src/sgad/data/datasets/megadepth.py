# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR MegaDepth dataset loading for SGAD region matching.

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from sgad.data.bbox_filtering import process_bboxes_for_training
from sgad.data.bbox_io import (
    read_bboxes_from_file,
    infer_bbox_path,
    load_pair_bbox_counts,
    resolve_bbox_dataset_root,
)
from sgad.data.megadepth_io import (
    read_megadepth_depth,
    read_megadepth_rgb,
)

class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
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
        """Manage the image pairs and metadata for one MegaDepth scene.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): Square image padding size when ``img_padding`` is enabled.
            df (int, optional): Reserved image-size division factor; the current RGB reader does not apply it.
            img_padding (bool): Whether to zero-pad images to ``img_resize`` square canvases.
            depth_padding (bool): Whether to zero-pad depth maps to 2000 x 2000.
            augment_fn (callable, optional): Photometric augmentation applied to image 1 during training.
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
        self.mode = mode
        self.scene_id = osp.splitext(str(npz_path))[0]
        self.bbox_root = resolve_bbox_dataset_root(root_dir, bbox_root)

        # Region-filtering parameters.
        self.min_bbox_area = min_bbox_area
        self.min_bbox_wh = min_bbox_wh
        self.min_bbox_count = min_bbox_count
        self.bbox_enable_filter = bbox_enable_filter
        self.bbox_enable_expand = bbox_enable_expand
        self.bbox_enable_random_gen = bbox_enable_random_gen

        # Load scene metadata and retain pairs above the overlap threshold.
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = np.asarray(self.scene_info.pop('pair_infos'), dtype=object)
        kept_mask = np.ones(len(self.pair_infos), dtype=bool)
        if pair_bbox_counts_path is not None:
            pair_bbox_counts = load_pair_bbox_counts(pair_bbox_counts_path, len(self.pair_infos))
            kept_mask &= pair_bbox_counts.min(axis=1) >= min_pair_bbox_count
        elif min_pair_bbox_count > 0:
            raise ValueError("pair_bbox_counts_path is required when min_pair_bbox_count is positive")
        kept_mask &= np.asarray([pair_info[1] > min_overlap_score for pair_info in self.pair_infos])
        self.pair_infos = self.pair_infos[kept_mask]

        # Configure image and depth padding.
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        if mode == 'val':
            # self.img_resize = 864
            self.img_resize = 2000
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # Square depth-map padding size.

        self.augment_fn = augment_fn if mode == 'train' else None
        # Downsampling factor corresponding to RESOLUTION_BBOX.
        self.bbox_scale = kwargs.get('bbox_scale', 0.5)
        
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # Resolve image paths.
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])

        # Load precomputed regions.
        bbox_base0 = infer_bbox_path(img_name0, self.root_dir, self.bbox_root)
        bbox_base1 = infer_bbox_path(img_name1, self.root_dir, self.bbox_root)
        bboxes0 = read_bboxes_from_file(bbox_base0)
        bboxes1 = read_bboxes_from_file(bbox_base1)

        # TODO: Apply paired augmentation consistently and seed it per worker.
        image0, mask0, scale0, _, image_shape0 = read_megadepth_rgb(
            img_name0, self.img_resize, self.df, self.img_padding)
        image1, mask1, scale1, _, image_shape1 = read_megadepth_rgb(
            img_name1, self.img_resize, self.df, self.img_padding, self.augment_fn)

        # Filter regions and, when configured, expand or synthesize replacements.
        image_h0, image_w0 = image_shape0[0].item(), image_shape0[1].item()
        bboxes0, _ = process_bboxes_for_training(
            bboxes0, image_h0, image_w0,
            self.min_bbox_area, self.min_bbox_wh, self.min_bbox_count,
            enable_filter=self.bbox_enable_filter,
            enable_expand=self.bbox_enable_expand,
            enable_random_gen=self.bbox_enable_random_gen
        )
        image_h1, image_w1 = image_shape1[0].item(), image_shape1[1].item()
        bboxes1, _ = process_bboxes_for_training(
            bboxes1, image_h1, image_w1,
            self.min_bbox_area, self.min_bbox_wh, self.min_bbox_count,
            enable_filter=self.bbox_enable_filter,
            enable_expand=self.bbox_enable_expand,
            enable_random_gen=self.bbox_enable_random_gen
        )
            

        depth0 = read_megadepth_depth(
            osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
        depth1 = read_megadepth_depth(
            osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size) # self.depth_max_size)
        
        # Read intrinsics for the original image coordinates.
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        # Compute relative camera poses.
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (3, h, w)
            'image_shape0': image_shape0,   # (h, w)
            'depth0': depth0,  # (h, w)
            'original_mask0': mask0,
            'bboxes0': bboxes0,
            'image1': image1,
            'image_shape1': image_shape1,
            'depth1': depth1,
            'original_mask1': mask1,
            'bboxes1': bboxes1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'root_dir': self.root_dir
        }

        # Downsample image-validity masks to the region-supervision resolution.
        if mask0 is not None:
            if self.bbox_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.bbox_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'bbox_mask0': ts_mask_0, 'bbox_mask1': ts_mask_1})

        return data
