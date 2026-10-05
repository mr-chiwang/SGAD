# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR ScanNet I/O for SGAD region matching.

import cv2
import numpy as np
import torch
from numpy.linalg import inv

from sgad.data import image_io
from sgad.data.image_io import (
    imread_rgb,
    load_array_from_s3,
)


def read_scannet_rgb(path, resize=(640, 480), augment_fn=None):
    image = imread_rgb(path, augment_fn)
    raw_h, raw_w = image.shape[0], image.shape[1]
    image = cv2.resize(image, resize)

    w, h = image.shape[1], image.shape[0]
    scale = torch.tensor([1, 1], dtype=torch.float)
    image_shape = torch.tensor([h, w], dtype=torch.int16)
    original_shape = torch.tensor([raw_h, raw_w], dtype=torch.int16)
    mask = torch.ones((h, w), dtype=torch.float)

    image = image.transpose(2, 0, 1)
    return image, mask, scale, image_shape, original_shape


def read_scannet_depth(path):
    if str(path).startswith("s3://"):
        client = image_io.SCANNET_CLIENT
        depth = load_array_from_s3(str(path), client, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()
    return depth


def read_scannet_pose(path):
    cam2world = np.loadtxt(path, delimiter=" ")
    world2cam = inv(cam2world)
    return world2cam
