# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR MegaDepth I/O for SGAD region matching.

import h5py
import numpy as np
import torch

from sgad.data import image_io
from sgad.data.image_io import (
    imread_rgb,
    load_array_from_s3,
    pad_bottom_right,
)


def read_megadepth_rgb(path, resize=None, df=None, padding=False, augment_fn=None, geometric_aug=None):
    client = image_io.MEGADEPTH_CLIENT
    image = imread_rgb(path, augment_fn, client=client)
    H_mat = None
    if geometric_aug is not None:
        image = torch.from_numpy(image).float() / 255
        image, H_mat = geometric_aug(image[None, None])
        image = image[0, 0].numpy() * 255

    w, h = image.shape[1], image.shape[0]
    scale = torch.tensor([1, 1], dtype=torch.float)
    image_shape = torch.tensor([h, w], dtype=torch.int16)

    if padding:
        pad_to = resize
        image = image.transpose(2, 0, 1)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)[0]
    else:
        mask = torch.ones((h, w), dtype=torch.float)

    return image, mask, scale, H_mat, image_shape


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith("s3://"):
        client = image_io.MEGADEPTH_CLIENT
        depth = load_array_from_s3(path, client, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, "r")["depth"])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()
    return depth
