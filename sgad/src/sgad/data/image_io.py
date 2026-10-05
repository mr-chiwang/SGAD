# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR image and depth I/O for SGAD region matching.

import io

import cv2
import h5py
import numpy as np
from loguru import logger


MEGADEPTH_CLIENT = None
SCANNET_CLIENT = None


def load_array_from_s3(path, client, cv_type, use_h5py=False):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, "r")["/depth"])
    except Exception as ex:
        logger.error(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_rgb(path, augment_fn=None, client=None):
    client = SCANNET_CLIENT if client is None else client
    cv_type = cv2.IMREAD_COLOR
    if str(path).startswith("s3://"):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augment_fn is not None:
        image = augment_fn(image)
    return image


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        if inp.shape[0] != 3:
            inp = inp.transpose(2, 0, 1)
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask
