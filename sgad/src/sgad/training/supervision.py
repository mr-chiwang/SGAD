# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR geometric supervision for SGAD region matching.

from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid

from sgad.geometry.warping import warp_kpts


# Geometry-derived point correspondences for region-overlap supervision

@torch.no_grad()
def spvs_bbox(data, config):
    """Build geometric point correspondences used for region supervision.

    Grid points are projected bidirectionally using depth, pose, and camera
    intrinsics. Round-trip mutual nearest neighbours are retained and stored in
    ``data['spv_bbox_matches']`` as one ``[M, 4]`` image-coordinate tensor per
    sample, with rows ``[x0, y0, x1, y1]``.
    """
        
    # 1. Build image- and region-grid scale metadata.
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['MODEL']['RESOLUTION_BBOX']
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. Create region-grid points and map them to image coordinates.
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # Map padded grid locations to (0, 0) before geometric projection.
    if 'bbox_mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['bbox_mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['bbox_mask1'])

    # Project points bidirectionally and map them back to region-grid coordinates.
    # No cross-view depth-consistency check is applied.
    # Zero-depth samples are not masked before projection.
    w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['T_0to1'], data['K0'], data['K1'])
    w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. Retain round-trip mutual nearest neighbours.
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # Map out-of-bounds projections to sentinel index 0.
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # Reserve source index 0 as the invalid sentinel.

    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
    
    # 4. Store image-coordinate correspondences for overlap supervision.
    matched_points = []
    for b in range(N):  # N is the batch size.
        mask = b_ids == b
        points_img0 = grid_pt0_i[b, i_ids[mask]]  # Image-0 coordinates
        points_img1 = grid_pt1_i[b, j_ids[mask]]  # Image-1 coordinates
        matched = torch.cat([points_img0, points_img1], dim=1)
        matched_points.append(matched)

    data.update({
        'spv_bbox_matches': matched_points
    })

    
    
def compute_supervision_bbox(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_bbox(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """Map padded region-grid locations to the projection sentinel ``(0, 0)``."""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt
