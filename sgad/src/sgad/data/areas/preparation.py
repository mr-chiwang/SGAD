import torch

from .geometry import calculate_patch_centers
from .membership import compute_point_bbox_membership
from .regions import bbox_transforms


def normalize_images(data):
    """Convert image tensors to floating point in the ``[0, 1]`` range."""
    image0 = data['image0'].float() / 255
    image1 = data['image1'].float() / 255

    data.update({
        'image0': image0,
        'image1': image1
    })


def bbox_filter_cal_overlap(data, bboxes0, bboxes1, matched_points, point_threshold=10):
    """Filter regions by correspondence count and compute pairwise overlap.

    Args:
        data: Batch metadata containing each region format.
        bboxes0: Regions from image 0.
        bboxes1: Regions from image 1.
        matched_points: Geometric correspondences with rows
            ``[x0, y0, x1, y1]``.
        point_threshold: Minimum number of correspondences inside a region.

    Returns:
        The filtered regions and their Jaccard-overlap matrix over matched-point
        membership sets.
    """

    # Split each geometric correspondence into image-0 and image-1 coordinates.
    matched_points_tensor0 = matched_points[:, :2]  # Image-0 coordinates
    matched_points_tensor1 = matched_points[:, 2:]  # Image-1 coordinates

    # Keep regions containing at least the requested number of correspondences.
    points_in_bbox0 = compute_point_bbox_membership(matched_points_tensor0, bboxes0, data['bboxes0_format'])
    points_per_bbox0 = points_in_bbox0.sum(dim=1)
    valid_bbox_mask0 = points_per_bbox0 >= point_threshold
    filtered_bboxes0 = bboxes0[valid_bbox_mask0]
    points_in_bbox0 = points_in_bbox0[valid_bbox_mask0, :]

    points_in_bbox1 = compute_point_bbox_membership(matched_points_tensor1, bboxes1, data['bboxes1_format'])

    points_per_bbox1 = points_in_bbox1.sum(dim=1)
    valid_bbox_mask1 = points_per_bbox1 >= point_threshold

    filtered_bboxes1 = bboxes1[valid_bbox_mask1]
    points_in_bbox1 = points_in_bbox1[valid_bbox_mask1, :]

    # Measure overlap through shared geometric correspondences.
    overlap_matrix = calculate_overlap_matrix(points_in_bbox0, points_in_bbox1)

    return filtered_bboxes0, filtered_bboxes1, overlap_matrix


def calculate_overlap_matrix(points_in_bbox0, points_in_bbox1):
    points_in_bbox0 = points_in_bbox0.float()
    points_in_bbox1 = points_in_bbox1.float()

    # Count correspondences contained by both regions.
    intersection = torch.mm(points_in_bbox0, points_in_bbox1.t())

    # Count correspondences contained by either region.
    union = points_in_bbox0.sum(1).unsqueeze(1) + points_in_bbox1.sum(1).unsqueeze(0) - intersection

    # Define overlap as zero when both membership sets are empty.
    overlap_matrix = torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(intersection)
    )

    # Guard against numerical values outside the valid overlap range.
    overlap_matrix = torch.clamp(overlap_matrix, 0, 1)

    return overlap_matrix


def prepare_bboxes_label(data, max_bboxes=100, point_threshold=0, is_train=True):
    """Prepare region-overlap supervision for all data phases.

    Dataset preprocessing handles area, size, and minimum-count filtering.
    During training, this function additionally caps each image to its largest
    ``max_bboxes`` regions before computing correspondence-based overlap.
    """
    normalize_images(data)

    overlap_matrix_batch = []
    filter_bboxes0_batch = []
    filter_bboxes1_batch = []

    for i in range(data['image0'].shape[0]):
        bboxes0, bboxes1 = data['bboxes0'][i], data['bboxes1'][i]

        # During training, retain at most max_bboxes regions by xywh area.
        if is_train and len(bboxes0) > max_bboxes:
            bbox_areas0 = bboxes0[:, 2] * bboxes0[:, 3]
            indices_to_keep0 = torch.argsort(bbox_areas0, descending=True)[:max_bboxes]
            bboxes0 = bboxes0[indices_to_keep0]

        if is_train and len(bboxes1) > max_bboxes:
            bbox_areas1 = bboxes1[:, 2] * bboxes1[:, 3]
            indices_to_keep1 = torch.argsort(bbox_areas1, descending=True)[:max_bboxes]
            bboxes1 = bboxes1[indices_to_keep1]

        # Build overlap supervision from geometric correspondences.
        filter_bboxes0, filter_bboxes1, overlap_matrix = bbox_filter_cal_overlap(
            data, bboxes0, bboxes1, data['spv_bbox_matches'][i], point_threshold=point_threshold
        )

        overlap_matrix_batch.append(overlap_matrix)
        filter_bboxes0_batch.append(filter_bboxes0)
        filter_bboxes1_batch.append(filter_bboxes1)

    data.update({
        'overlap_matrix': overlap_matrix_batch,
        'bboxes0': filter_bboxes0_batch,
        'bboxes1': filter_bboxes1_batch
    })


def _prepare_bbox_centers(data):
    bboxes0_format = data['bboxes0_format']
    bboxes1_format = data['bboxes1_format']
    bboxes_img_centers0, bboxes_img_centers1 = [], []

    for i, (bboxes0, bboxes1) in enumerate(zip(data['bboxes0'], data['bboxes1'])):
        image_size0 = data['image_shape0'][i]
        image_size1 = data['image_shape1'][i]
        bboxes_img_centers0.append(
            calculate_patch_centers(bboxes0, image_size0, format=bboxes0_format)
        )
        bboxes_img_centers1.append(
            calculate_patch_centers(bboxes1, image_size1, format=bboxes1_format)
        )
    return bboxes_img_centers0, bboxes_img_centers1


def prepare_bboxes_images(
        data,
        *,
        from_img=False,
        target_width=224,
        target_height=224,
        is_normalize=True):
    """Prepare bbox centers and, for crop mode, bbox image tensors."""
    centers0, centers1 = _prepare_bbox_centers(data)

    if from_img:
        return None, None, centers0, centers1

    bbox_img0_batch, bbox_img1_batch = [], []

    bboxes0_format = data['bboxes0_format']
    bboxes1_format = data['bboxes1_format']

    for i in range(data['image0'].shape[0]):
        bboxes0, bboxes1 = data['bboxes0'][i], data['bboxes1'][i]
        image0, image1 = data['image0'][i], data['image1'][i]

        bbox_img0 = bbox_transforms(
            bboxes0, image0, target_width, target_height, bboxes0_format, is_normalize
        )
        bbox_img1 = bbox_transforms(
            bboxes1, image1, target_width, target_height, bboxes1_format, is_normalize
        )

        bbox_img0_batch.append(bbox_img0)
        bbox_img1_batch.append(bbox_img1)

    bbox_img0_tensor = torch.cat(bbox_img0_batch, dim=0)
    bbox_img1_tensor = torch.cat(bbox_img1_batch, dim=0)

    # Replicate grayscale crops because DINO backbones expect three channels.
    if bbox_img0_tensor.shape[1] == 1:
        bbox_img0_tensor = bbox_img0_tensor.repeat(1, 3, 1, 1)
    if bbox_img1_tensor.shape[1] == 1:
        bbox_img1_tensor = bbox_img1_tensor.repeat(1, 3, 1, 1)

    return bbox_img0_tensor, bbox_img1_tensor, centers0, centers1
