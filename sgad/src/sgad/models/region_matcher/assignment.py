from __future__ import annotations

from typing import Tuple, List

import torch


def dual_matching(
    conf_matrix_bbox_0to1: torch.Tensor,
    conf_matrix_bbox_1to0: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Mutual nearest neighbor matching (dual matching) for bbox assignment.

    This implements a simple mutual-NN policy:
    - i -> j* = argmax_j score(i, j)
    - j* -> i* = argmax_i score(j*, i)
    - accept if i* == i and scores pass `threshold`
    """
    batch_size, num_bboxes0, num_bboxes1 = conf_matrix_bbox_0to1.shape
    if num_bboxes0 == 0 or num_bboxes1 == 0:
        return [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]

    batch_indices_area0: List[List[int]] = []
    batch_indices_area1: List[List[int]] = []

    for b in range(batch_size):
        conf0 = conf_matrix_bbox_0to1[b]
        conf1 = conf_matrix_bbox_1to0[b]

        max_scores_0to1, max_indices_0to1 = conf0.max(dim=1)
        max_scores_1to0, max_indices_1to0 = conf1.max(dim=1)

        indices_area0: List[int] = []
        indices_area1: List[int] = []
        used0 = set()
        used1 = set()

        for i in range(conf0.size(0)):
            j = int(max_indices_0to1[i].item())

            if float(max_scores_0to1[i].item()) <= float(threshold):
                continue

            if int(max_indices_1to0[j].item()) != i:
                continue
            if float(max_scores_1to0[j].item()) <= float(threshold):
                continue

            if i in used0 or j in used1:
                continue

            indices_area0.append(i)
            indices_area1.append(j)
            used0.add(i)
            used1.add(j)

        batch_indices_area0.append(indices_area0)
        batch_indices_area1.append(indices_area1)

    return batch_indices_area0, batch_indices_area1
