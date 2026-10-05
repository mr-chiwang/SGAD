from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9


class DualSoftmaxMatcher(nn.Module):
    """Differentiable dual-softmax matching between area descriptors."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        batch_size, num_areas0, _ = feat_c0.shape
        num_areas1 = feat_c1.shape[1]

        feat_c0, feat_c1 = (
            feature / feature.shape[-1] ** 0.5 for feature in (feat_c0, feat_c1)
        )

        if mask_c0 is not None and mask_c1 is not None:
            mask = (mask_c0[..., None] * mask_c1[:, None]).bool()
        else:
            mask = torch.ones(
                (batch_size, num_areas0, num_areas1),
                dtype=torch.bool,
                device=feat_c0.device,
            )

        similarity = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        if mask_c0 is not None:
            similarity.masked_fill_(~mask, -INF)
        confidence = F.softmax(similarity, 1) * F.softmax(similarity, 2)
        confidence.masked_fill_(~mask, 0)

        data.update({"conf_matrix_bbox": confidence, "conf_mask_bbox": mask})
        return confidence, mask


def match_area_descriptors_bidirectionally(
    area_matcher,
    *,
    feat_c0,
    feat_c1,
    data: dict[str, Any],
    mask_features0,
    mask_features1,
):
    """Run both matching directions, leaving the 0-to-1 result as the loss input."""
    conf_matrix_1to0, mask_1to0 = area_matcher(
        feat_c1,
        feat_c0,
        data,
        mask_c0=mask_features1,
        mask_c1=mask_features0,
    )
    conf_matrix_0to1, mask_0to1 = area_matcher(
        feat_c0,
        feat_c1,
        data,
        mask_c0=mask_features0,
        mask_c1=mask_features1,
    )
    return conf_matrix_0to1, mask_0to1, conf_matrix_1to0, mask_1to0
