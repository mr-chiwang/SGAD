import torch
import torch.nn as nn


class SoftOverlapLoss(nn.Module):
    """Soft-overlap matching loss used by the two SGAD++ variants."""

    SUPPORTED_TYPES = {"bce", "focal_soft"}

    def __init__(self, loss_type="bce"):
        super().__init__()
        if loss_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown soft overlap loss type: {loss_type}")
        self.loss_type = loss_type

    def forward(self, conf_matrix, overlap_matrix, weight=None):
        overlap_matrix = overlap_matrix.clamp(0, 1)
        if weight is not None and not weight.bool().any():
            return conf_matrix.new_tensor(0.0)

        conf_clamped = conf_matrix.clamp(1e-6, 1 - 1e-6)
        bce = -overlap_matrix * torch.log(conf_clamped) - (
            1 - overlap_matrix
        ) * torch.log(1 - conf_clamped)
        if self.loss_type == "bce":
            loss = bce
        else:
            alpha = 0.25
            gamma = 2.0
            scale = torch.abs(overlap_matrix - conf_clamped) ** gamma
            loss = alpha * scale * bce

        if weight is not None:
            return (loss * weight).sum() / weight.sum().clamp(min=1)
        return loss.mean()
