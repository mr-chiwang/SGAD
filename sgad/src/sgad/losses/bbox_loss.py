# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR focal loss for SGAD region matching.

import torch
import torch.nn as nn

from .overlap_losses import SoftOverlapLoss


class Loss(nn.Module):
    """Combined matching and ranking loss for the public SGAD variants."""

    def __init__(self, config):
        super().__init__()
        self.loss_config = config["model"]["loss"]
        self.c_pos_w = self.loss_config["pos_weight"]
        self.c_neg_w = self.loss_config["neg_weight"]
        self.iou_thr = config["model"]["bbox"]["iou_thr"]

        self.use_soft_overlap_loss = self.loss_config.get("use_soft_overlap_loss", False)
        self.soft_overlap_loss_weight = self.loss_config.get("soft_overlap_loss_weight", 1.0)
        if self.use_soft_overlap_loss:
            self.soft_overlap_loss_fn = SoftOverlapLoss(
                loss_type=self.loss_config["soft_overlap_loss_type"]
            )

    def compute_hard_focal_loss(self, conf, conf_gt, weight=None):
        """Compute the hard-label focal loss used by published SGAD."""
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.loss_config["focal_alpha"]
        gamma = self.loss_config["focal_gamma"]
        loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * conf[pos_mask].log()
        loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
            loss_neg = loss_neg * weight[neg_mask]
        return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

    @staticmethod
    def reshape_for_rank(y_pred, y_true):
        _, _, sequence_length = y_pred.shape
        return y_pred.reshape(-1, sequence_length), y_true.reshape(-1, sequence_length)

    def listMLE_loss(self, y_pred, y_true, eps=1e-10, padded_value_indicator=0.03):
        """Compute ListMLE over rows with at least two valid overlap targets."""
        y_pred_reshaped, y_true_reshaped = self.reshape_for_rank(y_pred, y_true)
        valid_elements = (y_true_reshaped > padded_value_indicator).sum(dim=1)
        valid_sequences_mask = valid_elements > 1
        if not valid_sequences_mask.any():
            return y_pred.new_tensor(0.0)

        y_pred_valid = y_pred_reshaped[valid_sequences_mask]
        y_true_valid = y_true_reshaped[valid_sequences_mask]
        random_indices = torch.randperm(
            y_pred_valid.shape[-1], device=y_pred_valid.device
        )
        y_pred_shuffled = y_pred_valid[:, random_indices]
        y_true_shuffled = y_true_valid[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        padded_mask = y_true_sorted <= padded_value_indicator
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[padded_mask] = float("-inf")

        max_pred_values = preds_sorted_by_true.max(dim=1, keepdim=True).values
        normalized_preds = preds_sorted_by_true - max_pred_values
        cumulative_sums = torch.cumsum(
            normalized_preds.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])
        observation_loss = torch.log(cumulative_sums + eps) - normalized_preds
        observation_loss[padded_mask] = 0.0
        sequence_loss = observation_loss.sum(dim=1) / (
            valid_elements[valid_sequences_mask] - 1
        )
        return sequence_loss.mean()

    def forward(self, data):
        conf_matrix_bbox = data["conf_matrix_bbox"]
        if (
            conf_matrix_bbox.numel() == 0
            or conf_matrix_bbox.shape[1] == 0
            or conf_matrix_bbox.shape[2] == 0
        ):
            device = data["image0"].device if "image0" in data else conf_matrix_bbox.device
            zero_loss = torch.tensor(0.0, device=device)
            zero_cpu = zero_loss.detach().cpu()
            data.update(
                {
                    "loss": zero_loss,
                    "loss_scalars": {
                        "loss_sim_bbox": zero_cpu,
                        "loss_rank_bbox": zero_cpu.clone(),
                        "loss_bbox": zero_cpu.clone(),
                    },
                }
            )
            return

        bbox_weight = data["conf_mask_bbox"].int()
        if self.use_soft_overlap_loss:
            loss_sim_bbox = self.soft_overlap_loss_fn(
                conf_matrix_bbox,
                data["overlap_matrix"],
                weight=bbox_weight,
            )
            loss_sim_bbox = self.soft_overlap_loss_weight * loss_sim_bbox
        else:
            bbox_gt = (data["overlap_matrix"] > self.iou_thr).int()
            loss_sim_bbox = self.compute_hard_focal_loss(
                conf_matrix_bbox, bbox_gt, weight=bbox_weight
            )

        loss_rank_bbox = self.listMLE_loss(conf_matrix_bbox, data["overlap_matrix"])
        loss_bbox = loss_sim_bbox + loss_rank_bbox
        data.update(
            {
                "loss": loss_bbox,
                "loss_scalars": {
                    "loss_sim_bbox": loss_sim_bbox.detach().cpu(),
                    "loss_rank_bbox": loss_rank_bbox.detach().cpu(),
                    "loss_bbox": loss_bbox.detach().cpu(),
                },
            }
        )
