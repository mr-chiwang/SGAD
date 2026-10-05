from __future__ import annotations

from typing import Any, Callable

import torch

from sgad.data.areas.preparation import prepare_bboxes_label
from sgad.training.supervision import compute_supervision_bbox


BatchCallback = Callable[[dict[str, Any]], Any]


def batch_has_empty_bboxes(batch: dict[str, Any]) -> bool:
    bboxes0_list = batch.get("bboxes0", [])
    bboxes1_list = batch.get("bboxes1", [])
    for i in range(len(bboxes0_list)):
        if len(bboxes0_list[i]) == 0 or len(bboxes1_list[i]) == 0:
            return True
    return False


def fill_empty_batch_results(batch: dict[str, Any]) -> None:
    """Create a differentiable zero loss for a batch skipped due to empty regions."""
    zero_loss = torch.tensor(0.0, device=batch["image0"].device, requires_grad=True)
    zero_cpu = zero_loss.detach().cpu()
    batch.update(
        {
            "loss": zero_loss,
            "loss_scalars": {
                "loss_sim_bbox": zero_cpu,
                "loss_rank_bbox": zero_cpu.clone(),
                "loss_bbox": zero_cpu.clone(),
            },
        }
    )


def run_trainval_inference(
    batch: dict[str, Any],
    *,
    config: Any,
    profiler: Any,
    matcher: BatchCallback,
    loss: BatchCallback,
    is_train: bool = True,
) -> bool:
    with profiler.profile("Compute bbox supervision"):
        compute_supervision_bbox(batch, config)

    with profiler.profile("Prepare bboxes label"):
        prepare_bboxes_label(batch, max_bboxes=50, is_train=is_train)

    if batch_has_empty_bboxes(batch):
        fill_empty_batch_results(batch)
        return False

    with profiler.profile("BboxMatcher"):
        matcher(batch)

    with profiler.profile("Compute losses"):
        loss(batch)

    return True
