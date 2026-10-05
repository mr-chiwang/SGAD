from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import numpy as np
import torch

from sgad.evaluation.visualization import bboxes_matches_top1


def should_log_training_step(
    is_rank_zero: bool,
    global_step: int,
    log_every_n_steps: int,
) -> bool:
    return is_rank_zero and global_step % log_every_n_steps == 0


def _is_valid_metric_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(np.isnan(value))
    except TypeError:
        return True


def log_training_step_metrics(
    batch: dict[str, Any],
    *,
    is_rank_zero: bool,
    global_step: int,
    log_every_n_steps: int,
    experiment: Any,
    compute_universal_metrics: Callable[..., dict[str, Any]],
    pred_score_threshold: float,
    enable_plotting: bool,
    make_top1_figure: Callable[[dict[str, Any]], Any] = bboxes_matches_top1,
    compute_sim_auc: Callable[..., dict[float, float]] | None = None,
) -> bool:
    if not should_log_training_step(is_rank_zero, global_step, log_every_n_steps):
        return False

    for key, value in batch["loss_scalars"].items():
        experiment.add_scalar(f"train/{key}", value, global_step)

    universal_metrics = compute_universal_metrics(
        batch,
        pred_score_threshold=pred_score_threshold,
    )
    mom_value = universal_metrics.get("MOM_mean")
    if _is_valid_metric_value(mom_value):
        experiment.add_scalar("train/MOM", mom_value, global_step)

    if compute_sim_auc is not None:
        bbox_sim_auc = compute_sim_auc(batch, overlap_thresholds=[0.2])[0.2]
        experiment.add_scalar("train/sim_auc", bbox_sim_auc, global_step)

    if enable_plotting:
        fig_top1 = make_top1_figure(batch)
        if fig_top1 is not None:
            experiment.add_figure("train_match/top1", fig_top1, global_step)

    return True


def compute_train_average_loss(outputs: list[dict[str, Any]]) -> float | None:
    if len(outputs) == 0:
        return None
    return torch.stack([output["loss"] for output in outputs]).mean().item()


def log_train_epoch_average(
    experiment: Any,
    avg_loss: float,
    current_epoch: int,
) -> None:
    experiment.add_scalar(
        "train/avg_loss_on_epoch",
        avg_loss,
        global_step=current_epoch,
    )


def build_train_epoch_log_entry(
    epoch: int,
    avg_loss: float,
    pending_val_metrics: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = {
        "epoch": epoch,
        "timestamp": timestamp,
        "train": {
            "avg_loss": round(avg_loss, 6),
        },
    }
    if pending_val_metrics is not None:
        log_entry["val"] = pending_val_metrics

    return log_entry


def process_train_epoch_outputs(
    outputs: list[dict[str, Any]],
    *,
    current_epoch: int,
    is_rank_zero: bool,
    experiment: Any,
    pending_val_metrics: dict[str, Any] | None,
    timestamp: str | None = None,
) -> dict[str, Any] | None:
    avg_loss = compute_train_average_loss(outputs)
    if avg_loss is None:
        return None

    if not is_rank_zero:
        return {
            "avg_loss": avg_loss,
            "log_entry": None,
            "consumed_pending_val_metrics": False,
        }

    log_train_epoch_average(experiment, avg_loss, current_epoch)
    return {
        "avg_loss": avg_loss,
        "log_entry": build_train_epoch_log_entry(
            current_epoch,
            avg_loss,
            pending_val_metrics=pending_val_metrics,
            timestamp=timestamp,
        ),
        "consumed_pending_val_metrics": pending_val_metrics is not None,
    }
