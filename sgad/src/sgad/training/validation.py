from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from sgad.evaluation.matching import deduplicate_sample_outputs
from sgad.utils.comm import gather
from sgad.utils.misc import flatten_list


CORE_VALIDATION_METRICS = ("MOM", "BMR", "HMQR", "Coverage", "AUC_ORC", "sim_auc")
VALIDATION_THRESHOLDS = (0.2, 0.4, 0.6)
TENSORBOARD_CORE_METRICS = (
    "MOM_mean",
    "BMR_mean",
    "HMQR_mean",
    "Coverage_mean",
    "AUC_ORC_mean",
)


def _is_valid_metric_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(np.isnan(value))
    except TypeError:
        return True


def _normalize_validation_outputs(outputs: list[Any]) -> list[list[dict]]:
    return [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs


def _aggregate_universal_metrics(outputs: list[dict]) -> dict[str, float]:
    universal_outputs = [o.get("universal_metrics", {}) for o in outputs]
    universal_metrics_agg = {}
    if not universal_outputs:
        return universal_metrics_agg

    all_universal_keys = set()
    for universal_metrics in universal_outputs:
        all_universal_keys.update(universal_metrics.keys())

    for key in all_universal_keys:
        values = [
            universal_metrics.get(key)
            for universal_metrics in universal_outputs
            if key in universal_metrics and _is_valid_metric_value(universal_metrics.get(key))
        ]
        if values:
            universal_metrics_agg[key] = np.mean(values)

    return universal_metrics_agg


def _aggregate_sim_auc(outputs: list[dict]) -> float:
    values = [output["sim_auc"] for output in outputs if _is_valid_metric_value(output.get("sim_auc"))]
    return float(np.mean(values)) if values else float("nan")


def _aggregate_loss_scalars(outputs: list[dict]) -> dict[str, list[torch.Tensor]]:
    loss_scalars = [o["loss_scalars"] for o in outputs]
    return {
        key: [scalars[key] for scalars in loss_scalars]
        for key in loss_scalars[0]
    }


def _aggregate_figures(outputs: list[dict]) -> dict[str, list[Any]]:
    figures = [o["figures"] for o in outputs]
    return {
        key: flatten_list([item[key] for item in figures])
        for key in figures[0]
    }


def _append_multi_val_metrics(
    multi_val_metrics: dict[str, list[float]],
    universal_metrics_agg: dict[str, float],
    sim_auc_value: float,
) -> None:
    multi_val_metrics["MOM"].append(universal_metrics_agg.get("MOM_mean", 0))
    multi_val_metrics["BMR"].append(universal_metrics_agg.get("BMR_mean", 0))
    multi_val_metrics["HMQR"].append(universal_metrics_agg.get("HMQR_mean", 0))
    multi_val_metrics["Coverage"].append(universal_metrics_agg.get("Coverage_mean", 0))
    multi_val_metrics["AUC_ORC"].append(universal_metrics_agg.get("AUC_ORC_mean", 0))
    multi_val_metrics["sim_auc"].append(sim_auc_value)
    for threshold in VALIDATION_THRESHOLDS:
        multi_val_metrics[f"MP@{threshold}"].append(universal_metrics_agg.get(f"MP@{threshold}_mean", 0))


def _log_validation_dataset(
    experiment: Any,
    valset_idx: int,
    cur_epoch: int,
    loss_scalars: dict[str, list[torch.Tensor]],
    figures: dict[str, list[Any]],
    universal_metrics_agg: dict[str, float],
    sim_auc_value: float,
) -> None:
    for key, values in loss_scalars.items():
        mean_value = torch.stack(values).mean()
        experiment.add_scalar(f"val_{valset_idx}/avg_{key}", mean_value, global_step=cur_epoch)

    for key in TENSORBOARD_CORE_METRICS:
        if key in universal_metrics_agg:
            experiment.add_scalar(f"val_{valset_idx}/{key}", universal_metrics_agg[key], global_step=cur_epoch)

    experiment.add_scalar(f"metrics_{valset_idx}/sim_auc", sim_auc_value, global_step=cur_epoch)

    for threshold in VALIDATION_THRESHOLDS:
        mp_key = f"MP@{threshold}_mean"
        if mp_key in universal_metrics_agg:
            experiment.add_scalar(f"val_{valset_idx}/{mp_key}", universal_metrics_agg[mp_key], global_step=cur_epoch)

    for key, values in figures.items():
        for plot_idx, fig in enumerate(values):
            if fig is not None:
                experiment.add_figure(f"val_match_{valset_idx}/{key}/pair-{plot_idx}", fig, cur_epoch, close=True)


def _compute_validation_summary(multi_val_metrics: dict[str, list[float]]) -> dict[str, float]:
    summary = {
        metric: float(np.mean(multi_val_metrics[metric]))
        for metric in CORE_VALIDATION_METRICS
    }
    for threshold in VALIDATION_THRESHOLDS:
        summary[f"MP@{threshold}"] = float(np.mean(multi_val_metrics[f"MP@{threshold}"]))
    return summary


def build_pending_validation_metrics(summary: dict[str, float]) -> dict[str, float]:
    return {
        "MOM": round(float(summary["MOM"]), 4),
        "BMR": round(float(summary["BMR"]), 4),
        "HMQR": round(float(summary["HMQR"]), 4),
        "Coverage": round(float(summary["Coverage"]), 4),
        "AUC_ORC": round(float(summary["AUC_ORC"]), 4),
        "sim_auc": round(float(summary["sim_auc"]), 4),
        "MP@0.2": round(float(summary["MP@0.2"]), 4),
        "MP@0.4": round(float(summary["MP@0.4"]), 4),
        "MP@0.6": round(float(summary["MP@0.6"]), 4),
    }


def process_validation_epoch_outputs(
    outputs: list[Any],
    *,
    current_epoch: int,
    ckpt_path: str | None,
    sanity_checking: bool,
    is_rank_zero: bool,
    experiment: Any = None,
    close_figures: Callable[[], None] | None = None,
    gather_fn: Callable[[Any], list[Any]] = gather,
) -> dict[str, float] | None:
    if len(outputs) == 0:
        return None

    multi_outputs = _normalize_validation_outputs(outputs)
    multi_val_metrics = defaultdict(list)

    for valset_idx, val_outputs in enumerate(multi_outputs):
        rank_outputs = gather_fn(val_outputs)

        if is_rank_zero:
            gathered_outputs = deduplicate_sample_outputs(flatten_list(rank_outputs))
            cur_epoch = -1 if ckpt_path is None and sanity_checking else current_epoch

            loss_scalars = _aggregate_loss_scalars(gathered_outputs)
            figures = _aggregate_figures(gathered_outputs)
            universal_metrics_agg = _aggregate_universal_metrics(gathered_outputs)
            sim_auc_value = _aggregate_sim_auc(gathered_outputs)

            _append_multi_val_metrics(multi_val_metrics, universal_metrics_agg, sim_auc_value)

            if experiment is not None:
                _log_validation_dataset(
                    experiment,
                    valset_idx,
                    cur_epoch,
                    loss_scalars,
                    figures,
                    universal_metrics_agg,
                    sim_auc_value,
                )

        if close_figures is not None:
            close_figures()

    if not is_rank_zero:
        return None
    return _compute_validation_summary(multi_val_metrics)
