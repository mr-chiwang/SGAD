from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from loguru import logger

from sgad.models.region_matcher.assignment import dual_matching
from sgad.evaluation.metrics import evaluate_bbox_matching_batch, sim_auc


DEFAULT_EVAL_THRESHOLDS = [0.2, 0.4, 0.6]
DEFAULT_SIM_AUC_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
EMPTY_UNIVERSAL_METRIC_KEYS = (
    "MOM_mean",
    "BMR_mean",
    "WBMR_mean",
    "HMQR_mean",
    "Coverage_mean",
    "AUC_ORC_mean",
    "MP@0.2_mean",
    "MP@0.4_mean",
    "MP@0.6_mean",
    "Recall@0.2_mean",
    "Recall@0.4_mean",
    "Recall@0.6_mean",
)


def _single_batch_value(value: Any) -> Any:
    """Extract a scalar metadata value from a validation/test batch of size one."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Evaluation sample identity requires batch_size=1")
        return value.item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Evaluation sample identity requires batch_size=1")
        return _single_batch_value(value[0])
    return value


def evaluation_sample_key(batch: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the dataset-stable identity used to remove sampler padding copies."""
    return tuple(
        _single_batch_value(batch[field])
        for field in ("dataset_name", "scene_id", "pair_id")
    )


def deduplicate_sample_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first output for each evaluation sample, preserving gather order."""
    unique_outputs = []
    seen = set()
    for output in outputs:
        sample_key = output["sample_key"]
        if sample_key in seen:
            continue
        seen.add(sample_key)
        unique_outputs.append(output)
    return unique_outputs


def create_empty_universal_metrics() -> dict[str, float]:
    return {key: float("nan") for key in EMPTY_UNIVERSAL_METRIC_KEYS}


def compute_sim_auc_by_overlap_threshold(
    batch: dict[str, Any],
    overlap_thresholds: list[float] | None = None,
) -> dict[float, float]:
    mask = batch["conf_mask_bbox"]
    flat_matching_scores = batch["conf_matrix_bbox"][mask].view(-1).detach().cpu().numpy()
    overlap_values = batch["overlap_matrix"][mask].view(-1)

    return {
        threshold: sim_auc(
            (overlap_values >= threshold).detach().cpu().numpy(),
            flat_matching_scores,
        )
        for threshold in overlap_thresholds or DEFAULT_SIM_AUC_THRESHOLDS
    }


def compute_matching_indices(
    conf_matrix_bbox: torch.Tensor,
    pred_score_threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[int]]]:
    conf_matrix_0to1 = conf_matrix_bbox
    conf_matrix_1to0 = conf_matrix_bbox.transpose(-1, -2)

    try:
        return dual_matching(
            conf_matrix_0to1,
            conf_matrix_1to0,
            threshold=pred_score_threshold,
        )
    except Exception as exc:
        logger.warning(f"dual_matching failed: {exc}")
        batch_size = conf_matrix_bbox.shape[0]
        return [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]


def build_bbox_matches(
    batch: dict[str, Any],
    batch_indices_area0: list[list[int]],
    batch_indices_area1: list[list[int]],
) -> dict[str, list[list[Any]]]:
    conf_matrix_0to1 = batch["conf_matrix_bbox"]

    batch_match_bboxes0 = []
    batch_match_bboxes1 = []
    batch_true_match_scores = []
    batch_pred_match_scores = []

    for i in range(len(batch_indices_area0)):
        bboxes0 = batch["bboxes0"][i]
        bboxes1 = batch["bboxes1"][i]
        overlap_matrix = batch["overlap_matrix"][i]
        conf_matrix_0to1_i = conf_matrix_0to1[i]

        match_bboxes0 = []
        match_bboxes1 = []
        true_match_scores = []
        pred_match_scores = []

        for idx0, idx1 in zip(batch_indices_area0[i], batch_indices_area1[i]):
            match_bboxes0.append(bboxes0[idx0])
            match_bboxes1.append(bboxes1[idx1])
            true_match_scores.append(overlap_matrix[idx0, idx1])
            pred_match_scores.append(conf_matrix_0to1_i[idx0, idx1])

        batch_match_bboxes0.append(match_bboxes0)
        batch_match_bboxes1.append(match_bboxes1)
        batch_true_match_scores.append(true_match_scores)
        batch_pred_match_scores.append(pred_match_scores)

    return {
        "bboxes0_matched": batch_match_bboxes0,
        "bboxes1_matched": batch_match_bboxes1,
        "bboxes_match_scores": batch_true_match_scores,
        "bboxes_pred_match_scores": batch_pred_match_scores,
    }


def assign_bbox_matches(
    batch: dict[str, Any],
    pred_score_threshold: float = 0.5,
) -> dict[str, list[list[Any]]]:
    batch_indices_area0, batch_indices_area1 = compute_matching_indices(
        batch["conf_matrix_bbox"],
        pred_score_threshold=pred_score_threshold,
    )
    bboxes_matches = build_bbox_matches(batch, batch_indices_area0, batch_indices_area1)
    batch["bboxes_matches"] = bboxes_matches
    return bboxes_matches


def make_index_pairs(
    batch_indices_area0: list[list[int]],
    batch_indices_area1: list[list[int]],
) -> list[list[tuple[int, int]]]:
    return [
        list(zip(indices0, indices1))
        for indices0, indices1 in zip(batch_indices_area0, batch_indices_area1)
    ]


def extract_overlap_matrices(batch: dict[str, Any]) -> list[np.ndarray]:
    batch_overlap_matrices = []
    for i in range(len(batch["overlap_matrix"])):
        overlap_matrix = batch["overlap_matrix"][i]
        if isinstance(overlap_matrix, torch.Tensor):
            overlap_matrix = overlap_matrix.detach().cpu().numpy()
        batch_overlap_matrices.append(overlap_matrix)
    return batch_overlap_matrices


def evaluate_match_indices(
    batch: dict[str, Any],
    batch_indices_area0: list[list[int]],
    batch_indices_area1: list[list[int]],
    eval_thresholds: list[float] | None = None,
) -> dict[str, float]:
    batch_matches = make_index_pairs(batch_indices_area0, batch_indices_area1)
    batch_overlap_matrices = extract_overlap_matrices(batch)
    return evaluate_bbox_matching_batch(
        batch_matches,
        batch_overlap_matrices,
        thresholds=eval_thresholds or DEFAULT_EVAL_THRESHOLDS,
    )


def compute_universal_metrics_for_batch(
    batch: dict[str, Any],
    pred_score_threshold: float = 0.5,
    eval_thresholds: list[float] | None = None,
) -> dict[str, float]:
    if "bboxes_matches" not in batch:
        assign_bbox_matches(
            batch,
            pred_score_threshold=pred_score_threshold,
        )

    batch_indices_area0, batch_indices_area1 = compute_matching_indices(
        batch["conf_matrix_bbox"],
        pred_score_threshold=pred_score_threshold,
    )
    return evaluate_match_indices(
        batch,
        batch_indices_area0,
        batch_indices_area1,
        eval_thresholds=eval_thresholds,
    )


def compute_all_matches_by_threshold(
    batch: dict[str, Any],
    pred_score_thresholds: list[float],
) -> tuple[dict[float, dict[str, list[list[Any]]]], dict[float, dict[str, list[list[int]]]]]:
    all_matches = {}
    all_match_indices = {}

    for pred_threshold in pred_score_thresholds:
        batch_indices_area0, batch_indices_area1 = compute_matching_indices(
            batch["conf_matrix_bbox"],
            pred_score_threshold=pred_threshold,
        )

        all_match_indices[pred_threshold] = {
            "batch_indices_area0": [list(idx) for idx in batch_indices_area0],
            "batch_indices_area1": [list(idx) for idx in batch_indices_area1],
        }
        all_matches[pred_threshold] = build_bbox_matches(
            batch,
            batch_indices_area0,
            batch_indices_area1,
        )

    return all_matches, all_match_indices


def evaluate_match_indices_by_threshold(
    batch: dict[str, Any],
    all_match_indices: dict[float, dict[str, list[list[int]]]],
    eval_thresholds: list[float] | None = None,
) -> dict[float, dict[str, float]]:
    results_by_threshold = {}
    for pred_threshold, indices_info in all_match_indices.items():
        results_by_threshold[pred_threshold] = evaluate_match_indices(
            batch,
            indices_info["batch_indices_area0"],
            indices_info["batch_indices_area1"],
            eval_thresholds=eval_thresholds,
        )
    return results_by_threshold


def _is_valid_metric_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(np.isnan(value))
    except TypeError:
        return True


def aggregate_test_outputs(
    outputs: list[tuple[dict[float, dict[str, float]], dict[float, float], float] | None],
) -> tuple[dict[float, dict[str, float]], dict[float, float], float, int]:
    universal_accumulator: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sim_auc_accumulator: dict[float, list[float]] = defaultdict(list)
    total_time_matcher = 0.0
    valid_outputs_count = 0

    for result in outputs:
        if result is None:
            continue

        results_by_threshold, sim_auc_by_threshold, time_matcher = result
        valid_outputs_count += 1
        total_time_matcher += time_matcher

        for pred_threshold, metrics in results_by_threshold.items():
            for key, value in metrics.items():
                if _is_valid_metric_value(value):
                    universal_accumulator[pred_threshold][key].append(value)

        for overlap_threshold, value in sim_auc_by_threshold.items():
            if _is_valid_metric_value(value):
                sim_auc_accumulator[overlap_threshold].append(value)

    universal_final = {}
    for pred_threshold, metric_values in universal_accumulator.items():
        universal_final[pred_threshold] = {}
        for metric_name, values in metric_values.items():
            if values:
                universal_final[pred_threshold][metric_name] = np.mean(values)

    sim_auc_final = {
        overlap_threshold: np.mean(values)
        for overlap_threshold, values in sim_auc_accumulator.items()
        if values
    }

    avg_time_matcher = total_time_matcher / valid_outputs_count if valid_outputs_count > 0 else 0
    return universal_final, sim_auc_final, avg_time_matcher, valid_outputs_count
