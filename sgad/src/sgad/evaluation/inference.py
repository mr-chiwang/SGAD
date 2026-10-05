from __future__ import annotations

import time
from typing import Any, Callable

from sgad.evaluation.matching import (
    compute_sim_auc_by_overlap_threshold,
    evaluate_match_indices_by_threshold,
)
from sgad.data.areas.preparation import prepare_bboxes_label
from sgad.training.supervision import compute_supervision_bbox
from sgad.training.inference import batch_has_empty_bboxes


DEFAULT_TEST_PRED_SCORE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
BatchCallback = Callable[[dict[str, Any]], Any]


def run_test_inference(
    batch: dict[str, Any],
    *,
    config: Any,
    profiler: Any,
    matcher: BatchCallback,
    compute_all_matches: Callable[
        [dict[str, Any], list[float]],
        tuple[dict[float, dict[str, Any]], dict[float, dict[str, list[list[int]]]]],
    ],
    pred_score_thresholds: list[float] | None = None,
    clock: Callable[[], float] = time.time,
) -> tuple[dict[float, dict[str, float]], dict[float, float], float] | None:
    with profiler.profile("Compute bbox supervision"):
        compute_supervision_bbox(batch, config)

    with profiler.profile("Prepare bboxes label"):
        prepare_bboxes_label(batch, max_bboxes=500, is_train=False)

    if batch_has_empty_bboxes(batch):
        return None

    time_start_matcher = clock()
    with profiler.profile("Matcher"):
        matcher(batch)
    time_end_matcher = clock()
    time_matcher = time_end_matcher - time_start_matcher

    thresholds = pred_score_thresholds or DEFAULT_TEST_PRED_SCORE_THRESHOLDS
    _, all_match_indices = compute_all_matches(batch, thresholds)
    results_by_threshold = evaluate_match_indices_by_threshold(
        batch,
        all_match_indices,
    )
    sim_auc_by_threshold = compute_sim_auc_by_overlap_threshold(batch)

    return results_by_threshold, sim_auc_by_threshold, time_matcher
