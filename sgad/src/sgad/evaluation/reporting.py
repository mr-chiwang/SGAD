from __future__ import annotations

from datetime import datetime
from typing import Any, Callable


TEST_CORE_METRICS = (
    ("MOM", "MOM_mean"),
    ("BMR", "BMR_mean"),
    ("WBMR", "WBMR_mean"),
    ("HMQR", "HMQR_mean"),
    ("Coverage", "Coverage_mean"),
    ("AUC_ORC", "AUC_ORC_mean"),
)
TEST_EVAL_THRESHOLDS = (0.2, 0.4, 0.6)


def iter_test_result_log_lines(
    universal_final: dict[float, dict[str, Any]],
    sim_auc_final: dict[float, float],
    avg_time_matcher: float,
    valid_outputs_count: int,
) -> list[str]:
    lines = [
        "=== Test Results (Universal Evaluation Framework) ===",
        f"Average Matcher Time: {avg_time_matcher:.4f}s",
        f"Valid outputs: {valid_outputs_count}",
    ]

    if sim_auc_final:
        lines.append("\nSimilarity AUC:")
        for overlap_threshold in sorted(sim_auc_final):
            lines.append(f"  AUC@{overlap_threshold}: {sim_auc_final[overlap_threshold]:.4f}")

    for pred_threshold in sorted(universal_final.keys()):
        um = universal_final[pred_threshold]
        lines.extend(
            [
                f"\nPrediction Threshold {pred_threshold}:",
                "  Core Metrics:",
                f"    MOM (Mean Overlap):     {um.get('MOM_mean', float('nan')):.4f}",
                f"    BMR (Best Match Recall): {um.get('BMR_mean', float('nan')):.4f}",
                f"    WBMR (Weighted BMR):    {um.get('WBMR_mean', float('nan')):.4f}",
                f"    HMQR (Harmonic MOM-BMR): {um.get('HMQR_mean', float('nan')):.4f}",
                f"    Coverage:               {um.get('Coverage_mean', float('nan')):.4f}",
                f"    AUC-ORC:                {um.get('AUC_ORC_mean', float('nan')):.4f}",
                "  Threshold Metrics:",
            ]
        )

        for tau in TEST_EVAL_THRESHOLDS:
            mp_key = f"MP@{tau}_mean"
            recall_key = f"Recall@{tau}_mean"
            if mp_key in um:
                lines.append(f"    MP@{tau}:     {um.get(mp_key, float('nan')):.4f}")
            if recall_key in um:
                lines.append(f"    Recall@{tau}: {um.get(recall_key, float('nan')):.4f}")

    return lines


def log_test_results(
    universal_final: dict[float, dict[str, Any]],
    sim_auc_final: dict[float, float],
    avg_time_matcher: float,
    valid_outputs_count: int,
    log_info: Callable[[str], None],
) -> None:
    for line in iter_test_result_log_lines(
        universal_final,
        sim_auc_final,
        avg_time_matcher,
        valid_outputs_count,
    ):
        log_info(line)


def build_test_log_metrics(
    universal_final: dict[float, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    universal_metrics_log = {}
    for pred_threshold in sorted(universal_final.keys()):
        pred_key = f"pred_{pred_threshold}"
        um = universal_final[pred_threshold]
        universal_metrics_log[pred_key] = {
            public_key: round(float(um.get(metric_key, 0)), 4)
            for public_key, metric_key in TEST_CORE_METRICS
        }

        for tau in TEST_EVAL_THRESHOLDS:
            mp_key = f"MP@{tau}_mean"
            recall_key = f"Recall@{tau}_mean"
            if mp_key in um:
                universal_metrics_log[pred_key][f"MP@{tau}"] = round(float(um.get(mp_key, 0)), 4)
            if recall_key in um:
                universal_metrics_log[pred_key][f"Recall@{tau}"] = round(float(um.get(recall_key, 0)), 4)

    return universal_metrics_log


def build_test_log_entry(
    universal_final: dict[float, dict[str, Any]],
    sim_auc_final: dict[float, float],
    avg_time_matcher: float,
    valid_outputs_count: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "type": "test",
        "timestamp": timestamp,
        "avg_matcher_time": round(avg_time_matcher, 4),
        "valid_outputs": valid_outputs_count,
        "sim_auc": {
            f"AUC@{overlap_threshold}": round(float(value), 4)
            for overlap_threshold, value in sorted(sim_auc_final.items())
        },
        "universal_metrics": build_test_log_metrics(universal_final),
    }
