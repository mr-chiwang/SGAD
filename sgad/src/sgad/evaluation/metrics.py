"""Model-independent metrics for bounding-box matching.

The framework reports match quality, best-match recall, source-region
coverage, overlap-recall AUC, thresholded precision and recall, and match
quality distribution statistics. See the repository-level
``docs/sgad_region_evaluation.md`` for the metric definitions.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Metrics and metadata for one evaluated image pair."""
    # Core metrics
    MOM: float  # Mean Overlap of Matches
    BMR: float  # Best Match Recall
    WBMR: float  # Weighted Best Match Recall
    HMQR: float  # Harmonic Mean of Quality and Recall
    Coverage: float  # Fraction of source regions included in a prediction
    AUC_ORC: float  # Area Under Overlap-Recall Curve

    # Match Quality Distribution statistics
    MQD_median: float
    MQD_std: float
    MQD_min: float
    MQD_max: float
    MQD_Q25: float
    MQD_Q75: float
    MQD_IQR: float

    # Threshold-dependent metrics
    MP_at_thresholds: Dict[float, float]  # MP@τ
    Recall_at_thresholds: Dict[float, float]  # Recall@τ

    # Evaluation metadata
    num_matches: int
    num_source_bboxes: int
    num_target_bboxes: int
    num_valid_source_bboxes: int  # Source regions with a positive-overlap target

    def to_dict(self) -> Dict:
        """Return a flat dictionary suitable for logging and aggregation."""
        result = {
            # Core metrics
            "MOM": self.MOM,
            "BMR": self.BMR,
            "WBMR": self.WBMR,
            "HMQR": self.HMQR,
            "Coverage": self.Coverage,
            "AUC_ORC": self.AUC_ORC,

            # Distribution statistics
            "MQD_median": self.MQD_median,
            "MQD_std": self.MQD_std,
            "MQD_min": self.MQD_min,
            "MQD_max": self.MQD_max,
            "MQD_Q25": self.MQD_Q25,
            "MQD_Q75": self.MQD_Q75,
            "MQD_IQR": self.MQD_IQR,

            # Evaluation metadata
            "num_matches": self.num_matches,
            "num_source_bboxes": self.num_source_bboxes,
            "num_target_bboxes": self.num_target_bboxes,
            "num_valid_source_bboxes": self.num_valid_source_bboxes,
        }

        # Threshold-dependent metrics
        for tau, value in self.MP_at_thresholds.items():
            result[f"MP@{tau}"] = value
        for tau, value in self.Recall_at_thresholds.items():
            result[f"Recall@{tau}"] = value

        return result


class BBoxMatchingEvaluator:
    """Evaluate predicted region correspondences against an overlap matrix.

    Example:
    ```python
    evaluator = BBoxMatchingEvaluator()

    # Evaluate one image pair.
    result = evaluator.evaluate(
        matches=[(0, 2), (1, 1), (3, 0)],
        overlap_matrix=overlap_matrix,
    )

    # Evaluate a batch of image pairs.
    results = evaluator.evaluate_batch(
        batch_matches=[...],
        batch_overlap_matrices=[...],
    )

    # Aggregate pair-level results.
    aggregated = evaluator.aggregate_results(results)
    ```
    """

    def __init__(self, thresholds: Optional[List[float]] = None, orc_num_points: int = 20):
        """Initialize the evaluator.

        Args:
            thresholds: Overlap cutoffs used by MP@tau and Recall@tau. Defaults
                to ``[0.2, 0.4, 0.6]``.
            orc_num_points: Number of overlap thresholds sampled for AUC-ORC.
        """
        self.thresholds = thresholds or [0.2, 0.4, 0.6]
        self.orc_num_points = orc_num_points

    def evaluate(
        self,
        matches: List[Tuple[int, int]],
        overlap_matrix: Union[torch.Tensor, np.ndarray],
    ) -> EvaluationResult:
        """Evaluate predicted matches for one image pair.

        Args:
            matches: Predicted ``(source_index, target_index)`` pairs.
            overlap_matrix: Ground-truth overlap matrix with shape ``[L, S]``.

        Returns:
            Pair-level metrics and metadata.
        """
        # Compute all metrics with NumPy arrays on the CPU.
        if isinstance(overlap_matrix, torch.Tensor):
            overlap_matrix = overlap_matrix.detach().cpu().numpy()

        # Preserve shape metadata while returning undefined quality metrics.
        if overlap_matrix.size == 0 or overlap_matrix.shape[0] == 0 or overlap_matrix.shape[1] == 0:
            L = overlap_matrix.shape[0] if len(overlap_matrix.shape) > 0 else 0
            S = overlap_matrix.shape[1] if len(overlap_matrix.shape) > 1 else 0
            return EvaluationResult(
                MOM=float('nan'),
                BMR=float('nan'),
                WBMR=float('nan'),
                HMQR=float('nan'),
                Coverage=0.0,
                AUC_ORC=float('nan'),
                MQD_median=float('nan'),
                MQD_std=float('nan'),
                MQD_min=float('nan'),
                MQD_max=float('nan'),
                MQD_Q25=float('nan'),
                MQD_Q75=float('nan'),
                MQD_IQR=float('nan'),
                MP_at_thresholds={tau: float('nan') for tau in self.thresholds},
                Recall_at_thresholds={tau: 0.0 for tau in self.thresholds},
                num_matches=0,
                num_source_bboxes=L,
                num_target_bboxes=S,
                num_valid_source_bboxes=0,
            )

        L, S = overlap_matrix.shape
        K = len(matches)

        # Extract the ground-truth overlap of each predicted match.
        if K > 0:
            match_overlaps = np.array([overlap_matrix[i, j] for i, j in matches])
        else:
            match_overlaps = np.array([])

        # Compute the core metrics.
        MOM = self._compute_MOM(match_overlaps)
        BMR, WBMR, num_valid = self._compute_BMR_WBMR(matches, overlap_matrix)
        HMQR = self._compute_HMQR(MOM, BMR)
        Coverage = self._compute_coverage(matches, L)
        AUC_ORC = self._compute_AUC_ORC(matches, overlap_matrix)

        # Summarize the predicted-match overlap distribution.
        mqd_stats = self._compute_MQD_stats(match_overlaps)

        # Compute threshold-dependent metrics.
        MP_at_thresholds = {}
        Recall_at_thresholds = {}
        for tau in self.thresholds:
            MP_at_thresholds[tau] = self._compute_MP_at_threshold(match_overlaps, tau)
            Recall_at_thresholds[tau] = self._compute_Recall_at_threshold(matches, overlap_matrix, tau)

        return EvaluationResult(
            MOM=MOM,
            BMR=BMR,
            WBMR=WBMR,
            HMQR=HMQR,
            Coverage=Coverage,
            AUC_ORC=AUC_ORC,
            MQD_median=mqd_stats['median'],
            MQD_std=mqd_stats['std'],
            MQD_min=mqd_stats['min'],
            MQD_max=mqd_stats['max'],
            MQD_Q25=mqd_stats['Q25'],
            MQD_Q75=mqd_stats['Q75'],
            MQD_IQR=mqd_stats['IQR'],
            MP_at_thresholds=MP_at_thresholds,
            Recall_at_thresholds=Recall_at_thresholds,
            num_matches=K,
            num_source_bboxes=L,
            num_target_bboxes=S,
            num_valid_source_bboxes=num_valid,
        )

    def evaluate_batch(
        self,
        batch_matches: List[List[Tuple[int, int]]],
        batch_overlap_matrices: List[Union[torch.Tensor, np.ndarray]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple image pairs independently.

        Args:
            batch_matches: Predicted matches for each image pair.
            batch_overlap_matrices: Ground-truth overlap matrix for each pair.

        Returns:
            One :class:`EvaluationResult` per image pair.
        """
        results = []
        for matches, overlap_matrix in zip(batch_matches, batch_overlap_matrices):
            result = self.evaluate(matches, overlap_matrix)
            results.append(result)
        return results

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict:
        """Macro-average each defined metric across image pairs.

        NaN values are excluded independently for each metric.

        Args:
            results: Pair-level evaluation results.

        Returns:
            Dataset-level mean, standard deviation, and count metrics.
        """
        if len(results) == 0:
            return {}

        # Drop NaN values before aggregating each metric independently.
        MOMs = [r.MOM for r in results if not np.isnan(r.MOM)]
        BMRs = [r.BMR for r in results if not np.isnan(r.BMR)]
        WBMRs = [r.WBMR for r in results if not np.isnan(r.WBMR)]
        HMQRs = [r.HMQR for r in results if not np.isnan(r.HMQR)]
        Coverages = [r.Coverage for r in results if not np.isnan(r.Coverage)]
        AUC_ORCs = [r.AUC_ORC for r in results if not np.isnan(r.AUC_ORC)]

        # Aggregate core metrics.
        aggregated = {
            "MOM_mean": np.mean(MOMs) if MOMs else float('nan'),
            "MOM_std": np.std(MOMs) if MOMs else float('nan'),
            "BMR_mean": np.mean(BMRs) if BMRs else float('nan'),
            "BMR_std": np.std(BMRs) if BMRs else float('nan'),
            "WBMR_mean": np.mean(WBMRs) if WBMRs else float('nan'),
            "WBMR_std": np.std(WBMRs) if WBMRs else float('nan'),
            "HMQR_mean": np.mean(HMQRs) if HMQRs else float('nan'),
            "HMQR_std": np.std(HMQRs) if HMQRs else float('nan'),
            "Coverage_mean": np.mean(Coverages) if Coverages else float('nan'),
            "Coverage_std": np.std(Coverages) if Coverages else float('nan'),
            "AUC_ORC_mean": np.mean(AUC_ORCs) if AUC_ORCs else float('nan'),
            "AUC_ORC_std": np.std(AUC_ORCs) if AUC_ORCs else float('nan'),
        }

        # Aggregate threshold-dependent metrics.
        for tau in self.thresholds:
            MPs = [r.MP_at_thresholds[tau] for r in results if not np.isnan(r.MP_at_thresholds.get(tau, float('nan')))]
            Recalls = [r.Recall_at_thresholds[tau] for r in results if not np.isnan(r.Recall_at_thresholds.get(tau, float('nan')))]

            aggregated[f"MP@{tau}_mean"] = np.mean(MPs) if MPs else float('nan')
            aggregated[f"Recall@{tau}_mean"] = np.mean(Recalls) if Recalls else float('nan')

        # Aggregate evaluation counts.
        aggregated["num_samples"] = len(results)
        aggregated["num_samples_with_matches"] = sum(1 for r in results if r.num_matches > 0)
        aggregated["total_matches"] = sum(r.num_matches for r in results)

        return aggregated

    # Core metric calculations

    def _compute_MOM(self, match_overlaps: np.ndarray) -> float:
        """Compute Mean Overlap of Matches (MOM).

        MOM = (1/|M|) * Σ O[i,j] for (i,j) in M
        """
        if len(match_overlaps) == 0:
            return float('nan')
        return float(np.mean(match_overlaps))

    def _compute_BMR_WBMR(
        self,
        matches: List[Tuple[int, int]],
        overlap_matrix: np.ndarray,
    ) -> Tuple[float, float, int]:
        """Compute Best Match Recall (BMR) and its overlap-weighted variant.

        Let J*_i be the set of maximum-overlap targets for source region i. A
        best match is recovered when any ``(i, j)`` with ``j`` in J*_i appears
        in the predicted set M. BMR counts recovered valid source regions, and
        WBMR weights each source region by its maximum overlap.

        V contains source regions whose maximum target overlap is positive.

        Returns:
            (BMR, WBMR, |V|)
        """
        L, S = overlap_matrix.shape

        # Use a set for constant-time predicted-match lookup.
        match_set = set(matches)

        # Find the best-overlap target for each source region.
        best_match_indices = np.argmax(overlap_matrix, axis=1)  # [L]
        best_match_overlaps = np.max(overlap_matrix, axis=1)  # [L]

        # Only source regions with a positive-overlap target contribute.
        valid_mask = best_match_overlaps > 0
        num_valid = int(np.sum(valid_mask))

        if num_valid == 0:
            return float('nan'), float('nan'), 0

        # Count recovered best matches, weighted and unweighted.
        best_match_found = 0
        weighted_found = 0.0
        total_weight = 0.0

        for i in range(L):
            if not valid_mask[i]:
                continue

            j_star = best_match_indices[i]
            o_star = best_match_overlaps[i]

            # Treat any tied maximum-overlap target as a best match.
            max_overlap = o_star
            found = False

            # Search every target tied at the maximum overlap.
            for j in range(S):
                if overlap_matrix[i, j] == max_overlap and (i, j) in match_set:
                    found = True
                    break

            if found:
                best_match_found += 1
                weighted_found += o_star

            total_weight += o_star

        BMR = best_match_found / num_valid
        WBMR = weighted_found / total_weight if total_weight > 0 else float('nan')

        return float(BMR), float(WBMR), num_valid

    def _compute_HMQR(self, MOM: float, BMR: float) -> float:
        """Compute the harmonic mean of MOM and BMR (HMQR).

        HMQR = 2 * MOM * BMR / (MOM + BMR)
        """
        if np.isnan(MOM) or np.isnan(BMR):
            return float('nan')
        if MOM + BMR == 0:
            return 0.0
        return float(2 * MOM * BMR / (MOM + BMR))

    def _compute_coverage(self, matches: List[Tuple[int, int]], L: int) -> float:
        """Compute the fraction of source regions present in any prediction.

        Coverage = |{i : ∃j, (i,j) in M}| / L
        """
        if L == 0:
            return float('nan')
        matched_sources = set(i for i, j in matches)
        return float(len(matched_sources) / L)

    def _compute_AUC_ORC(
        self,
        matches: List[Tuple[int, int]],
        overlap_matrix: np.ndarray,
    ) -> float:
        """Approximate AUC-ORC over the configured overlap-threshold samples.

        The implementation averages defined recall values at
        ``orc_num_points`` thresholds linearly spaced from 0.05 to 0.95.
        """
        # Sample overlap thresholds away from the degenerate endpoints.
        taus = np.linspace(0.05, 0.95, self.orc_num_points)

        recalls = []
        for tau in taus:
            recall = self._compute_Recall_at_threshold(matches, overlap_matrix, tau)
            if not np.isnan(recall):
                recalls.append(recall)

        if len(recalls) == 0:
            return float('nan')

        return float(np.mean(recalls))

    # Threshold-dependent metric calculations

    def _compute_MP_at_threshold(self, match_overlaps: np.ndarray, tau: float) -> float:
        """Compute Matching Precision at overlap threshold ``tau``.

        MP@τ = |{(i,j) in M : O[i,j] >= τ}| / |M|
        """
        if len(match_overlaps) == 0:
            return float('nan')
        return float(np.mean(match_overlaps >= tau))

    def _compute_Recall_at_threshold(
        self,
        matches: List[Tuple[int, int]],
        overlap_matrix: np.ndarray,
        tau: float,
    ) -> float:
        """Compute recall over overlap-matrix entries above ``tau``.

        Recall@τ = |{(i,j) in M : O[i,j] >= τ}| / |{(i,j) : O[i,j] >= τ}|
        """
        # Count ground-truth region pairs above the overlap threshold.
        gt_positives = np.sum(overlap_matrix >= tau)

        if gt_positives == 0:
            # Return NaN so aggregation skips thresholds with no positive pairs.
            return float('nan')

        # Count predicted matches that satisfy the same overlap threshold.
        pred_positives = sum(1 for i, j in matches if overlap_matrix[i, j] >= tau)

        return float(pred_positives / gt_positives)

    # Match-quality distribution statistics

    def _compute_MQD_stats(self, match_overlaps: np.ndarray) -> Dict[str, float]:
        """Compute Match Quality Distribution (MQD) summary statistics."""
        if len(match_overlaps) == 0:
            return {
                'median': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'Q25': float('nan'),
                'Q75': float('nan'),
                'IQR': float('nan'),
            }

        Q25 = float(np.percentile(match_overlaps, 25))
        Q75 = float(np.percentile(match_overlaps, 75))

        return {
            'median': float(np.median(match_overlaps)),
            'std': float(np.std(match_overlaps)) if len(match_overlaps) > 1 else 0.0,
            'min': float(np.min(match_overlaps)),
            'max': float(np.max(match_overlaps)),
            'Q25': Q25,
            'Q75': Q75,
            'IQR': Q75 - Q25,
        }


# Convenience functions

def evaluate_bbox_matching(
    matches: List[Tuple[int, int]],
    overlap_matrix: Union[torch.Tensor, np.ndarray],
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Evaluate one image pair and return a flat metric dictionary.

    Args:
        matches: Predicted ``(source_index, target_index)`` pairs.
        overlap_matrix: Ground-truth overlap matrix with shape ``[L, S]``.
        thresholds: Optional overlap thresholds for MP and recall.

    Returns:
        Pair-level metrics and metadata.
    """
    evaluator = BBoxMatchingEvaluator(thresholds=thresholds)
    result = evaluator.evaluate(matches, overlap_matrix)
    return result.to_dict()


def evaluate_bbox_matching_batch(
    batch_matches: List[List[Tuple[int, int]]],
    batch_overlap_matrices: List[Union[torch.Tensor, np.ndarray]],
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Evaluate and aggregate a batch of image pairs.

    Args:
        batch_matches: Predicted matches for each image pair.
        batch_overlap_matrices: Ground-truth overlap matrix for each pair.
        thresholds: Optional overlap thresholds for MP and recall.

    Returns:
        Dataset-level aggregate metrics.
    """
    evaluator = BBoxMatchingEvaluator(thresholds=thresholds)
    results = evaluator.evaluate_batch(batch_matches, batch_overlap_matrices)
    return evaluator.aggregate_results(results)


def sim_auc(y_true, y_pred, epsilon=1e-7):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(np.unique(y_true)) == 1:
        if y_true[0] == 1:
            y_true = np.append(y_true, 0)
            y_pred = np.append(y_pred, np.min(y_pred) - epsilon)
        else:
            y_true = np.append(y_true, 1)
            y_pred = np.append(y_pred, np.max(y_pred) + epsilon)

    return roc_auc_score(y_true, y_pred)
