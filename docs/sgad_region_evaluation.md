# Region Matching Metrics

The framework evaluates region correspondences against a geometric
ground-truth overlap matrix. Metrics measure overlap quality, best-match
recall, and source-region coverage.

## Metrics

| Metric | Definition |
|---|---|
| MOM | Mean ground-truth overlap of predicted matches. |
| BMR | Fraction of source regions with positive ground-truth overlap whose prediction selects any maximum-overlap target. |
| WBMR | BMR weighted by each source region's maximum ground-truth overlap. |
| HMQR | Harmonic mean of MOM and BMR. |
| MP@τ | Fraction of predicted matches with ground-truth overlap ≥ τ; defaults: 0.2, 0.4, 0.6. |
| Recall@τ | Fraction of ground-truth pairs with overlap ≥ τ that are predicted. |
| Coverage | Fraction of source regions included in at least one predicted match. |
| AUC-ORC | Mean defined Recall@τ over 20 uniformly spaced thresholds from 0.05 to 0.95. |
| MQD | Overlap distribution: median, standard deviation, quartiles, interquartile range, minimum, and maximum. |

Core and threshold metrics are computed per image pair and averaged over
defined values for each model, excluding NaN. MQD is reported per pair.

| Condition | Metric values |
|---|---|
| No predictions, nonempty overlap matrix | MOM, MP, and HMQR are NaN; Coverage is 0; recall is 0 where ground-truth positives exist. |
| No positive ground-truth overlap | BMR and WBMR are NaN. |
| No ground-truth positives at τ | Recall@τ is NaN. |
| Empty overlap matrix | Coverage and Recall@τ are 0; other metrics are NaN. |
