# Metric verification

| Metric or operation | Verified case | Expected result |
|---|---|---|
| MAE / RMSE | Predictions 0, 7 against references 1, 4 | MAE 2, RMSE √5 |
| Local point AP | Confidence order gives FP, TP, duplicate FP, TP for two heads | Precision-envelope AP 0.5 |
| Strict normalized threshold | A prediction is 5 px from each of two GT points 10 px apart | AP 0 at δ=0.50 and 0.5 at δ=0.51 |
| Empty and coincident GT | No labels across images, and one exact prediction for two coincident labels | Null AP on empty GT and AP 0.5 for coincident labels |
| Image isolation | A prediction in an empty image sits at a GT coordinate in another image | No cross-image match |
| Decode and stitching | Known fractional offsets and irregular final tile overlap | Native coordinates recovered, complete disjoint ownership and highest score kept among radius duplicates |
| Full-frame duplicate suppression | Two retained detections straddle an ownership seam by 1 px | Exactly one point is counted, with the higher-confidence coordinates |

The validation tests exercise the actual functions used by training and evaluation. Score floors, matching ties, sparse and coincident ground truth, and AP integration are specified in the [protocol](../../docs/DATA_AND_EVAL_PROTOCOL.md). The local nAP is reported as `nap_local_v1`. Upstream numeric parity has not been verified. NWPU official F1 is `not_ready`. The official ShanghaiTech test sets were not scored in LC-002.
