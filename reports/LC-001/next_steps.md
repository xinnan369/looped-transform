# After LC-001 review

1. Obtain the original NWPU-Crowd image archives. Run the strict converter and reconcile any exclusions with the official labels before relying on capacity statistics. Audit the official training and validation sets. Keep hidden test labels and predictions outside model selection.
2. Inspect crop sensitivity from multiple fixed seeds and compare capacity across training subgroups. LC-001 uses one crop per image and the audit measures slot overflow directly. Freeze the training-data label policy, stride and candidate count before any model comparison.
3. Run the pinned upstream P2PNet baseline in a recorded WSL CUDA environment when the experiment is approved. Freeze the checkpoint and threshold using validation. Add a verified localization evaluator and record the label policy before comparing MAE, RMSE or nAP.
4. After LC-001 acceptance, specify the fixed-loop 0/1/2/4 and non-shared-control comparisons in LC-002. Record RTX 3090 peak memory and step time on the common 512 × 512 setup.
