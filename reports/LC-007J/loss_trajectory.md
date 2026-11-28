# Fixed late loss comparison

The absolute 3% guard compares the equally weighted mean final training loss at E31–35 and E36–40. The 512 run moved from .391311 to .382477 (−2.26%). The 768 run moved from .270919 to .259742 (−4.13%) and exceeded the guard. It finished at .256955, after a small increase from E38 to E39. The fall in the average loss is descriptive and does not itself establish instability. The 768 sample MAE CV of 5.70% passed its separate 10% limit. The 768 joint failure remains unchanged.

The compared crops differ in sampled GT and valid pixels. Within-run change is therefore the guard used here. [Epoch losses and guard calculation](evidence_summary.json) · [Original decision](../LC-007I/training_results/research_decision.json)
