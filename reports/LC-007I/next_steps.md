# Review handoff

Review the exact source commit, tree, config hash, paired exposure, CPU tests and phase-specific budget in the draft PR. Record the independent review before the frozen launcher can run. Use the owner's standing GPU permission for the reviewed experiment scope. Cached data and zero external downloads are required for this scope.

The only planned GPU run is seed-42 S4-shift-768 with fixed 512 validation. CPU archive the completed or partial attempt and retain Shared-2 aux+EMA as the formal baseline. Do not expand the run matrix based on these validation results.
