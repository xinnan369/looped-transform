# LC-004B follow-up

The frozen threshold diagnostic failed the preregistered count-gap criterion. Stop the claim that score calibration mainly explains the Shared-2 gap under this protocol.

A formal calibration task would reserve a partition before any model training. Retrain the compared arms on the same reduced training set, fit each global cutoff on the unseen calibration partition and freeze it before internal validation. This LC-004B fit reused images already seen by the models.

The same-checkpoint prefix comparison can motivate a separate intermediate-supervision hypothesis. Predefine the same shared core and head, a two-call early-plus-final loss arm against final-only loss, a fixed early-loss weight and matched compute and optimizer budgets. Use the unchanged count cutoff and candidate pool for the preregistered internal-validation endpoints. LC-004B ends with the frozen diagnostic.
