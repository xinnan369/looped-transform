# CPU audit verification

Four regression tests passed in each of WSL Python 3.10, WSL Python 3.11 and Windows Python 3.13. The tests reject malformed count or cancellation identities, duplicate and missing coverage, and reordered image IDs. They also exercise undefined precision/recall denominators and the signed-error decomposition.

| Log | SHA-256 |
| --- | --- |
| WSL Python 3.10 | `7abac67c3cceb2084525d7ddb489e8f7971a0c333e0a75d67d8c7219a6e9627b` |
| WSL Python 3.11 | `8da42575b3b9a53ef64fcdc917aad1469f669aa94911d6c53b5762e082669e00` |
| Windows Python 3.13 | `1e4b0652bffe48e2678647e345854d1b268011f5037406e13193e1457275e79a` |

Re-running the complete local audit produced identical file bytes for both tables and the evidence summary. The command uses only the Python standard library. It reads the checkpoint bytes for SHA-256 under `--inspect-local`, together with the prior E40 artifact, with no model load, inference or optimizer step.

```bash
python -m scripts.audit_lc007j --output reports/LC-007J --inspect-local
```
