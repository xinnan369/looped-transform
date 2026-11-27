# E40 read-only diagnostic resource audit

The approved one-attempt diagnostic ran on 2026-11-27 from 09:20:25 to 09:21:07 Asia/Shanghai and exited with code 0. The exact execution head was `856dfb30eec4987849f8809575382f59b750c21f` and the execution config SHA-256 was `2309902dbe1d059981ec190635dd55bd461a2dff27ebd3cb012539fe495cb064`. The frozen WSL launcher enforced the 300 s envelope. [Issue authorization and limits](https://github.com/xinnan369/looped-transform/issues/31) · [Launch review](https://github.com/xinnan369/looped-transform/pull/32#pullrequestreview-6082211195) · [Local authorization record](e40_authorization.json)

| Measure | Observed |
| --- | ---: |
| Charged time, from runner preflight through cleanup | 39.766867 s |
| Launcher wall time, clock rounded to seconds | 42 s |
| Four synchronized case phases, including analysis and parity | 9.542 / 9.561 / 10.110 / 9.379 s |
| GPU peak allocated | 195,512,320 bytes (186.46 MiB) |
| GPU peak reserved | 234,881,024 bytes (224 MiB) |
| CUDA runtime free / total at runner preflight | 24,436,015,104 / 25,769,279,488 bytes |
| WSL `nvidia-smi` free before / after | 18,808 / 18,787 MiB |
| Completed checkpoint sweeps | 4 / 4 |
| Image evaluations | 160 |
| Training updates / external download bytes | 0 / 0 |

CUDA runtime memory and `nvidia-smi` are recorded from separate interfaces and points in time. The same background compute PID 163671 appeared before and after, with no intervention or memory quota. No OOM, timeout, retry, extra sweep, dataset transfer, or hosted CI occurred.

The checks passed on all four checkpoints for exact input provenance, count and assignment parity, archived AP at every saved threshold, all 4,295 GT per checkpoint, model tensors including BN buffers, CPU/device RNG, and checkpoint immutability. The GPU run retained the four per-image JSONL files and execution logs locally. The repository contains their SHA-256 digests, file sizes, per-case state hashes, compact metrics, and phase accounting. [Checkpoint/output index](epoch_index.csv) · [Execution evidence](e40_execution_evidence.json)
