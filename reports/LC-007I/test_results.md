# CPU verification

Execution freeze `7bf410956113184493b0d101694873b7b6219ac5`. All test processes hid CUDA. The WSL suites ran from an offline local bundle, and the shell launcher passed `bash -n`.

| Environment | Full pytest result | Local log SHA-256 |
| --- | --- | --- |
| WSL Python 3.10.16 | 202 passed in 69.04 s | `6e728aaefd2d19eabd017335fa45e89cf41cdef2b9b10baf5c225676ae7a8a45` |
| WSL Python 3.11.11 | 202 passed in 70.38 s | `134c05b1109d647f60d6425f7aee03233881f99a19ff4f46510b17cf4a3525da` |
| Windows Python 3.13.2 | 202 passed in 104.86 s, 12 existing dependency/API deprecation warnings | `d0abbf08c1a3219633950750d1e38115e153dbc60be1d7a11022e55153cf2bd9` |

Five added tests cover native crop geometry and padding, valid stride-4 masks, two-slot GT overflow, legacy crop identity and evaluation parity, larger-crop training with two decodes and fixed-size live/EMA validation, crop logging and resume guards, distinct authorization, time/update/OOM stops, partial archive handling and the strict nAP gate. The full suite includes the existing checkpoint and digest regressions.

The CPU preflight decoded the 400 cached images, rebuilt the historical initialization and generated paired 512/768 exposure and update hashes for all 40 epochs. Historical 512 crop, LR and GT/slot count parity passed for every epoch. The clean checkout reproduced the audit SHA-256 `091e9fe7e9763486569d291ff63cb4a3a45656c9cd990d3595f5c026ef1b03ef`. No GPU profile or real optimization ran.
