# CPU verification

All environments hid CUDA and used existing dependencies. The WSL suites ran at execution freeze `6a83290a2de0c502df0c5c8abd088f26a3b6b56e` in a clean checkout from a local bundle. The launcher passed shell syntax validation.

| Environment | Full suite | Local log SHA-256 |
| --- | --- | --- |
| WSL Python 3.10.16 | 208 passed in 81.47 s | `525f1710d77e3ebcd507848589ba64540a12a0621f65c355075c982c8d4eb88d` |
| WSL Python 3.11.11 | 208 passed in 81.86 s | `db5f9279d33dda0b367eb4093ec9a12f8d97efc64e7496d0ed55b61e9eb776e2` |
| Windows Python 3.13.2 | 208 passed in 120.98 s, 12 deprecation warnings | `db332d6df77da12004d4c841fec96a5bb35e839a7433362a768660f17d999e4e` |

Five new cases cover authorization isolation from LC-007I, actual archived 768 exposure and mismatch rejection, selecting the unshared trajectory, the four-cell arithmetic with a failed historical stability guard, and partial archiving without a joint result. The inherited suites cover crop geometry, validation isolation, loss reads, checkpoint compatibility, finite/update/time guards and quota rejection.

The CPU preflight decoded all 400 cached images, matched all 40 actual S768 exposure rows, checked legacy U512 and S512 order and LR, and confirmed matched-size nominal dense computation. The clean and working checkouts both produced audit SHA-256 `6de6c47f94bae2f67d7215e816be7d5fd0dcfbdfa0d4b74688b4ffc885c9fcca`. [Audit data](initialization_audit.json)
