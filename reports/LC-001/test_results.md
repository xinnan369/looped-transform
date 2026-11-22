# LC-001 verification

## Environments and test results

| Environment | Dependencies | Result |
|---|---|---|
| Ubuntu 22.04 under WSL2, Python 3.11.11 | NumPy 1.26.4, SciPy 1.13.0, Pillow 10.4.0, pytest 8.3.5, jsonschema 4.26.0 | `pytest -q`: 26 passed in 4.52 s |
| Windows, Python 3.13.2 | NumPy 2.3.5, SciPy 1.17.1, Pillow 12.2.0, pytest 9.2.0, jsonschema 4.26.0 | `pytest -q`: 26 passed in 9.50 s, 1 third-party AnyIO deprecation warning |
| WSL2 CUDA smoke | PyTorch 2.2.2, torchvision 0.17.2, CUDA 12.1, NVIDIA RTX 3090 24 GiB (SM 8.6) | CUDA visible and one CUDA tensor allocation/reduction passed |

```sh
python -m pip install -e '.[test]'
pytest -q
python scripts/build_manifest.py --help
python scripts/audit_dataset.py --help
python scripts/prepare_p2pnet.py --help
pytest -q tests/test_audit.py -k cpu_cli_end_to_end
python -m jsonschema -i reports/LC-001/shanghaitech_a_audit.json reports/LC-001/data_audit.schema.json
python -m jsonschema -i reports/LC-001/shanghaitech_b_audit.json reports/LC-001/data_audit.schema.json
```

The CPU end-to-end tests generate ShanghaiTech MAT and NWPU JSON fixtures, convert them via the CLI, verify raw-to-manifest point counts, run both audit modes twice and check deterministic output and schema validity. The unit suite covers empty images, duplicates, missing files, invalid coordinates, crop boundaries, pooling, and 20,000 points assigned to a single cell. All three help commands returned 0. Both checked-in real JSON outputs validated with JSON Schema Draft 2020-12.

## Real data and baseline inputs

The following commands reproduce the aggregates after extracting the linked author archive described in [status.md](status.md). They explicitly request a local rejection log. The audits built in WSL2 and Windows matched byte-for-byte for both JSON and Markdown.

```sh
for part in A B; do
  dataset="shanghaitech_$(printf '%s' "$part" | tr '[:upper:]' '[:lower:]')"
  python scripts/build_manifest.py --config "configs/$dataset.json" --root "data/part_${part}_final" --output "artifacts/$dataset.jsonl" --out-of-bounds-rejections "artifacts/${dataset}_rejections.json"
  python scripts/audit_dataset.py --manifest "artifacts/$dataset.jsonl" --source real --aggregate-only --output-prefix "reports/LC-001/${dataset}_audit"
done
```

The [official NWPU JSON archive](https://huggingface.co/datasets/Junyu3/NWPU-Crowd/blob/main/jsons.zip) SHA-256 was `58d1aec986e4a4ce005015ad72f0aee0edda1c6eaa2a56a3ca2f85189f179e30`. The parser accepted 3,609 released labels with 1,488,690 points and 248 zero-head images. The official split parser returned 3,109 train IDs, 500 validation IDs and 1,500 test IDs, with no overlap. The archive's JSON IDs matched the train and validation union. Image-coordinate bounds and the real NWPU audit require the original images.

The P2PNet adapter produced the training and internal validation lists in WSL2. Part A used 270/30 images and Part B used 360/40. With the torchvision version-check correction in the baseline guide, the pinned upstream `python train.py --help` passed and its `loading_data` function loaded one actual training and one validation sample per part. Upstream emitted a deprecation warning for `upsample_bilinear`. Model metrics are `not_run`.

## Failures encountered

- Two initial tests failed. The conversion summary tried to count unlabeled test points, and one test expected the opposite seed-42 assignment for two fixture IDs. The summary now counts labeled points only and the test checks the actual deterministic assignment.
- Strict conversion rejected out-of-image points from the ShanghaiTech release as designed. Explicit logged exclusion then produced both real manifests. The exact reconciliation is in [status.md](status.md).
- The WSL partial clone stalled during checkout. The pinned GitHub source archive provided the same code for the upstream loader smoke test.
