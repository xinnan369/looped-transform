# Executed resources

The CPU scope used the existing Windows and WSL Python environments. It checked local B manifest metadata, cached image paths, cached ResNet18 SHA-256 and historical CSVs. Tests used synthetic examples. CPU FLOP counts came from synthetic forwards. [Input audit](initialization_audit.json)

GPU preflight, project optimization, real-data sweeps, GPU latency and memory, hosted CI, official test, NWPU/WiderPerson and Stage 2 are `not_run`. External experiment downloads and real-data project updates are both zero. [Resource proposal](resource_plan.md)
