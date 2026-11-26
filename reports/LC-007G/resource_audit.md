# Executed CPU resources

Used existing local Windows and WSL environments with CUDA hidden, generated test tensors and images, cached B metadata and cached ResNet18 weights. The CPU audit verifies historical CSVs, full-state/RNG parity and matched crop/LR trajectories. The reachability probe uses model-generated synthetic tokens only. [Input audit](initialization_audit.json)

GPU preflight, real-data optimization and evaluation, GPU latency, process VRAM, hosted CI, official test, NWPU/WiderPerson and Stage 2 are `not_run`. External experiment downloads and real project optimizer updates are zero.
