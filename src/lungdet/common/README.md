# common/

Shared helpers, written in **Phase 0**.

- **device** → CUDA → Apple MPS → CPU fallback, so the same code runs on the server **and** the Mac.
- **paths** → loads `configs/default.yaml`.
- **seeds** → fixes random seeds (42–46), because results must be repeatable within GPU noise.
- **logging** → every run writes its config and git commit next to its results.
