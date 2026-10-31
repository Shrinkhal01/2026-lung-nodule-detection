# Decisions log (Claude's working log; your own notes go in `phases/PHASE_N.md`)

## Phase 0: setup

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **idc-index** download from the public IDC bucket | no login, parallel (s5cmd), resumable | NBIA Data Retriever (GUI + login) | speed test 4.94 MB/s; full run **25.9 MB/s** → 128 GB in ~1.5 h |
| **/scratch** for data, env, caches | `/home` is shared and 100% full | `/home`; streaming through RAM only | `/home` 0.4 GiB free vs `/scratch` 2,683 GiB free |
| **Reuse the old venv** (copied to scratch) | it already has torch 2.14.1 + CUDA 13 | fresh torch install | torch ≈ 2.5 GB at pip's ~0.1–0.5 MB/s = hours |
| **pip `-c keep.txt`** (pins torch + numpy) | new packages must not silently replace torch | plain `pip install` | NOT MEASURED (prevention) |
| **`setuptools<81`** | pylidc imports `pkg_resources`, which newer setuptools removed | patching pylidc | import failed: `No module named 'pkg_resources'` |
| **pylidc shims** (`np.int`, `SafeConfigParser`) | old pylidc code on NumPy 2 / Python 3.13 | pinning old NumPy (breaks torch) | pylidc issue #69 (`np.int`) |
| **Public repo** + https clone | the server needs no GitHub key | private repo + deploy key | clone asked for a username while private |
| **Mac: project `.venv`, test kit only** | tests run on fake scans; no CT data on the Mac | full server stack on the Mac | Mac: 8 GB RAM, 22 GiB free |
