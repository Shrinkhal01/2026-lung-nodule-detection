# Phase 1 · ground truth report (`scripts/02_ground_truth.py`)

- pylidc scans: **1018**, downloaded: **1018**, eligible (≤ 2.5 mm): **897** from **896** patients
- clusters: **2291** → ground truth **1192** (≥ 3 readers, ≥ 3.0 mm), ignore zones **1099**
- clusters with more than 4 marks (pylidc merged too much): **13**
- GT diameter (mm): min **4.5**, median **9.1**, 90th **21.8**, 99th **37.6**, max **46.8**
- GT nodules above 30 mm: **48**, above 40 mm: **5** (input for the Phase 2 size cap)

## Per split

| split | patients | scans | gt_nodules |
|---|---|---|---|
| test | 135 | 135 | 225 |
| train | 627 | 628 | 820 |
| val | 134 | 134 | 147 |

## Gates

| Gate | Result | Value |
|---|:-:|---|
| eligible scans within 5% of LUNA16's 888 | PASS | 897 (+1.0%) |
| GT nodules within 5% of LUNA16's 1,186 | PASS | 1192 (+0.5%) |
| 0 patients shared between splits | PASS | 0 |
| 0 scans failed clustering | PASS | 0 |

Runtime: 14 s
