# Phase 2 · missed TRAIN nodules (`scripts/04b_miss_diagnosis.py`)

- missed: **59** of **820** TRAIN nodules, runtime 26 s

## Causes (old bundling)

| cause | nodules | median_diam | median_hu |
|---|---|---|---|
| bundling | 2 | 9.254999999999999 | -27.5 |
| cap | 56 | 7.775 | -520.0 |
| hu_rule | 1 | 8.24 | -743.0 |

## What a fix would recover (within the 3,000 cap)

| Change | Recovered | TRAIN recall would be |
|---|:-:|:-:|
| scale-aware bundling | 0 | 92.8% |
| scale-aware bundling + no HU rule | 0 | 92.8% |

- candidates per scan before the cap (these scans): old median **20860**, scale-aware median **22154**
