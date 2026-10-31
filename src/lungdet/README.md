# src/lungdet/

**The library.** One sub-folder per FINDERS layer, so the code reads in the same order as the pipeline.

| Folder | Layer | Phase |
|---|:-:|:-:|
| `common/` | — | 0 |
| `fetch/` | **F** Fetch | 0–1 |
| `isolate/` | **I** Isolate | 1 |
| `nominate/` | **N** Nominate | 2 |
| `discard/` | **D** Discard | 3 |
| `explain/` | **E** Explain | 4 |
| `rate/` | **R** Rate | 4 |

- **S**hip lives outside the library: `scripts/09_predict.py` and `app/`.
- **Rule:** library code never reads files by itself; scripts pass in the paths. That makes it **testable on fake data** on the Mac.
