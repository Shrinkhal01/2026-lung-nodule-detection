# data/

**No data in git.** Real data lives on the GPU server:

```text
/scratch/mtech_shrinkhal/
├── lidc_raw/        ← CT scans (DICOM) from IDC, about 124 GB if all are downloaded
├── cache/           ← HU, 1 mm, lung mask per scan (.npz)
├── candidates/      ← candidates.csv + 32 mm cubes
└── checkpoints/     ← trained models (.pth)
```

- **Why not on the Mac:** 8 GB RAM and little free disk. The Mac only runs tests on fake scans.
- **Why not in `/home` on the server:** it is 100% full and shared.
- **Licence:** LIDC-IDRI is CC BY 3.0. Cite https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
