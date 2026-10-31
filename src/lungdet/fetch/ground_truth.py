"""Ground truth from the readers' marks (pure logic; pylidc objects are passed in)."""
import numpy as np


def summarise_cluster(anns, spacing, min_readers=3, min_diameter_mm=3.0):
    """One nodule = one cluster of reader marks. Centre and diameter = mean over the readers."""
    ijk = np.mean([np.asarray(a.centroid, dtype=float) for a in anns], axis=0)
    diameter = float(np.mean([a.diameter for a in anns]))
    mm = ijk * np.asarray(spacing, dtype=float)
    gt = len(anns) >= min_readers and diameter >= min_diameter_mm
    return {"n_readers": len(anns), "diameter_mm": round(diameter, 2),
            "i": round(ijk[0], 2), "j": round(ijk[1], 2), "k": round(ijk[2], 2),
            "mm_i": round(mm[0], 2), "mm_j": round(mm[1], 2), "mm_k": round(mm[2], 2),
            "role": "gt" if gt else "ignore"}


def assign_splits(patient_ids, fractions=(0.70, 0.15, 0.15), seed=42):
    """Patient -> 'train' / 'val' / 'test'. Every scan of a patient follows the patient."""
    patients = np.array(sorted(set(patient_ids)))
    perm = np.random.default_rng(seed).permutation(len(patients))
    n_train = int(round(fractions[0] * len(patients)))
    n_val = int(round(fractions[1] * len(patients)))
    split = {}
    for rank, p in enumerate(patients[perm]):
        split[p] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return split
