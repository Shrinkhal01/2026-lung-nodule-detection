import os
import sys
from types import SimpleNamespace as Ann

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.fetch.ground_truth import assign_splits, summarise_cluster


def test_three_readers_make_ground_truth():
    anns = [Ann(centroid=(100, 200, 40), diameter=8.0), Ann(centroid=(102, 198, 42), diameter=10.0),
            Ann(centroid=(101, 202, 41), diameter=9.0)]
    r = summarise_cluster(anns, spacing=(0.7, 0.7, 2.5))
    assert r["role"] == "gt" and r["n_readers"] == 3 and r["diameter_mm"] == 9.0
    assert (r["i"], r["j"], r["k"]) == (101.0, 200.0, 41.0)
    assert (r["mm_i"], r["mm_j"], r["mm_k"]) == (70.7, 140.0, 102.5)


def test_two_readers_or_tiny_is_ignore():
    two = [Ann(centroid=(1, 1, 1), diameter=8.0)] * 2
    tiny = [Ann(centroid=(1, 1, 1), diameter=2.5)] * 4
    assert summarise_cluster(two, (1, 1, 1))["role"] == "ignore"
    assert summarise_cluster(tiny, (1, 1, 1))["role"] == "ignore"


def test_splits_by_patient():
    patients = [f"P{i:03d}" for i in range(100)] + ["P000", "P001"]   # two patients have 2 scans
    s = assign_splits(patients, (0.7, 0.15, 0.15), seed=42)
    assert len(s) == 100
    assert sorted(list(s.values()).count(x) for x in ("train", "val", "test")) == [15, 15, 70]
    assert s == assign_splits(patients, (0.7, 0.15, 0.15), seed=42)   # same seed -> same split
