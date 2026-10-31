import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.fetch.inventory import scan_series_folders, series_status


def _touch(path, n):
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        open(os.path.join(path, f"{i}.dcm"), "w").close()


def test_counts_dcm_files_per_series(tmp_path):
    _touch(tmp_path / "lidc_idri" / "LIDC-IDRI-0001" / "1.2.study" / "CT_1.2.3", 3)
    _touch(tmp_path / "lidc_idri" / "LIDC-IDRI-0002" / "1.2.study" / "CT_4.5.6", 2)
    (tmp_path / "lidc_idri" / "LIDC-IDRI-0002" / "1.2.study" / "CT_4.5.6" / "notes.txt").write_text("x")
    assert scan_series_folders(tmp_path) == {"1.2.3": 3, "4.5.6": 2}


def test_status_rules():
    assert series_status(0, 100) == "missing"
    assert series_status(40, 100) == "partial"
    assert series_status(100, 100) == "done"
    assert series_status(5, 0) == "done"  # unknown expected count
