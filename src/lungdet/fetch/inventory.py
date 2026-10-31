"""Which LIDC series are already on disk? Pure file-system logic, no network.

idc-index stores a series as  <raw>/<collection>/<PatientID>/<StudyUID>/CT_<SeriesUID>/*.dcm
so the folder name, minus its 'CT_' prefix, is the series UID.
"""
import os


def scan_series_folders(raw_dir):
    """Walk raw_dir once -> {series_uid: number of .dcm files}."""
    counts = {}
    for dirpath, _dirs, files in os.walk(raw_dir):
        n = sum(f.endswith(".dcm") for f in files)
        if n:
            name = os.path.basename(dirpath)
            uid = name.split("_", 1)[1] if name.startswith("CT_") else name
            counts[uid] = counts.get(uid, 0) + n
    return counts


def series_status(n_on_disk, n_expected):
    """'done' / 'partial' / 'missing'. n_expected <= 0 means unknown -> any file counts as done."""
    if n_on_disk == 0:
        return "missing"
    if n_expected <= 0 or n_on_disk >= n_expected:
        return "done"
    return "partial"
