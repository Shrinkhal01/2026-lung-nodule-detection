"""pylidc (2016-2020 code) on today's Python/NumPy.

Two names it uses are gone:
- np.int / np.float / np.bool  -> removed in NumPy 1.24
- configparser.SafeConfigParser -> removed in Python 3.12
We put the old names back BEFORE importing pylidc. Nothing else changes.
"""
import configparser
import os

import numpy as np

_NUMPY_ALIASES = {"int": int, "float": float, "bool": bool, "object": object}


def apply_shims():
    for name, typ in _NUMPY_ALIASES.items():
        if name not in np.__dict__:
            setattr(np, name, typ)
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser


def import_pylidc():
    apply_shims()
    import pylidc
    return pylidc


def write_pylidcrc(dicom_root, rc_path=None):
    """pylidc reads the DICOM root from ~/.pylidcrc. Returns the path written."""
    rc_path = rc_path or os.path.expanduser("~/.pylidcrc")
    with open(rc_path, "w") as fh:
        fh.write(f"[dicom]\npath = {dicom_root}\nwarn = True\n")
    return rc_path

