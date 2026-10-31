import configparser
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config
from lungdet.fetch.pylidc_compat import apply_shims, write_pylidcrc


def test_config_expands_scratch(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/tmp/fake_scratch")
    cfg = load_config()
    assert cfg["paths"]["raw"] == "/tmp/fake_scratch/lidc_raw"
    assert cfg["ground_truth"]["min_readers"] == 3


def test_shims_restore_old_names():
    apply_shims()
    assert np.int is int and np.float is float
    assert configparser.SafeConfigParser is not None


def test_pylidcrc_format(tmp_path):
    p = write_pylidcrc("/data/lidc_idri", rc_path=str(tmp_path / "rc"))
    text = open(p).read()
    assert "[dicom]" in text and "path = /data/lidc_idri" in text
