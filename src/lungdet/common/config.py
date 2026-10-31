"""Load configs/default.yaml and expand $SCRATCH-style variables in every string."""
import os

import yaml


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "default.yaml")
    with open(path) as fh:
        return _expand(yaml.safe_load(fh))
