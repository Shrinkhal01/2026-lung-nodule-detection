"""Device choice + environment report. GPU is an accelerator, never a requirement (CUDA -> MPS -> CPU)."""
import logging
import os
import platform
import random
import sys

log = logging.getLogger(__name__)


def get_device():
    """'cuda', 'mps' or 'cpu'. FORCE_CPU=1 forces the CPU (used by the tests)."""
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        log.info("torch not installed; CPU-only code path")
    return "cpu"


def set_seed(seed=0):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def env_report():
    """Python, OS, device and the versions that matter, so every number can be traced to its hardware."""
    rep = {"python": sys.version.split()[0], "os": platform.platform(), "device": get_device()}
    for name in ("numpy", "scipy", "pandas", "torch", "pydicom", "gradio", "matplotlib"):
        try:
            rep[name] = __import__(name).__version__
        except ImportError:
            rep[name] = None
    return rep
