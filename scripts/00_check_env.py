"""Phase 0 · environment check. Runs the same on the Mac and the GPU server.

Usage:  <python> scripts/00_check_env.py
Prints: Python, every package we need (OK / MISSING), GPUs, free disk,
        and ONE pip command that installs only what is missing.
"""
import importlib
import os
import platform
import shutil
import sys

# import name -> pip name, and the phase that first needs it
PACKAGES = [
    ("numpy", "numpy", 1), ("scipy", "scipy", 1), ("pandas", "pandas", 1),
    ("yaml", "pyyaml", 0), ("tqdm", "tqdm", 0), ("pytest", "pytest", 0),
    ("idc_index", "idc-index", 0), ("pydicom", "pydicom", 1),
    ("pylidc", "pylidc", 1), ("SimpleITK", "SimpleITK", 1),
    ("skimage", "scikit-image", 1), ("sklearn", "scikit-learn", 3),
    ("xgboost", "xgboost", 3), ("torch", "torch", 3), ("shap", "shap", 4),
    ("matplotlib", "matplotlib", 4), ("gradio", "gradio", 5),
    ("monai", "monai", 3),  # stretch rung only
]


def check_packages():
    missing = []
    print("\n== Packages ==")
    for mod_name, pip_name, phase in PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            print(f"  OK       {pip_name:14s} {getattr(mod, '__version__', '?'):12s} (phase {phase})")
        except Exception as e:  # ImportError, or old code breaking on new numpy
            print(f"  MISSING  {pip_name:14s} {type(e).__name__}: {str(e)[:60]}")
            missing.append(pip_name)
    return missing


def check_gpu():
    print("\n== Compute ==")
    try:
        import torch
    except Exception:
        print("  torch not importable -> no GPU check")
        return
    if torch.cuda.is_available():
        print(f"  CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible GPU(s)")
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  free {free/2**30:.0f} / {total/2**30:.0f} GiB")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  Apple MPS available (Mac) -> small smoke runs only")
    else:
        print("  CPU only")


def check_disk():
    print("\n== Disk ==")
    for label, path in [("scratch", os.environ.get("SCRATCH", "")), ("home", os.path.expanduser("~"))]:
        if path and os.path.exists(path):
            free = shutil.disk_usage(path).free / 2**30
            print(f"  {label:8s} {path}: {free:,.1f} GiB free")
    for var in ("TMPDIR", "PIP_CACHE_DIR", "CUDA_VISIBLE_DEVICES"):
        print(f"  {var} = {os.environ.get(var, '(not set)')}")


def main():
    print("== Python ==")
    print(f"  {sys.version.split()[0]} at {sys.executable}  [{platform.system()} {platform.machine()}]")
    missing = check_packages()
    check_gpu()
    check_disk()
    print("\n== Summary ==")
    if missing:
        print(f"  {len(missing)} missing: {' '.join(missing)}")
        print(f"  install: {sys.executable} -m pip install {' '.join(missing)}")
        if "torch" in missing:
            print("  (torch on the server: install it alone first, it is ~2.5 GB)")
    else:
        print("  everything installed")


if __name__ == "__main__":
    main()
