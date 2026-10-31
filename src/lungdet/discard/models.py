"""Which trained models exist, and how to load and run them (used by Phase 4 and 5)."""
import json
import os

import numpy as np
import torch

from lungdet.discard.cnn import Small3DNet, predict
from lungdet.discard.hessian import hessian_channels


def model_specs(cfg):
    """{name: (rung, suffix, seeds, in_ch)}: R2 baseline, R5 = HUSH, and the FINAL model chosen by the length check."""
    seeds = cfg["discard"]["seeds"]
    specs = {"R2": ("r2", "", seeds, 1), "R5": ("r5", "", seeds, 3)}
    f = os.path.join(cfg["paths"]["results"], "check", "final_choice.json")
    ch = json.load(open(f)) if os.path.exists(f) else {"rung": "r5", "suffix": "", "seeds": seeds}
    final = (ch["rung"], ch["suffix"], ch["seeds"], 3 if ch["rung"] == "r5" else 1)
    if final[:2] not in [(v[0], v[1]) for v in specs.values()]:
        specs["FINAL"] = final
    return specs, f"{ch['rung'].upper()}{ch['suffix']}"


def load_ensemble(cfg, spec, device):
    rung, suffix, seeds, in_ch = spec
    models = []
    for s in seeds:
        m = Small3DNet(in_ch).to(device)
        m.load_state_dict(torch.load(os.path.join(cfg["paths"]["checkpoints"], f"{rung}{suffix}_seed{s}.pth"),
                                     map_location=device))
        m.eval()
        models.append(m)
    sig = tuple(cfg["hush"]["hessian_sigmas_mm"])
    transform = (lambda x: hessian_channels(x, sig)) if in_ch == 3 else None
    return models, transform


def ensemble_predict(models, transform, cubes_i16, device, batch=None):
    batch = batch or (1024 if transform is not None else 4096)
    return np.mean([predict(m, cubes_i16, batch=batch, transform=transform, device=device) for m in models], axis=0)
