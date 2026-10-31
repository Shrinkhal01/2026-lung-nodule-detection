"""R2/R3: a small 3D CNN on 32 mm cubes -> one logit (nodule vs not)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class Small3DNet(nn.Module):
    """32^3 -> 16^3 -> 8^3 -> 4^3 -> global average -> logit (about 1.5 M weights)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), nn.MaxPool3d(2), _block(32, 64), nn.MaxPool3d(2),
                                      _block(64, 128), nn.MaxPool3d(2), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def focal_loss(logit, y, gamma=2.0, alpha=0.5):
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


def augment(x, rot90=True):
    """Random flips per sample on the 3 axes (+ one in-plane 90 deg rotation per batch). x: (n, 1, 32, 32, 32)."""
    for dim in (2, 3, 4):
        m = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1, 1), x.flip(dim), x)
    if rot90:
        x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    return x


@torch.no_grad()
def predict(model, cubes_i16, batch=4096, tta=False):
    """cubes on the GPU as int16 (n, 32, 32, 32) -> probabilities (n,) on the CPU."""
    model.eval()
    out = []
    flips = [()] if not tta else [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    for s in range(0, len(cubes_i16), batch):
        x = ((cubes_i16[s:s + batch].float().clamp(-1000, 400) + 1000) / 1400).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            p = torch.stack([torch.sigmoid(model(x.flip(f) if f else x).float()) for f in flips]).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()
