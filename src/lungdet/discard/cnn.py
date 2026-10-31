"""3D CNNs on 32 mm cubes.
R2/R3: Small3DNet(in_ch=1).  HUSH: Small3DNet(in_ch=3 [HU, blobness, vesselness], size_head=True/False)
U (pretraining): Restorer = the SAME encoder + a decoder that fills in hidden blocks of the cube.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


def encoder(in_ch=1):
    """32^3 -> 16^3 -> 8^3 -> 4^3, 256 channels."""
    return nn.Sequential(_block(in_ch, 32), nn.MaxPool3d(2), _block(32, 64), nn.MaxPool3d(2),
                         _block(64, 128), nn.MaxPool3d(2), _block(128, 256))


class Small3DNet(nn.Module):
    """Encoder -> global average -> logit (+ optional size head: log diameter in mm)."""

    def __init__(self, in_ch=1, size_head=False):
        super().__init__()
        self.features = encoder(in_ch)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3))
        self.cls = nn.Linear(256, 1)
        self.size = nn.Linear(256, 1) if size_head else None

    def forward_both(self, x):
        z = self.pool(self.features(x))
        return self.cls(z).squeeze(1), (self.size(z).squeeze(1) if self.size is not None else None)

    def forward(self, x):
        return self.forward_both(x)[0]

    def load_state_dict(self, state, strict=True):
        # checkpoints from Phase 3 stored the classifier as head.3.*; map them to cls.*
        state = {k.replace("head.3.", "cls."): v for k, v in state.items()}
        return super().load_state_dict(state, strict=strict)


class Restorer(nn.Module):
    """Masked cube restoration (U): the encoder sees a cube with ~60% of its 4^3 blocks hidden, the decoder redraws HU."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.features = encoder(in_ch)

        def up(cin, cout):
            return nn.Sequential(nn.ConvTranspose3d(cin, cout, 2, stride=2), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                                 nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(up(256, 128), up(128, 64), up(64, 32), nn.Conv3d(32, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.decoder(self.features(x)))


def block_mask(n, ratio=0.6, block=4, size=32, device="cpu"):
    """(n, 1, size, size, size) float mask: 1 = hidden. Hides whole block^3 cubes."""
    g = size // block
    m = (torch.rand(n, 1, g, g, g, device=device) < ratio).float()
    return F.interpolate(m, scale_factor=block, mode="nearest")


def focal_loss(logit, y, gamma=2.0, alpha=0.5):
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


def augment(x, rot90=True):
    """Random flips per sample on the 3 axes (+ one in-plane 90 deg rotation per batch). x: (n, c, 32, 32, 32)."""
    for dim in (2, 3, 4):
        m = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1, 1), x.flip(dim), x)
    if rot90:
        x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    return x


def to_unit(cubes_i16):
    """int16 HU cubes (n, 32, 32, 32) -> float (n, 1, 32, 32, 32) in [0, 1] (window -1000..400)."""
    return ((cubes_i16.float().clamp(-1000, 400) + 1000) / 1400).unsqueeze(1)


@torch.no_grad()
def predict(model, cubes_i16, batch=4096, tta=False, transform=None):
    """cubes on the GPU as int16 -> probabilities (n,) on the CPU. transform: e.g. Hessian channels (+ ablation)."""
    model.eval()
    out = []
    flips = [()] if not tta else [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    for s in range(0, len(cubes_i16), batch):
        x = to_unit(cubes_i16[s:s + batch])
        if transform is not None:
            x = transform(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            p = torch.stack([torch.sigmoid(model(x.flip(f) if f else x).float()) for f in flips]).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()
