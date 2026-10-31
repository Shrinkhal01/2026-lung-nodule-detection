"""R1 features: hand-made numbers per candidate cube (the candidate sits at index 16)."""
import numpy as np

FEATURES = ["ball_mean", "ball_std", "ball_max", "ball_dense", "shell_mean", "contrast",
            "centre_hu", "cube_mean", "cube_std", "cube_dense"]
_g = np.indices((32, 32, 32)) - 16
_DIST = np.sqrt((_g ** 2).sum(0))


def cube_features(cubes, sigma, chunk=20000):
    """cubes (n, 32, 32, 32) int16, sigma (n,) blob size in mm -> (n, 10) float32."""
    sigma = np.asarray(sigma, float)
    out = np.zeros((len(cubes), len(FEATURES)), np.float32)
    radius = np.clip(np.rint(np.sqrt(3) * sigma), 2, 12).astype(int)
    for start in range(0, len(cubes), chunk):
        x = np.clip(np.asarray(cubes[start:start + chunk], np.float32), -1000, 400)
        rr = radius[start:start + chunk]
        flat = x.reshape(len(x), -1)
        o = out[start:start + chunk]
        o[:, 6] = x[:, 16, 16, 16]
        o[:, 7], o[:, 8], o[:, 9] = flat.mean(1), flat.std(1), (flat > -400).mean(1)
        for r in np.unique(rr):
            sel = rr == r
            ball = (_DIST <= r).ravel()
            shell = ((_DIST > r) & (_DIST <= 2 * r)).ravel()
            b, s = flat[sel][:, ball], flat[sel][:, shell]
            o[sel, 0], o[sel, 1], o[sel, 2] = b.mean(1), b.std(1), b.max(1)
            o[sel, 3], o[sel, 4] = (b > -400).mean(1), s.mean(1)
            o[sel, 5] = o[sel, 0] - o[sel, 4]
    return out
