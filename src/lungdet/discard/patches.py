"""32 mm cubes around candidates. A cube is cut so the candidate sits at index 16 (size // 2)."""
import numpy as np


def crop_cubes(hu, centers, size=32, pad=-1000):
    """hu: 3D int16 volume; centers: (n, 3) voxel indices inside it. Returns (n, size, size, size) int16."""
    h = size // 2
    padded = np.pad(hu, h, mode="constant", constant_values=pad)
    centers = np.clip(np.asarray(centers, int).reshape(-1, 3), 0, np.array(hu.shape) - 1)
    out = np.empty((len(centers), size, size, size), np.int16)
    for n, (i, j, k) in enumerate(centers):
        out[n] = padded[i:i + size, j:j + size, k:k + size]   # padded index c + h is the cube centre
    return out


def normalise(cubes):
    """HU -> [0, 1] with the window [-1000, 400]."""
    x = np.clip(np.asarray(cubes, np.float32), -1000, 400)
    return (x + 1000.0) / 1400.0
