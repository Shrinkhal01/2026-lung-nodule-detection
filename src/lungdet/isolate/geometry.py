"""Coordinates and resampling. ONE rule for the whole project:

    voxel index v sits at  v * spacing  millimetres   (the first voxel's centre = 0 mm)

so a point at m mm lands at index m / new_spacing in the resampled grid.
Image volume and nodule centres both go through this rule, so they can never drift apart.
"""
import numpy as np
from scipy import ndimage


def voxel_to_mm(ijk, spacing):
    return np.asarray(ijk, dtype=float) * np.asarray(spacing, dtype=float)


def mm_to_index(mm, new_spacing=1.0):
    return np.asarray(mm, dtype=float) / float(new_spacing)


def resampled_shape(shape, spacing, new_spacing=1.0):
    """Last input voxel sits at (n-1)*s mm -> keep every output voxel that lies inside."""
    return tuple(int(np.floor((n - 1) * s / new_spacing + 1e-6)) + 1 for n, s in zip(shape, spacing))


def resample(vol, spacing, new_spacing=1.0, order=1, cval=-1000.0):
    """Output voxel o reads the input at o * new_spacing / spacing (trilinear when order=1)."""
    scale = float(new_spacing) / np.asarray(spacing, dtype=float)
    return ndimage.affine_transform(
        np.asarray(vol, dtype=np.float32), scale, offset=0.0,
        output_shape=resampled_shape(vol.shape, spacing, new_spacing),
        order=order, mode="constant", cval=cval)
