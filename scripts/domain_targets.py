"""Targets for the bounded domain pilot; no disk maps and no implicit negatives."""
import math

import numpy as np


def validate_points(points, width, height):
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) > 512
            or not np.isfinite(values).all()
            or (values < 0).any() or (values[:, 0] >= width).any()
            or (values[:, 1] >= height).any()):
        raise ValueError("Head points must be finite coordinates inside the ROI")
    return values


def normalized_density(points, width, height, sigma=2.0):
    """One unit per annotation, including coincident points and clipped kernels.

    Coordinates are zero-based input pixels; output stride is two. New labels
    intentionally use a fixed kernel, distinct from historical adaptive targets.
    """
    if (type(width) is not int or type(height) is not int
            or width % 2 or height % 2 or not 32 <= min(width, height)
            or max(width, height) > 512 or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("Invalid bounded density geometry")
    points = validate_points(points, width, height)
    result = np.zeros((height//2, width//2), dtype=np.float32)
    radius = math.ceil(4*sigma)
    for px, py in points:
        cx, cy = px/2, py/2
        left, top = max(0, int(cx)-radius), max(0, int(cy)-radius)
        right, bottom = min(width//2, int(cx)+radius+1), min(height//2, int(cy)+radius+1)
        yy, xx = np.mgrid[top:bottom, left:right]
        kernel = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
        kernel /= kernel.sum(dtype=np.float64)
        result[top:bottom, left:right] += kernel.astype(np.float32)
    return result


def supervision_mask(points, uncertain, width, height, supervision, sigma=2.0):
    """Mask ambiguous heads; partial labels supervise only known positive kernels."""
    points = validate_points(points, width, height)
    uncertain = validate_points(uncertain, width, height)
    if supervision not in ("exhaustive", "negative", "partial_positive"):
        raise ValueError("Unknown supervision policy")
    if supervision == "negative" and (len(points) or len(uncertain)):
        raise ValueError("Negative regions cannot contain head candidates")
    target = normalized_density(points, width, height, sigma)
    mask = (target > 0) if supervision == "partial_positive" else np.ones(target.shape, dtype=bool)
    # Crop boundaries are not treated as certain background or complete heads.
    border = 16  # 32 input pixels
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    yy, xx = np.indices(target.shape)
    for px, py in uncertain:
        mask[(xx-px/2)**2+(yy-py/2)**2 <= (4*sigma)**2] = False
    if not mask.any():
        raise ValueError("No supervised pixels remain")
    return target, mask


def density_loss(prediction, target, mask, partial=False):
    """Den1 uses 100 model units/person. Partial positives can only push upwards."""
    import torch

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction/target/mask geometry mismatch")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Nonempty boolean supervision required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid density values")
    error = target*100-prediction
    if partial:
        error = error.clamp_min(0)
    return error[mask].square().mean()
