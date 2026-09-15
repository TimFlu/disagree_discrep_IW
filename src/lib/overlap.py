"""Shared hard selection and unnormalized importance-weighted risk."""
import numpy as np


def select_region(weights, threshold):
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(threshold) or threshold <= 0 or threshold > 1e6:
        raise ValueError('IW threshold must be in (0, 1e6].')
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be a finite nonnegative vector.')
    return weights <= threshold


def iw_contribution(errors, weights, mask):
    errors, weights, mask = np.asarray(errors), np.asarray(weights), np.asarray(mask, dtype=bool)
    if errors.ndim != 1 or errors.shape != weights.shape or errors.shape != mask.shape or not len(errors):
        raise ValueError('Errors, weights and mask must be aligned nonempty vectors.')
    if not np.isfinite(errors).all() or ((errors < 0) | (errors > 1)).any():
        raise ValueError('Expected errors in [0, 1].')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Expected finite nonnegative weights.')
    return float(np.sum(errors[mask] * weights[mask]) / len(errors))


def weight_diagnostics(weights, mask):
    weights = np.asarray(weights)[np.asarray(mask, dtype=bool)]
    total = weights.sum()
    return {
        'iw_weight_sum': float(total),
        'iw_effective_n': float(total ** 2 / np.square(weights).sum()) if total else 0.,
        'iw_weight_max': float(weights.max()) if len(weights) else np.nan,
        'iw_weight_p50': float(np.quantile(weights, .5)) if len(weights) else np.nan,
        'iw_weight_p95': float(np.quantile(weights, .95)) if len(weights) else np.nan,
    }
