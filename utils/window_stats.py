"""Training-input statistics used to standardize EvoST inputs (the `norm` entry of a bundle).

Copied unchanged from the original benchmark entry point, where the function was private; the maintenance and
annual bundles store the mean and standard deviation it returns for the training split of each year.
"""
import numpy as np


def window_input_stats(raw_split: np.ndarray, x_len: int, y_len: int):
    """Mean/std over the exact input windows without materialising them.

    ``generate_dataset`` emits ``len(raw_split) - x_len - y_len`` overlapping
    windows in reverse order.  Window order does not affect the statistics, so
    temporal occurrence counts are used as weights here.
    """
    n_windows = len(raw_split) - x_len - y_len
    if n_windows <= 0:
        raise ValueError(
            f"split length {len(raw_split)} is too short for x={x_len}, y={y_len}"
        )
    weights = np.convolve(
        np.ones(n_windows, dtype=np.float64),
        np.ones(x_len, dtype=np.float64),
    )
    used = np.asarray(raw_split[: len(weights)], dtype=np.float64)
    finite = np.isfinite(used)
    weighted = weights[:, None]
    count = float((weighted * finite).sum())
    if count <= 0:
        return 0.0, 1.0
    total = float((np.where(finite, used, 0.0) * weighted).sum())
    total_sq = float((np.where(finite, used * used, 0.0) * weighted).sum())
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = var ** 0.5
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


_window_input_stats = window_input_stats    # original name
