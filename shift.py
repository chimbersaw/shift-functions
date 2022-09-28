import numpy as np

from distribution import Distribution


def quantile(xs: np.ndarray, alpha: float):
    n = len(xs)
    k = int(alpha * (n - 1))
    k_succ = k + 1 if k < n - 1 else k
    if k + 1 > alpha * n:
        return xs[k]
    elif k + 1 == alpha * n:
        return (xs[k] + xs[k_succ]) / 2
    else:
        return xs[k_succ]


def get_shift_function(xs: np.ndarray, ys: np.ndarray):
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)

    def shift_function(alpha: float):
        return quantile(xs_sorted, alpha) - quantile(ys_sorted, alpha)

    return shift_function


def get_shift_from_actual_function(xs: np.ndarray, distribution: Distribution):
    xs_sorted = np.sort(xs)

    def shift_from_actual_function(alpha: float):
        return distribution.d.ppf(alpha, **distribution.args) - quantile(xs_sorted, alpha)

    return shift_from_actual_function
