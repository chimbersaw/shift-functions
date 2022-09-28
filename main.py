from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from distribution import *
from shift import get_shift_function, get_shift_from_actual_function


def plot_pdf(name: str, filename: str, d: stats.rv_continuous, args: dict, n_plot: int = 10000):
    start = d.ppf(0.01, **args)
    end = d.ppf(0.99, **args)
    points = np.linspace(start, end, n_plot)

    plt.figure(figsize=(12, 9))
    plt.plot(points, d.pdf(points, **args))
    plt.title(f"{name} pdf", fontsize=25)
    plt.savefig(Path("img") / Path(f"{filename}-pdf.png"))
    plt.show()


def run(distr: Distribution, ns: List[int] = None, n_plot: int = 10000):
    run.counter = getattr(run, 'counter', 0) + 1
    if ns is None:
        ns = [10, 100, 1000, 10000]
    if distr.name == "SkewNormal":
        n_plot = 100  # otherwise too heavy(long) computations

    d, args = distr.d, distr.args
    name = f"#{run.counter}. {distr.name} distribution"
    filename = f"{distr.name}-{run.counter}"
    plot_pdf(name, filename, d, args, n_plot=n_plot)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    plt.suptitle(name, fontsize=25)
    plot_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for n, plot_index in zip(ns, plot_indices):
        xs = d.rvs(size=n, **args)
        ys = d.rvs(size=n, **args)
        shift = get_shift_function(ys, xs)
        shift_from_actual = get_shift_from_actual_function(xs, distr)

        quantiles = np.linspace(0, 1, n_plot)
        shifts = [shift(q) for q in quantiles]
        shifts_from_actual = [shift_from_actual(q) for q in quantiles]

        ax[plot_index].set_title(f"n = {n}")
        ax[plot_index].plot(quantiles, shifts, label="Two-sided shift function")
        ax[plot_index].plot(quantiles, shifts_from_actual, label="Shift from actual distr")

    plt.legend()
    plt.savefig(Path("img") / Path(f"{filename}.png"))
    plt.show()


def main():
    np.random.seed(239)

    distributions = [
        Normal(0, 1),
        Normal(-20.39, 3.22),
        Normal(13, 31.4),
        SkewNormal(10, 0, 1),
        Exponential(0, 1),
        Exponential(0, 5),
        Beta(0.5, 0.5),
        Beta(5, 1),
        Beta(1, 3),
        Beta(2, 2),
        Beta(2, 5),
        Cauchy(0, 1),
        Cauchy(15, 12.34),
        DoubleGamma(5),
        DoubleWeibull(6)
    ]

    for distr in distributions:
        run(distr)


if __name__ == "__main__":
    main()
