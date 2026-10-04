"""Chapter 3: regenerates every figure in the chapter (grayscale)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from from_scratch import (grad_two_valleys, standardize, two_valleys,
                          used_cars)

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_line():
    t = np.array([1.0, 2, 3, 4, 5])
    y = np.array([18.0, 15, 14, 11, 9])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    tt = np.linspace(0.5, 5.5, 10)
    ax.plot(tt, 20 - 2.2 * tt, color="0.3", lw=0.9)
    for ti, yi in zip(t, y):
        ax.plot([ti, ti], [yi, 20 - 2.2 * ti], color="0.0", lw=1.4)
    ax.plot(t, y, "ko", ms=3.5)
    ax.set_xlabel("age (years)"); ax.set_ylabel("price (\\$1000)")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch03_line.pdf")


def fig_saddle():
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    g1, g2 = np.linspace(-1.7, 1.7, 300), np.linspace(-1.1, 1.3, 300)
    X, Y = np.meshgrid(g1, g2)
    Z = (X ** 2 - 1) ** 2 + Y ** 2
    ax.contour(X, Y, Z, levels=[0.1, 0.4, 0.8, 1.0, 1.3, 2, 3, 5],
               colors="0.5", linewidths=0.6)
    for start, style in (([0.0, 1.0], "-"), ([0.01, 1.0], "--")):
        path = [np.array(start)]
        minimize(two_valleys, start, jac=grad_two_valleys,
                 callback=lambda xk: path.append(xk.copy()))
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "k" + style, lw=1.0, marker=".",
                ms=3)
    ax.plot([-1, 1], [0, 0], "ko", ms=4)
    ax.plot(0, 0, "ks", ms=4, mfc="white")
    box = dict(facecolor="white", edgecolor="none", pad=1.0)
    ax.text(-1.0, -0.3, "minimum", fontsize=7, ha="center", bbox=box)
    ax.text(1.0, -0.3, "minimum", fontsize=7, ha="center", bbox=box)
    ax.text(0.0, -0.3, "saddle", fontsize=7, ha="center", bbox=box)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch03_saddle.pdf")


def fig_valley():
    age, km, price = used_cars()
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
    ax = axes[0]
    ax.plot(age, km / 1000, "k.", ms=2)
    ax.set_xlabel("age (years)"); ax.set_ylabel("distance (1000 km)")
    ax = axes[1]
    Z = np.column_stack([standardize(age), standardize(km)])
    y = price - price.mean()
    best = np.linalg.lstsq(Z, y, rcond=None)[0]
    g = np.linspace(-9, 2, 200)
    C1, C2 = np.meshgrid(g, g)
    F = sum((y[i] - C1 * Z[i, 0] - C2 * Z[i, 1]) ** 2
            for i in range(y.size)) / 2
    fmin = ((y - Z @ best) ** 2).sum() / 2
    ax.contour(C1, C2, F, levels=fmin + np.array([5, 20, 60, 150, 300]),
               colors="0.3", linewidths=0.6)
    ax.plot(*best, "ko", ms=3)
    ax.set_xlabel("coefficient of age"); ax.set_ylabel("coefficient of distance")
    ax.set_aspect("equal")
    for ax in axes:
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch03_valley.pdf")


if __name__ == "__main__":
    fig_line()
    fig_saddle()
    fig_valley()
