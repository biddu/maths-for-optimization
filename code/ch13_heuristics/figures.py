"""Chapter 13: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import rosen, searches
from from_scratch import nearest_neighbor, two_opt, tour_length
from in_practice import (BOX_HIGH, BOX_LOW, cities, route, save_cache,
                         validation_loss)

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def simplexes(f, x0, size=0.5, count=22):
    """The first few triangles of Nelder-Mead (reflect, expand,
    contract, shrink), recorded for the picture."""
    pts = [np.array(x0, float)] + [np.array(x0, float) + size * e
                                   for e in np.eye(2)]
    out = []
    for _ in range(count):
        pts.sort(key=f)
        out.append(np.array(pts))
        center = (pts[0] + pts[1]) / 2
        step = center - pts[2]
        trial = center + step
        if f(trial) < f(pts[0]):
            far = center + 2 * step
            trial = far if f(far) < f(trial) else trial
        elif f(trial) >= f(pts[1]):
            trial = center - 0.5 * step
            if f(trial) >= f(pts[2]):
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                continue
        pts[2] = trial
    return out


def fig_simplex():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    u, v = np.meshgrid(np.linspace(-1.8, 1.4, 300), np.linspace(-0.4, 2.0, 300))
    ax.contour(u, v, rosen([u, v]), levels=[1, 4, 15, 50, 150, 400],
               colors="0.7", linewidths=0.5)
    for k, tri in enumerate(simplexes(rosen, [-1.2, 1.0])):
        shade = str(0.65 - 0.65 * k / 21)
        ax.fill(tri[:, 0], tri[:, 1], fc="none", ec=shade, lw=0.7)
    ax.plot(1, 1, "k+", ms=8)
    ax.plot(-1.2, 1.0, "ko", ms=3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch13_simplex.pdf")


def fig_tours():
    P, D = cities(40, 13)
    nn = nearest_neighbor(D)
    tours = [("nearest stop first", nn), ("after 2-opt", two_opt(nn, D)),
             ("optimal", route(D, 5))]
    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.9))
    for ax, (name, tour) in zip(axes, tours):
        loop = np.append(tour, tour[0])
        ax.plot(P[loop, 0], P[loop, 1], "k-", lw=0.7)
        ax.plot(P[:, 0], P[:, 1], "o", color="0.3", ms=1.8)
        ax.set_title(f"{name}: {tour_length(tour, D):.0f} km", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ch13_tours.pdf")


def fig_search():
    steps = np.linspace(BOX_LOW[0], BOX_HIGH[0], 15)
    decays = np.linspace(BOX_LOW[1], BOX_HIGH[1], 17)
    fine = np.array([[validation_loss((a, b)) for b in decays]
                     for a in steps])
    found = searches(0)
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.2), sharey=True)
    for ax, name in zip(axes, ("random", "Bayesian")):
        cs = ax.contourf(steps, decays, fine.T,
                         levels=[0.32, 0.34, 0.38, 0.45, 0.55, 0.7, 2.1],
                         cmap="Greys", alpha=0.8)
        ax.contour(steps, decays, fine.T, levels=[0.34, 0.38, 0.45, 0.55],
                   colors="0.45", linewidths=0.5)
        X, y = found[name]
        ax.plot(X[:, 0], X[:, 1], "o", mfc="k", mec="white", ms=3.2,
                mew=0.5)
        ax.plot(*X[np.argmin(y)], "*", mfc="white", mec="k", ms=9,
                mew=0.8)
        ax.set_title(f"{name} search", fontsize=8)
        ax.set_xlabel("$\\log_{10}$ step size")
        ax.tick_params(width=0.5, length=2)
    axes[0].set_ylabel("$\\log_{10}$ weight decay")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(OUT / "ch13_search.pdf")
    save_cache()


if __name__ == "__main__":
    fig_simplex(); fig_tours(); fig_search()
