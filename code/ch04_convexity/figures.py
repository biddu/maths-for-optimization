"""Chapter 4: regenerates every figure in the chapter (grayscale)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from from_scratch import STORES, TRIPS, one_depot, two_depots

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_tests():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0))
    x = np.linspace(-2.2, 2.2, 300)
    f = lambda t: 0.5 * t ** 2 + 0.3
    ax = axes[0]
    ax.plot(x, f(x), "k", lw=1.0)
    a, b = -1.6, 1.9
    ax.plot([a, b], [f(a), f(b)], color="0.4", lw=0.9, ls="--")
    ax.plot([a, b], [f(a), f(b)], "ko", ms=3)
    ax.text(-0.9, 1.75, "chord", fontsize=7)
    p = 0.8
    ax.plot(x, f(p) + p * (x - p), color="0.4", lw=0.9, ls=":")
    ax.plot(p, f(p), "ko", ms=3)
    ax.text(1.25, 0.25, "tangent", fontsize=7)
    ax.set_ylim(-0.8, 2.8); ax.set_title("convex", fontsize=8)
    g = lambda t: t ** 4 - 3 * t ** 2 + t
    x = np.linspace(-2, 2, 300)
    ax = axes[1]
    ax.plot(x, g(x), "k", lw=1.0)
    a, b = -1.3, 1.13
    ax.plot([a, b], [g(a), g(b)], color="0.4", lw=0.9, ls="--")
    ax.plot([a, b], [g(a), g(b)], "ko", ms=3)
    ax.text(-0.75, -3.6, "chord", fontsize=7)
    ax.set_ylim(-4.5, 3); ax.set_title("not convex", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch04_tests.pdf")


def draw_stores(ax):
    ax.scatter(STORES[:, 0], STORES[:, 1], s=9 * TRIPS, c="k", zorder=3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.tick_params(width=0.5, length=2)


def fig_depot():
    fig, ax = plt.subplots(figsize=(2.9, 2.8))
    g = np.linspace(0, 10, 200)
    X, Y = np.meshgrid(g, g)
    Z = sum(w * np.hypot(X - a[0], Y - a[1])
            for a, w in zip(STORES, TRIPS))
    ax.contour(X, Y, Z, levels=[82, 85, 90, 100, 115, 135],
               colors="0.5", linewidths=0.6)
    draw_stores(ax)
    best = minimize(one_depot, [5.0, 5.0]).x
    ax.plot(*best, marker="s", color="k", mfc="white", ms=6, zorder=4)
    ax.set_xlabel("km"); ax.set_ylabel("km")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch04_depot.pdf")


def fig_two_depots():
    rng = np.random.default_rng(5)
    options = {"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20_000}
    found = {}
    for _ in range(60):
        r = minimize(two_depots, rng.uniform(0, 10, 4),
                     method="Nelder-Mead", options=options)
        found.setdefault(round(r.fun, 1), r.x)
    keys = sorted(k for k in found if k < 60)
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.85))
    for ax, k in zip(axes, keys):
        z = found[k]
        d1 = np.linalg.norm(STORES - z[:2], axis=1)
        d2 = np.linalg.norm(STORES - z[2:], axis=1)
        for a, near1 in zip(STORES, d1 <= d2):
            c = z[:2] if near1 else z[2:]
            ax.plot([a[0], c[0]], [a[1], c[1]], color="0.55", lw=0.6)
        draw_stores(ax)
        for c in (z[:2], z[2:]):
            ax.plot(*c, marker="s", color="k", mfc="white", ms=5, zorder=4)
        ax.set_title(f"cost {k:.1f}", fontsize=7.5)
        ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    fig.savefig(OUT / "ch04_two_depots.pdf")


if __name__ == "__main__":
    fig_tests()
    fig_depot()
    fig_two_depots()
