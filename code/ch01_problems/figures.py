"""Chapter 1: regenerates every figure in the chapter (grayscale, print size)."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_regions():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.35))
    g = np.linspace(0, 10, 400)
    X, Y = np.meshgrid(g, g)

    ax = axes[0]
    poly = np.array([[0, 0], [6, 0], [6, 4], [2, 8], [0, 8]])
    ax.fill(poly[:, 0], poly[:, 1], color="0.85", zorder=1)
    ax.plot(*np.vstack([poly, poly[0]]).T, color="0.2", lw=0.8, zorder=2)
    cs = ax.contour(X, Y, 3 * X + 2 * Y, levels=[8, 14, 20, 26],
                    colors="0.35", linewidths=0.6, linestyles="dashed")
    ax.clabel(cs, fmt="%d", fontsize=6)
    ax.plot(6, 4, "ko", ms=4, zorder=3)
    ax.annotate("(6, 4)", (6, 4), xytext=(6.5, 4.5), fontsize=7)
    ax.set_title("Version A: linear rate", fontsize=8)

    ax = axes[1]
    tri = np.array([[0, 0], [10, 0], [0, 10]])
    ax.fill(tri[:, 0], tri[:, 1], color="0.85", zorder=1)
    ax.plot(*np.vstack([tri, tri[0]]).T, color="0.2", lw=0.8, zorder=2)
    Z = np.log2(1 + X) + np.log2(1 + Y / 4)
    cs = ax.contour(X, Y, Z, levels=[2.0, 3.0, 3.8138, 4.5],
                    colors="0.35", linewidths=0.6, linestyles="dashed")
    ax.clabel(cs, fmt="%.1f", fontsize=6)
    ax.plot(6.5, 3.5, "ko", ms=4, zorder=3)
    ax.annotate("(6.5, 3.5)", (6.5, 3.5), xytext=(6.9, 4.1), fontsize=7)
    ax.set_title("Version B: Shannon rate", fontsize=8)

    for ax in axes:
        ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$ (watts)"); ax.set_ylabel("$x_2$ (watts)")
        ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch01_regions.pdf")


def fig_local_global():
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    x = np.linspace(-2, 2, 400)
    f = lambda t: t ** 4 - 3 * t ** 2 + t
    ax.plot(x, f(x), color="0.1", lw=1.0)
    for t, label, dx, dy in [(-1.3008, "global minimum", -0.55, -0.95),
                             (1.1309, "local minimum", -0.55, -1.0)]:
        ax.plot(t, f(t), "ko", ms=4)
        ax.annotate(label, (t, f(t)), xytext=(t + dx, f(t) + dy), fontsize=7)
    ax.set_ylim(-5.2, 3.5)
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch01_local_global.pdf")


if __name__ == "__main__":
    fig_regions()
    fig_local_global()
