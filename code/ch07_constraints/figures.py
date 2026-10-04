"""Chapter 7: regenerates every figure in the chapter (grayscale)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from from_scratch import waterfill

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_can():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    r = np.linspace(2.0, 6.0, 300)
    h = np.linspace(2.0, 14.0, 300)
    R, H = np.meshgrid(r, h)
    area = 2 * np.pi * R ** 2 + 2 * np.pi * R * H
    cs = ax.contour(R, H, area, levels=[150, 200, 264.36, 350, 450],
                    colors="0.55", linewidths=0.6, linestyles="dashed")
    ax.clabel(cs, fmt="%.0f", fontsize=6,
              manual=[(2.6, 6.6), (3.0, 7.6), (4.6, 4.6), (4.3, 8.7), (4.9, 9.7)])
    ax.plot(r, 330 / (np.pi * r ** 2), "k", lw=1.2)
    rs = (330 / (2 * np.pi)) ** (1 / 3)
    ax.plot(rs, 2 * rs, "ko", ms=4)
    box = dict(fc="white", ec="none", pad=1.0)
    ax.annotate("volume = 330", (2.95, 12.6), fontsize=7, bbox=box)
    ax.annotate("optimum", (rs, 2 * rs), xytext=(3.95, 7.0), fontsize=7, bbox=box)
    ax.set_xlim(2, 6); ax.set_ylim(2, 14)
    ax.set_xlabel("radius $r$ (cm)"); ax.set_ylabel("height $\\ell$ (cm)")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch07_can.pdf")


def fig_active():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.5))
    g = np.linspace(-1.6, 2.2, 300)
    X, Y = np.meshgrid(g, g)
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, c, title in ((axes[0], (0.4, 0.3), "inactive: $\\lambda = 0$"),
                         (axes[1], (1.5, 1.0), "active: $\\lambda > 0$")):
        ax.fill(np.cos(th), np.sin(th), color="0.87")
        ax.plot(np.cos(th), np.sin(th), color="0.2", lw=0.8)
        Z = (X - c[0]) ** 2 + (Y - c[1]) ** 2
        ax.contour(X, Y, Z, levels=[0.1, 0.4, 0.9, 1.6, 2.5, 3.6],
                   colors="0.5", linewidths=0.5, linestyles="dashed")
        ax.plot(*c, marker="s", color="k", mfc="white", ms=4.5)
        norm = np.hypot(*c)
        best = c if norm <= 1 else (c[0] / norm, c[1] / norm)
        ax.plot(*best, "ko", ms=4)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(-1.6, 2.2); ax.set_ylim(-1.6, 2.2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "ch07_active.pdf")


def fig_waterfill():
    g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
    x, level = waterfill(g, 10.0)
    floors = 1 / g
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    k = np.arange(1, 6)
    ax.bar(k, floors, width=0.9, color="0.35", label="floor $1/g_k$")
    ax.bar(k, x, width=0.9, bottom=floors, color="0.82", edgecolor="0.4",
           linewidth=0.4, label="power $x_k$")
    ax.axhline(level, color="k", lw=0.9, ls="--")
    ax.text(0.45, level + 0.2, "water level 4.375", fontsize=7)
    ax.set_xlabel("channel"); ax.set_ylabel("height")
    ax.set_ylim(0, 11); ax.set_xlim(0.4, 5.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 0.88))
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch07_waterfill.pdf")


if __name__ == "__main__":
    fig_can()
    fig_active()
    fig_waterfill()
