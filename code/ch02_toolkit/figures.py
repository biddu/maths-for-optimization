"""Chapter 2: regenerates every figure in the chapter (grayscale)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from from_scratch import rate, grad_rate

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_gradient():
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    g = np.linspace(0, 10.5, 400)
    X, Y = np.meshgrid(g, g)
    Z = np.log2(1 + X) + np.log2(1 + Y / 4)
    levels = sorted(rate(np.array(p)) for p in
                    [(2, 8), (4, 6), (6.5, 3.5), (1, 2), (9, 8)])
    ax.contour(X, Y, Z, levels=levels, colors="0.45", linewidths=0.6,
               linestyles="dashed")
    ax.plot([0, 10], [10, 0], color="0.2", lw=0.9)
    for p in [(2.0, 8.0), (4.0, 6.0), (6.5, 3.5)]:
        gr = grad_rate(np.array(p))
        ax.annotate("", xy=(p[0] + 6 * gr[0], p[1] + 6 * gr[1]), xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
        ax.plot(*p, "ko", ms=3)
    box = dict(facecolor="white", edgecolor="none", pad=1.0)
    ax.text(0.5, 6.9, "(2, 8)", fontsize=7, bbox=box)
    ax.text(2.4, 4.9, "(4, 6)", fontsize=7, bbox=box)
    ax.text(4.1, 2.5, "(6.5, 3.5)", fontsize=7, bbox=box)
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5); ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ (watts)"); ax.set_ylabel("$x_2$ (watts)")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch02_gradient.pdf")


def fig_quadratics():
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.75))
    g = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(g, g)
    cases = [("positive definite", [[2, 1], [1, 2]]),
             ("indefinite", [[1, 2], [2, 1]]),
             ("positive semidefinite", [[1, 1], [1, 1]])]
    for ax, (title, M) in zip(axes, cases):
        Z = M[0][0] * X ** 2 + 2 * M[0][1] * X * Y + M[1][1] * Y ** 2
        ax.contour(X, Y, Z, levels=[0.5, 2, 4.5, 8], colors="0.15",
                   linewidths=0.7)
        ax.contour(X, Y, Z, levels=[-4.5, -2, -0.5], colors="0.15",
                   linewidths=0.7, linestyles="dashed")
        ax.set_title(title, fontsize=7.5)
        ax.set_aspect("equal"); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        ax.tick_params(width=0.5, length=2, labelsize=7)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(OUT / "ch02_quadratics.pdf")


def fig_fd_error():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.array([4.0, 6.0])
    exact = grad_rate(x)[0]
    hs = 10.0 ** -np.arange(1, 15.01, 0.25)
    e = np.array([1.0, 0.0])
    fwd = [abs((rate(x + h * e) - rate(x)) / h - exact) for h in hs]
    cen = [abs((rate(x + h * e) - rate(x - h * e)) / (2 * h) - exact)
           for h in hs]
    ax.loglog(hs, fwd, color="0.0", lw=0.9, label="forward")
    ax.loglog(hs, cen, color="0.5", lw=0.9, ls="--", label="central")
    ax.set_xlabel("step $h$"); ax.set_ylabel("error in the derivative")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch02_fd_error.pdf")


if __name__ == "__main__":
    fig_gradient()
    fig_quadratics()
    fig_fd_error()
