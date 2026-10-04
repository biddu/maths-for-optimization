"""Chapter 6: regenerates every figure in the chapter (grayscale)."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from from_scratch import bfgs, conjugate_gradient, line_search, newton
from problems import churn_matrix, logistic, smoother, temperatures

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_model():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.linspace(-0.6, 1.8, 300)
    f = lambda t: np.exp(t) - 2 * t
    ax.plot(x, f(x), "k", lw=1.1, label="$f(x) = e^x - 2x$")
    x0 = 0.0
    model = f(x0) + (np.exp(x0) - 2) * (x - x0) + 0.5 * np.exp(x0) * (x - x0) ** 2
    ax.plot(x, model, color="0.45", lw=0.9, ls="--",
            label="quadratic model at $x_0 = 0$")
    ax.plot([x0, 1.0], [f(x0), f(x0) - 0.5], "ko", ms=3.5)
    ax.plot(np.log(2), f(np.log(2)), marker="s", color="k", mfc="white", ms=4.5)
    ax.annotate("$x_0$", (0, f(0)), xytext=(-0.22, 1.08), fontsize=8)
    ax.annotate("$x_1$", (1.0, 0.5), xytext=(1.03, 0.36), fontsize=8)
    ax.set_ylim(0.2, 2.2); ax.set_xlabel("$x$")
    ax.legend(frameon=False, fontsize=7, loc="upper center")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch06_model.pdf")


def gd_norms(f, grad, x, steps):
    norms, t = [], 1.0
    for _ in range(steps):
        g = grad(x)
        norms.append(np.linalg.norm(g))
        t = 2 * t
        while f(x - t * g) > f(x) - 0.3 * t * (g @ g):
            t *= 0.5
        x = x - t * g
    return norms


def fig_convergence():
    f, grad, hess = logistic(*churn_matrix())
    start = np.zeros(3)
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.semilogy(gd_norms(f, grad, start, 31), color="0.55", lw=1.0,
                label="gradient descent")
    ax.semilogy(bfgs(f, grad, start)[1], "k--", lw=1.0, marker=".", ms=3,
                label="BFGS")
    ax.semilogy(newton(f, grad, hess, start)[1], "k-", lw=1.0, marker="o",
                ms=3, label="Newton")
    ax.set_xlim(0, 30); ax.set_ylim(1e-10, 30)
    ax.set_xlabel("step $k$"); ax.set_ylabel("length of the gradient")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch06_convergence.pdf")


def fig_smoothing():
    noisy, clean = temperatures()
    matvec = smoother(100.0)
    x, res = conjugate_gradient(matvec, noisy)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.1))
    ax = axes[0]
    days = np.arange(730)
    ax.plot(days, noisy[:730], ".", color="0.6", ms=1.5)
    ax.plot(days, x[:730], "k", lw=1.0)
    ax.set_xlabel("day"); ax.set_ylabel("temperature")
    ax = axes[1]
    L = 401.0
    z, gd = np.zeros(noisy.size), []
    for _ in range(400):
        r = noisy - matvec(z)
        gd.append(np.linalg.norm(r))
        z += r / L
    ax.semilogy(res / res[0], "k", lw=1.0, label="conjugate gradient")
    ax.semilogy(np.array(gd) / gd[0], color="0.55", lw=1.0,
                label="gradient descent")
    ax.set_xlim(0, 400); ax.set_ylim(1e-9, 2)
    ax.set_xlabel("step $k$"); ax.set_ylabel("relative residual")
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes:
        ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch06_smoothing.pdf")


if __name__ == "__main__":
    fig_model()
    fig_convergence()
    fig_smoothing()
