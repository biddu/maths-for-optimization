"""Chapter 5: regenerates every figure in the chapter (grayscale)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from from_scratch import (churn_data, gradient_descent, logistic_grad,
                          logistic_loss, standardize)

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def path(step, n):
    x = np.array([10.0, 1.0])
    pts = [x]
    for _ in range(n):
        x = x - step * np.array([x[0], 10 * x[1]])
        pts.append(x)
    return np.array(pts)


def fig_paths():
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 2.5), sharex=True)
    g1, g2 = np.linspace(-1, 10.5, 300), np.linspace(-1.3, 1.3, 200)
    X, Y = np.meshgrid(g1, g2)
    Z = 0.5 * (X ** 2 + 10 * Y ** 2)
    for ax, step, label in ((axes[0], 0.1, "step $1/L = 0.1$"),
                            (axes[1], 0.19, "step $0.19$")):
        ax.contour(X, Y, Z, levels=[1, 5, 12, 22, 35, 50], colors="0.6",
                   linewidths=0.5)
        p = path(step, 25)
        ax.plot(p[:, 0], p[:, 1], "k-", lw=0.8, marker=".", ms=3)
        ax.set_ylabel("$x_2$"); ax.set_yticks([-1, 0, 1])
        ax.text(0.2, 0.92, label, fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.tick_params(width=0.5, length=2)
    axes[1].set_xlabel("$x_1$")
    fig.tight_layout(pad=0.4, h_pad=0.6)
    fig.savefig(OUT / "ch05_paths.pdf")


def fig_rates():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for kappa, style in ((10, "-"), (100, "--"), (1000, ":")):
        f = lambda x: 0.5 * (x[0] ** 2 + kappa * x[1] ** 2)
        grad = lambda x: np.array([x[0], kappa * x[1]])
        _, h = gradient_descent(f, grad, [1.0, 1.0], 1 / kappa, tol=0,
                                max_iter=600)
        ax.semilogy(h / h[0], "k" + style, lw=0.9,
                    label=f"$\\kappa = {kappa}$")
    ax.set_ylim(1e-8, 2); ax.set_xlabel("step $k$")
    ax.set_ylabel("$f(x_k) / f(x_0)$")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch05_rates.pdf")


def fig_churn():
    hours, months, y = churn_data()
    ones = np.ones(y.size)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.15))
    ax = axes[0]
    stay = y < 0
    ax.plot(hours[stay], months[stay], "o", ms=2.5, mfc="white",
            mec="0.3", mew=0.5, label="stayed")
    ax.plot(hours[~stay], months[~stay], "ko", ms=2.5, label="cancelled")
    b = np.array([2.1952, -0.2857, -0.0496])
    hh = np.array([0, 20])
    ax.plot(hh, -(b[0] + b[1] * hh) / b[2], "k-", lw=0.9)
    ax.set_xlim(0, 20); ax.set_ylim(0, 48)
    ax.set_xlabel("hours watched per week"); ax.set_ylabel("months subscribed")
    ax.legend(frameon=True, fontsize=6.5, loc="upper right")
    ax = axes[1]
    for A, style, label in (
            (np.column_stack([ones, hours, months]), "--", "raw"),
            (np.column_stack([ones, standardize(hours),
                              standardize(months)]), "-", "standardized")):
        f = lambda x: logistic_loss(x, A, y)
        grad = lambda x: logistic_grad(x, A, y)
        L = np.linalg.eigvalsh(A.T @ A)[-1] / (4 * y.size)
        _, h = gradient_descent(f, grad, np.zeros(3), 1 / L, tol=0,
                                max_iter=100_000)
        k = np.arange(1, h.size)
        ax.loglog(k, h[1:] - 0.3411415224574, "k" + style, lw=0.9,
                  label=label)
    ax.set_ylim(1e-10, 1); ax.set_xlabel("step $k$")
    ax.set_ylabel("loss minus optimal loss")
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes:
        ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch05_churn.pdf")


if __name__ == "__main__":
    fig_paths()
    fig_rates()
    fig_churn()
