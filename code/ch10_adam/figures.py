"""Chapter 10: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import bowl, raw_churn
from from_scratch import adagrad, adam, momentum, rmsprop
from in_practice import OPTIMIZERS, X_test, Y_test, network, train

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_paths():
    kappa = 25
    f, grad = bowl(kappa)
    r = np.sqrt(kappa)
    runs = [("gradient descent", momentum(grad, [-9, 1.6], 1.9 / kappa, 0.0, 25)),
            ("with momentum", momentum(grad, [-9, 1.6], 4 / (r + 1) ** 2,
                                       ((r - 1) / (r + 1)) ** 2, 25))]
    u, v = np.meshgrid(np.linspace(-10, 2, 200), np.linspace(-3.3, 3.3, 200))
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 3.6), sharex=True)
    for ax, (name, p) in zip(axes, runs):
        ax.contour(u, v, 0.5 * (u ** 2 + kappa * v ** 2),
                   levels=[1, 5, 15, 30, 50, 75, 110], colors="0.7",
                   linewidths=0.5)
        ax.plot(p[:, 0], p[:, 1], "k-o", ms=2.2, lw=0.8)
        ax.plot(0, 0, "k+", ms=7)
        ax.text(-9.8, -3.0, f"{name}, 25 steps", fontsize=7,
                bbox=dict(fc="white", ec="none", pad=1.0))
        ax.set_ylim(-3.3, 3.3); ax.set_ylabel("$x_2$")
        ax.tick_params(width=0.5, length=2)
    axes[1].set_xlabel("$x_1$")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch10_paths.pdf")


def fig_churn():
    f, grad, best, L = raw_churn()
    z = np.zeros(3)
    runs = [("gradient descent", momentum(grad, z, 1 / L, 0, 100_000), "0.6", "-", 1.2),
            ("momentum 0.99", momentum(grad, z, 1 / L, 0.99, 20_000), "0.35", "--", 1.0),
            ("AdaGrad", adagrad(grad, z, 0.3, 20_000), "0.35", ":", 1.2),
            ("RMSProp", rmsprop(grad, z, 0.003, 20_000), "0.6", "-.", 1.0),
            ("Adam", adam(grad, z, 0.03, 20_000), "k", "-", 1.3)]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for name, p, color, ls, lw in runs:
        idx = np.unique(np.geomspace(1, len(p) - 1, 400).astype(int))
        gaps = np.array([f(p[i]) - best for i in idx])
        ax.plot(idx, np.maximum(gaps, 1e-12), color=color, ls=ls, lw=lw,
                label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e5); ax.set_ylim(1e-9, 1)
    ax.set_xlabel("steps"); ax.set_ylabel("gap to the optimal loss")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch10_churn.pdf")


def fig_network():
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.3))
    model = network()
    train(model, torch.optim.Adam(model.parameters(), lr=0.01))
    u, v = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = torch.tensor(np.column_stack([u.ravel(), v.ravel()]),
                        dtype=torch.float32)
    with torch.no_grad():
        z = model(grid).squeeze(1).numpy().reshape(u.shape)
    X, Y = X_test[:300].numpy(), Y_test[:300].numpy()
    a.plot(*X[Y > 0].T, "o", color="0.15", ms=2.2)
    a.plot(*X[Y < 0].T, "s", mfc="white", mec="0.15", ms=2.2, mew=0.5)
    a.contour(u, v, z, levels=[0], colors="k", linewidths=1.1)
    a.set_xlim(-3, 3); a.set_ylim(-3, 3)
    a.set_xlabel("vibration (standardized)")
    a.set_ylabel("temperature (standardized)")

    steps = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    styles = {"SGD": ("0.6", "-"), "momentum": ("0.6", "--"),
              "RMSProp": ("k", ":"), "Adam": ("k", "-")}
    for name, make in OPTIMIZERS.items():
        row = []
        for t in steps:
            m = network()
            row.append(train(m, make(m.parameters(), t))[-1])
        color, ls = styles[name]
        c.plot(steps, np.nan_to_num(row, nan=1.0), color=color, ls=ls,
               lw=1.1, marker="o", ms=2.2, label=name)
    c.set_xscale("log"); c.set_ylim(0.15, 1.0)
    c.set_xlabel("step size"); c.set_ylabel("loss after 60 passes")
    c.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2)
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch10_network.pdf")


if __name__ == "__main__":
    fig_paths(); fig_churn(); fig_network()
