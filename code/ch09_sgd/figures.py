"""Chapter 9: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import A, DECAY, L, d, gap, n, y
from from_scratch import gradient, running_mean_sgd, sgd

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_mean():
    calls = np.random.default_rng(2).normal(6, 2, 2000)
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    k = np.arange(1, 2001)
    ax.plot(k, running_mean_sgd(calls, 0.1), color="0.6", lw=0.6,
            label="step 0.1")
    ax.plot(k, running_mean_sgd(calls, 0.01), "k--", lw=1.0,
            label="step 0.01")
    ax.plot(k, running_mean_sgd(calls, lambda j: 1 / (j + 1)), "k",
            lw=1.2, label="step $1/k$")
    ax.axhline(6, color="0.3", lw=0.5, ls=":")
    ax.set_xlim(0, 2000); ax.set_ylim(0, 8)
    ax.set_xlabel("calls seen"); ax.set_ylabel("estimate (minutes)")
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=3)
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch09_mean.pdf")


def fig_churn():
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x, gd = np.zeros(d), []
    for _ in range(100):
        x -= gradient(x, A, y) / L
        gd.append(gap(x))
    ax.plot(np.arange(1, 101), gd, "k-o", ms=2.5, lw=0.8,
            markevery=[0, 1, 4, 9, 19, 49, 99],
            label="gradient descent")
    passes = np.arange(1, 501) / 100
    styles = [(0.1, "0.65", "-"), (0.02, "0.4", "-")]
    for step, color, ls in styles:
        _, snaps = sgd(A, y, np.zeros(d), step, epochs=5, every=1000)
        ax.plot(passes, [gap(s) for s in snaps], color=color, lw=0.6,
                ls=ls, label=f"SGD, step {step}")
    _, snaps = sgd(A, y, np.zeros(d), DECAY, epochs=5, every=1000)
    ax.plot(passes, [gap(s) for s in snaps], "k", lw=1.2,
            label="SGD, decreasing step")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, 100); ax.set_ylim(3e-5, 1)
    ax.set_xlabel("passes through the data")
    ax.set_ylabel("gap to the optimal loss")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch09_churn.pdf")


def fig_batch():
    rng = np.random.default_rng(1)
    full = gradient(np.zeros(d), A, y)
    sizes = [1, 4, 16, 64, 256, 1024]
    var = [np.mean([np.sum((gradient(np.zeros(d), A[r], y[r]) - full) ** 2)
                    for r in (rng.choice(n, b, replace=False)
                              for _ in range(2000))]) for b in sizes]
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.1))
    a.loglog(sizes, var, "ko-", ms=3, lw=0.9)
    a.set_xlabel("batch size $b$")
    a.set_ylabel("variance of the gradient")
    batches = [1, 8, 64, 512, 4096]
    # Measured by experiments.py (examples to reach a gap of 0.003).
    seen = [14000, 14000, 13888, 30720, 221184]
    c.loglog(batches, seen, "ko-", ms=3, lw=0.9)
    c.set_ylim(5e3, 5e5)
    c.set_xlabel("batch size $b$")
    c.set_ylabel("examples to reach 0.003")
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch09_batch.pdf")


if __name__ == "__main__":
    fig_mean(); fig_churn(); fig_batch()
