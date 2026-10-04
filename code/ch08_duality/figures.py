"""Chapter 8: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications import portfolio, svm
from data import ASSETS, RETURN, VOL, cars, parts, standardize
from from_scratch import dual_function, lasso

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BOX = dict(fc="white", ec="none", pad=1.0)


def fig_dual():
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.5, 2.1))
    g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
    lam = np.linspace(0.12, 0.8, 300)
    a.plot(lam, [-dual_function(l, g, 10.0) for l in lam], "k", lw=1.1)
    a.axhline(6.5171, color="0.4", lw=0.7, ls="--")
    a.plot(0.3298, 6.5171, "ko", ms=3.5)
    a.text(0.47, 6.2, "best rate 6.517", fontsize=7)
    a.set_ylim(5.8, 9.2)
    a.set_xlabel("price of power $\\lambda$")
    a.set_ylabel("bound on the rate")
    a.set_title("five channels", fontsize=8)

    cost = np.array([6.0, 5.0, 5.0, 3.0])
    gain = np.array([30.0, 24.0, 23.0, 9.0])
    lam = np.linspace(3.0, 7.0, 500)
    vals = [10 * l + np.maximum(gain - l * cost, 0).sum() for l in lam]
    b.plot(lam, vals, "k", lw=1.1)
    b.axhline(47, color="0.4", lw=0.7, ls="--")
    b.plot(4.8, 49.2, "ko", ms=3.5)
    b.annotate("", (5.9, 47), (5.9, 49.2),
               arrowprops=dict(arrowstyle="<->", lw=0.6))
    b.text(6.0, 47.75, "gap 2.2", fontsize=7)
    b.text(3.1, 46.1, "best plan 47", fontsize=7)
    b.set_ylim(45, 54); b.set_xlim(3, 7)
    b.set_xlabel("price of power $\\lambda$")
    b.set_title("four on-off channels", fontsize=8)
    for ax in (a, b):
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch08_dual.pdf")


def fig_frontier():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    targets = np.linspace(2.05, 9.5, 60)
    risks = [portfolio(t)[1] for t in targets]
    ax.plot(risks, targets, "k", lw=1.2)
    ax.plot(VOL, RETURN, "o", color="0.45", ms=3.5)
    offsets = [(0.8, -0.1), (0.8, -0.45), (0.8, -0.45),
               (0.8, -0.45), (-7.6, -0.75)]
    for name, v, r, (dx, dy) in zip(ASSETS, VOL, RETURN, offsets):
        ax.text(v + dx, r + dy, name, fontsize=7)
    w, risk, _ = portfolio(6.0)
    ax.plot(risk, 6.0, "ks", ms=4)
    ax.text(risk - 0.6, 6.25, "6% target", fontsize=7, ha="right")
    ax.set_xlim(0, 26); ax.set_ylim(1, 10.5)
    ax.set_xlabel("volatility (percent a year)")
    ax.set_ylabel("expected return (percent)")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch08_frontier.pdf")


def fig_lasso():
    A_raw, price = cars()
    A, b = standardize(A_raw), price - price.mean()
    T_raw, t_price = cars(n=1000, seed=4)
    T, tb = standardize(T_raw, A_raw), t_price - price.mean()
    lams = np.geomspace(0.01, 8, 50)
    X = np.array([lasso(A, b, l, steps=5000) for l in lams])
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.1))
    a.plot(lams, X[:, 3:], color="0.7", lw=0.5)
    styles = ["-", "--", ":"]
    for k, name in enumerate(["age", "distance", "power"]):
        a.plot(lams, X[:, k], "k", lw=1.1, ls=styles[k], label=name)
    a.set_xscale("log")
    a.set_xlabel("weight $\\lambda$"); a.set_ylabel("coefficient")
    a.legend(fontsize=6.5, frameon=False, loc=(0.04, 0.44))
    train = [np.sqrt(np.mean((A @ x - b) ** 2)) for x in X]
    test = [np.sqrt(np.mean((T @ x - tb) ** 2)) for x in X]
    c.plot(lams, test, "k", lw=1.1, label="1000 new cars")
    c.plot(lams, train, "k", lw=0.8, ls="--", label="the 60 cars")
    c.set_xscale("log"); c.set_ylim(0, 6)
    c.set_xlabel("weight $\\lambda$")
    c.set_ylabel("typical error ($1000)")
    c.legend(fontsize=6.5, frameon=False, loc="upper left")
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch08_lasso.pdf")


def fig_svm():
    X, y = parts()
    w, b, alpha, _ = svm(X, y)
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.plot(*X[y > 0].T, "o", color="0.15", ms=3.5, label="sound")
    ax.plot(*X[y < 0].T, "s", mfc="white", mec="0.15", ms=3.5,
            mew=0.7, label="faulty")
    sv = alpha > 1e-5
    ax.plot(*X[sv].T, "o", mfc="none", mec="k", ms=8, mew=0.7)
    u = np.array([0.5, 8.5])
    for level, style in [(0, "-"), (1, "--"), (-1, "--")]:
        ax.plot(u, (level - b - w[0] * u) / w[1], "k", ls=style,
                lw=1.1 if level == 0 else 0.6)
    ax.set_xlim(0.5, 8.5); ax.set_ylim(1.5, 8.5)
    ax.set_xlabel("vibration reading")
    ax.set_ylabel("temperature reading")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch08_svm.pdf")


if __name__ == "__main__":
    fig_dual(); fig_frontier(); fig_lasso(); fig_svm()
