"""Chapter 11: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import klee_minty, workshop
from from_scratch import simplex

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BOX = dict(fc="white", ec="none", pad=1.0)


def fig_workshop():
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    poly = np.array([[0, 0], [30, 0], [20, 20], [10, 30], [0, 30]])
    ax.fill(poly[:, 0], poly[:, 1], color="0.88")
    ax.plot(*np.vstack([poly, poly[0]]).T, color="0.3", lw=0.8)
    u = np.linspace(0, 45, 2)
    for level in (600, 1000, 1400):
        ax.plot(u, (level - 40 * u) / 30, color="0.5", lw=0.6, ls="--")
        ax.text(level / 40 + 0.6, 1.0, str(level), fontsize=6.5,
                color="0.3", bbox=BOX if level < 1400 else None)
    path = np.array([[0, 0], [30, 0], [20, 20]])
    for p, q in zip(path[:-1], path[1:]):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                              color="k"))
    ax.plot(poly[:, 0], poly[:, 1], "o", color="k", ms=3.5)
    labels = [("(0, 0)", (0.8, 3.0)), ("(30, 0)", (30.6, 3.0)),
              ("(20, 20)", (21, 20.5)), ("(10, 30)", (10.5, 31)),
              ("(0, 30)", (0.8, 31))]
    for text, pos in labels:
        ax.text(*pos, text, fontsize=7)
    ax.text(19.6, 5.5, "assembly", fontsize=7, rotation=-63, color="0.3")
    ax.text(3, 27.3, "kits", fontsize=7, color="0.3")
    ax.text(9.5, 21.5, "sawing", fontsize=7, rotation=-45, color="0.3")
    ax.set_xlim(0, 42); ax.set_ylim(0, 35)
    ax.set_aspect("equal")
    ax.set_xlabel("desks per week $x_1$")
    ax.set_ylabel("bookcases per week $x_2$")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch11_workshop.pdf")


def fig_sensitivity():
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.2))
    saw = np.linspace(0, 60, 121)
    a.plot(saw, [workshop(saw=s)[1] for s in saw], "k", lw=1.2)
    for s in (30, 45):
        a.axvline(s, color="0.6", lw=0.5, ls=":")
    a.plot(40, 1400, "ko", ms=3.5)
    a.text(8, 880, "40 a unit", fontsize=7, rotation=62)
    a.text(31.5, 1150, "20", fontsize=7)
    a.text(50, 1390, "0", fontsize=7)
    a.set_xlabel("sawing hours available")
    a.set_ylabel("best weekly profit")
    a.set_xlim(0, 60); a.set_ylim(0, 1700)

    sizes = range(2, 11)
    piv = []
    for n in sizes:
        cc, M, bb = klee_minty(n)
        piv.append(len(simplex(cc, M, bb)[3]) - 1)
    rng = np.random.default_rng(0)
    rand_n = (5, 10, 20, 50, 100, 200)
    rand = [np.mean([len(simplex(rng.uniform(0, 1, n),
                                 rng.uniform(0, 1, (n, n)),
                                 rng.uniform(1, 2, n) * n / 2)[3]) - 1
                     for _ in range(10)]) for n in rand_n]
    c.loglog(list(sizes), piv, "ks--", ms=3, lw=0.8, label="twisted cube")
    c.loglog(rand_n, rand, "ko-", ms=3, lw=0.9, label="random")
    c.set_xlabel("number of variables $n$")
    c.set_ylabel("pivots")
    c.legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2, which="both")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "ch11_sensitivity.pdf")


if __name__ == "__main__":
    fig_workshop(); fig_sensitivity()
