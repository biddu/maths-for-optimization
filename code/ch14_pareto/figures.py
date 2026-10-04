"""Chapter 14: figures."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import CARS, reliability, warehouse_options, workshop
from from_scratch import G, link_design, pareto, waterfill

OUT = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BOX = dict(fc="white", ec="none", pad=1.0)


def fig_cars():
    keep = pareto(CARS)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    front = CARS[keep][np.argsort(CARS[keep][:, 1])]
    ax.step(front[:, 1], front[:, 0], where="post", color="0.5", lw=0.8,
            ls="--")
    for i, (price, age) in enumerate(CARS):
        good = i in keep
        ax.plot(age, price, "o", ms=5, mfc="k" if good else "white",
                mec="k", mew=0.7)
        ax.text(age + 0.25, price + 0.25, str(i + 1), fontsize=7)
    ax.set_xlabel("age (years)"); ax.set_ylabel("price ($1000)")
    ax.set_xlim(0, 12.5); ax.set_ylim(5, 21)
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch14_cars.pdf")


def fig_link():
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.3), sharey=True)
    P = np.linspace(0, 20, 400)
    R = np.array([waterfill(G, max(p, 1e-9))[1] for p in P])
    a.fill_between(P, R, 0, color="0.9")
    a.plot(P, R, "k", lw=1.2)
    a.plot(10, 6.517, "ko", ms=3.5)
    t = np.array([5.5, 14.5])
    a.plot(t, 6.517 + 0.330 * (t - 10), color="0.4", lw=0.7, ls="--")
    a.text(10.6, 5.7, "slope 0.33", fontsize=7)
    a.text(9.5, 2.0, "achievable", fontsize=7, color="0.3")
    a.text(1.5, 8.6, "not achievable", fontsize=7, color="0.3")
    a.set_title("no overhead", fontsize=8)
    budgets = np.linspace(0.51, 20, 400)
    R2 = np.array([link_design(G, b)[0] for b in budgets])
    c.fill_between(budgets, R2, 0, color="0.9")
    c.plot(budgets, R2, "k", lw=1.2)
    for b in (2.915, 7.618, 17.744):
        c.plot(b, link_design(G, b)[0], "o", mfc="white", mec="k", ms=3.5,
               mew=0.7)
    k = np.argmax(R2 / budgets)
    c.plot([0, 2.6 * budgets[k]], [0, 2.6 * R2[k]], color="0.4", lw=0.7,
           ls="--")
    c.plot(budgets[k], R2[k], "ks", ms=3.5)
    c.text(4.3, 3.2, "most rate\nper watt", fontsize=7, va="top")
    c.set_title("0.5 W per active channel", fontsize=8)
    for ax in (a, c):
        ax.set_xlim(0, 20); ax.set_ylim(0, 10)
        ax.set_xlabel("total power (W)")
        ax.tick_params(width=0.5, length=2)
    a.set_ylabel("rate (Mbit/s)")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(OUT / "ch14_link.pdf")


def fig_choices():
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.3))
    options = warehouse_options()
    pts = np.array([[spare, cost] for _, cost, spare in options])
    keep = pareto(np.column_stack([pts[:, 1], -pts[:, 0]]))
    hull = [k for k in keep if options[k][0] in
            {(3, 4), (1, 3, 4), (1, 2, 3, 4)}]
    order = sorted(hull, key=lambda k: pts[k, 0])
    a.plot(pts[order, 0], pts[order, 1], color="0.5", lw=0.8, ls="--")
    for k, (name, cost, spare) in enumerate(options):
        style = dict(mfc="white", mec="0.5") if k not in keep else (
            dict(mfc="k", mec="k") if k in hull else
            dict(mfc="0.6", mec="k"))
        a.plot(spare, cost, "o", ms=4.5, mew=0.7, **style)
    a.set_xlabel("spare capacity (pallets)")
    a.set_ylabel("weekly cost (euro)")
    a.set_xlim(0, 580); a.set_ylim(1900, 3150)

    margins = np.linspace(0, 0.2, 21)
    profit = [workshop(m)[1] for m in margins]
    rel = [100 * reliability(workshop(m)[0], trials=100_000)
           for m in margins]
    c.plot(rel, profit, "k", lw=1.2)
    for m in (0.0, 0.05, 0.10, 0.15):
        plan, value = workshop(m)
        r = 100 * reliability(plan, trials=100_000)
        c.plot(r, value, "ko", ms=3)
        c.text(r - 3, value - 12, f"{int(100 * m)}%", fontsize=7,
               ha="right", va="top")
    c.set_xlabel("chance the plan fits (%)")
    c.set_ylabel("weekly profit (dollars)")
    c.set_xlim(0, 103); c.set_ylim(1150, 1420)
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch14_choices.pdf")


def fig_library():
    from in_practice import evolve, portfolio
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.5, 2.3))
    caps = np.linspace(0.22, 0.6, 39)
    for target, ls in ((3, "-"), (4, "--"), (5, "-."), (6, ":")):
        vols = [(portfolio(target, cap) or [np.nan])[0] for cap in caps]
        a.plot(100 * caps, vols, "k", ls=ls, lw=1.1,
               label=f"return {target}%")
    a.set_xlabel("largest holding allowed (%)")
    a.set_ylabel("volatility (percent a year)")
    a.set_xlim(20, 60); a.set_ylim(0, 11.5)
    a.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2,
             handlelength=1.8, columnspacing=0.8, borderpad=0.1)

    budgets = np.linspace(0.51, 20, 300)
    c.plot(budgets, [link_design(G, b)[0] for b in budgets], color="0.6",
           lw=2.2, label="exact front")
    power, rate, _ = evolve(20)
    c.plot(power, rate, "o", mfc="white", mec="0.35", ms=2.6, mew=0.5,
           label="20 generations")
    power, rate, _ = evolve(200)
    c.plot(power, rate, "k.", ms=3, label="200 generations")
    c.set_xlabel("total power (W)"); c.set_ylabel("rate (Mbit/s)")
    c.set_xlim(0, 20); c.set_ylim(0, 9.5)
    c.legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax in (a, c):
        ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / "ch14_library.pdf")


if __name__ == "__main__":
    fig_cars(); fig_link(); fig_choices(); fig_library()
