"""Chapter 12: figures."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from in_practice import JOBS, schedule_cp

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BOX = dict(fc="white", ec="none", pad=1.0)


def fig_lattice():
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    corner = np.array([42 * 3 - 5 * 19, 6 * 19 - 42]) / 13
    poly = np.array([[0, 0], [7, 0], corner, [0, 19 / 3]])
    ax.fill(poly[:, 0], poly[:, 1], color="0.9")
    ax.plot(*np.vstack([poly, poly[0]]).T, color="0.3", lw=0.8)
    pts = np.array([(i, j) for i in range(9) for j in range(8)])
    ok = (6 * pts[:, 0] + 5 * pts[:, 1] <= 42) & (pts[:, 0] + 3 * pts[:, 1] <= 19)
    ax.plot(*pts[ok].T, "o", color="0.25", ms=2.6)
    ax.plot(*pts[~ok].T, "o", mfc="white", mec="0.6", ms=2.6, mew=0.5)
    u = np.linspace(-1, 9, 2)
    for level, style in ((65.77, "--"), (63, ":")):
        ax.plot(u, (level - 9 * u) / 8, color="0.4", lw=0.7, ls=style)
    ax.plot(*corner, "k*", ms=8)
    ax.plot(7, 0, "ks", ms=5)
    ax.plot(2, 5, "kD", ms=4, mfc="white")
    ax.annotate("best fractional plan\n(2.38, 5.54), profit 65.8",
                corner, xytext=(3.3, 6.5), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.annotate("best whole plan\n(7, 0), profit 63", (7, 0),
                xytext=(5.5, 2.4), fontsize=7, bbox=BOX,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.annotate("rounded down\n(2, 5), profit 58", (2, 5),
                xytext=(0.25, 2.6), fontsize=7, bbox=BOX,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.3, 7.5)
    ax.set_aspect("equal")
    ax.set_xlabel("lathes $x_1$"); ax.set_ylabel("presses $x_2$")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch12_lattice.pdf")


def fig_schedule():
    value, _, _, plan, _ = schedule_cp(JOBS)
    names = ["saw", "bench", "paint"]
    shades = ["0.25", "0.5", "0.72", "0.9"]
    fig, ax = plt.subplots(figsize=(4.4, 1.5))
    for (j, k), start in plan.items():
        machine, hours = JOBS[j][k]
        ax.barh(2 - machine, hours, left=start, height=0.62,
                color=shades[j], edgecolor="k", linewidth=0.5)
        ax.text(start + hours / 2, 2 - machine, f"job {j + 1}",
                ha="center", va="center", fontsize=6.5,
                color="white" if j < 2 else "black")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(names)
    ax.set_xlim(0, value); ax.set_xticks(range(0, int(value) + 1))
    ax.set_xlabel("hours")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch12_schedule.pdf")


def fig_gap():
    data = json.loads((HERE / "gap_history.json").read_text())
    limit, value, bound, _ = zip(*data["milp"])
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.plot(limit, value, "ko-", ms=3, lw=1.0, label="best schedule found")
    ax.plot(limit, bound, "ks--", ms=3, lw=0.9, mfc="white",
            label="proved lower bound")
    ax.axhline(data["optimum"], color="0.5", lw=0.6, ls=":")
    ax.text(0.55, data["optimum"] + 1.5, "optimum 57", fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 40, 80])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "40", "80"])
    ax.minorticks_off()
    ax.set_xlabel("time limit (seconds)")
    ax.set_ylabel("hours to finish all jobs")
    ax.set_ylim(40, 100)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "ch12_gap.pdf")


if __name__ == "__main__":
    fig_lattice(); fig_schedule(); fig_gap()
