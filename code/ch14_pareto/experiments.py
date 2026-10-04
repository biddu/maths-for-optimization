"""Chapter 14: every number quoted in the chapter."""
import importlib.util
import itertools
import pathlib

import numpy as np
from scipy.optimize import linprog

from from_scratch import (G, LN2, at_price, link_design, pareto, waterfill,
                          weighted_sum, with_limit)

HERE = pathlib.Path(__file__).resolve().parent


def load(folder, name):
    spec = importlib.util.spec_from_file_location(
        name, HERE.parent / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Eight used cars: price ($1000) and age (years).
CARS = np.array([[19.5, 1], [16.0, 2], [17.0, 3], [12.5, 4],
                 [13.5, 6], [9.0, 7], [10.0, 9], [6.5, 11]])


def warehouse_options():
    """Every set of warehouses that can serve the cities of
    Chapter 12: weekly cost and spare capacity."""
    m = load("ch12_integer", "models")
    out = []
    for r in range(1, 5):
        for s in itertools.combinations(range(4), r):
            s = list(s)
            if m.CAPACITY[s].sum() < m.DEMAND.sum():
                continue
            n = len(s)
            A = np.zeros((n + 4, 4 * n))
            for i in range(n):
                A[i, 4 * i:4 * i + 4] = 1
            for j in range(4):
                A[n + j, j::4] = -1
            b = np.concatenate([m.CAPACITY[s], -m.DEMAND])
            res = linprog(m.RATE[s].ravel(), A_ub=A, b_ub=b)
            out.append((tuple(i + 1 for i in s),
                        res.fun + m.RENT[s].sum(),
                        m.CAPACITY[s].sum() - m.DEMAND.sum()))
    return out


def workshop(margin):
    """Chapter 11's workshop, planning for hours that may run
    over by the given fraction."""
    A = np.array([[1.0, 1.0], [2.0, 1.0]]) * (1 + margin)
    res = linprog([-40, -30], A_ub=np.vstack([A, [0, 1]]),
                  b_ub=[40, 60, 30])
    return res.x, -res.fun


def reliability(plan, spread=0.05, trials=200_000, seed=0):
    """Chance that the plan fits when sawing and assembly times
    each vary by a random factor with the given spread."""
    rng = np.random.default_rng(seed)
    slow_saw = 1 + spread * rng.normal(size=trials)
    slow_bench = 1 + spread * rng.normal(size=trials)
    saw = (plan[0] + plan[1]) * slow_saw
    bench = (2 * plan[0] + plan[1]) * slow_bench
    return np.mean((saw <= 40) & (bench <= 60))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # --- eight cars -------------------------------------------------
    keep = pareto(CARS)
    print("cars not dominated:", [k + 1 for k in keep], CARS[keep].tolist())
    for i, car in enumerate(CARS):
        by = [j + 1 for j, other in enumerate(CARS)
              if np.all(other <= car) and np.any(other < car)]
        print("  car", i + 1, car, "dominated by", by)
    for w in (0.2, 0.5, 1.0, 2.0, 5.0):
        print("  a year valued at", w, "thousand: car",
              weighted_sum(CARS, [1, w]) + 1)
    found = sorted({weighted_sum(CARS, [1, w]) + 1
                    for w in np.linspace(0.01, 20, 4000)})
    print("  cars found by some weight:", found)
    for limit in (2, 4, 7, 11):
        print("  age at most", limit, ": car", with_limit(CARS, limit) + 1)

    # --- the link: rate against power -----------------------------
    for P in (1, 2, 5, 10, 20, 40):
        x, rate = waterfill(G, P)
        lam = 1 / ((x + 1 / G)[x > 0].max() * LN2)
        print(f"power {P}: rate {rate:.3f}, price {lam:.3f}, channels "
              f"on {(x > 0).sum()}, rate per watt {rate / P:.3f}")
    for lam in (2.5, 2.0, 1.5, 1.0, 0.5, 0.33, 0.1):
        x, rate = at_price(G, lam)
        print(f"price {lam}: power {x.sum():.2f}, rate {rate:.3f}")
    print("largest useful price", G.max() / LN2)
    grid = np.linspace(0.01, 40, 4000)
    eff = np.array([waterfill(G, P)[1] / (P + 2.0) for P in grid])
    best = grid[eff.argmax()]
    x, rate = waterfill(G, best)
    print(f"with 2 W of overhead: best efficiency {eff.max():.4f} at "
          f"power {best:.2f}, rate {rate:.3f}, price "
          f"{1 / ((x + 1 / G)[x > 0].max() * LN2):.4f}")
    for P in (1, 10, 20):
        print("  efficiency at", P, waterfill(G, P)[1] / (P + 2))

    # --- warehouses: cost against spare capacity -------------------
    options = warehouse_options()
    costs = np.array([[c, -spare] for _, c, spare in options])
    keep = pareto(costs)
    for k, (s, c, spare) in enumerate(options):
        print("  rent", s, "cost", round(c, 2), "spare", int(spare),
              "PARETO" if k in keep else "")
    hull = sorted({weighted_sum(costs, [1, w]) for w in
                   np.linspace(0, 20, 20001)})
    print("found by weights:", [options[k][0] for k in hull])
    print("missed by weights:", [options[k][0] for k in keep
                                 if k not in hull])
    for need in (0, 50, 100, 150, 200, 300, 400, 500):
        k = with_limit(costs, -need)
        print("  spare at least", need, "->", options[k][0],
              round(options[k][1]))

    # --- robustness as an objective --------------------------------
    for margin in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        plan, profit = workshop(margin)
        print(f"margin {margin:.2f}: plan {plan.round(2)}, profit "
              f"{profit:.0f}, reliability {reliability(plan):.3f}")

    # --- many objectives ---------------------------------------------
    rng = np.random.default_rng(0)
    for d in (2, 3, 5, 8, 12):
        pts = rng.uniform(size=(1000, d))
        print(f"{d} objectives: {len(pareto(pts))} of 1000 random "
              f"points are not dominated")

    # --- capstone ------------------------------------------------------
    rate, active, x = link_design(G, 10.0)
    print("capstone: rate", round(rate, 4), "channels",
          [k + 1 for k in active], "powers", x.round(3), "total",
          x.sum() + 0.5 * len(active))
    table = []
    for size in range(1, 6):
        idx = list(range(size))
        xp, r = waterfill(G[idx], 10 - 0.5 * size)
        table.append((size, round(r, 4), xp.round(3)))
    print("  best channels first:", table)
    print("  bound ignoring circuit power:", waterfill(G, 10.0)[1])
    lam = 1 / ((x[list(active)] + 1 / G[list(active)]).max() * LN2)
    print("  price of power", round(lam, 4), "rate lost by switching on "
          "one more channel:", [round(rate - link_design(G, 10.0)[0], 4)])
    front = []
    for budget in np.linspace(0.6, 20, 195):
        r, act, xx = link_design(G, budget)
        front.append((budget, r, len(act)))
    front = np.array(front)
    eff = front[:, 1] / front[:, 0]
    k = eff.argmax()
    print(f"  most rate per watt: {eff[k]:.4f} at {front[k, 0]:.1f} W, "
          f"rate {front[k, 1]:.3f}, {int(front[k, 2])} channels")
    for budget in (1, 2, 3, 5, 10, 20):
        r, act, xx = link_design(G, budget)
        print(f"  budget {budget}: rate {r:.3f}, channels {len(act)}, "
              f"rate per watt {r / budget:.3f}")
    changes = [front[i, 0] for i in range(1, len(front))
               if front[i, 2] != front[i - 1, 2]]
    print("  budgets where a channel is added:", np.round(changes, 1))
    for drop in (0.8,):
        worst = G * drop
        r_nom = np.sum(np.log2(1 + worst * x))
        r_re, act2, x2 = link_design(worst, 10.0)
        print(f"  gains 20% lower: nominal plan gives {r_nom:.3f}, "
              f"replanned {r_re:.3f} with {len(act2)} channels")

    # --- capstone: the requirement and the margin ------------------
    from scipy.optimize import brentq
    need = lambda g: brentq(lambda b: link_design(g, b)[0] - 5.0, 1, 20)
    b_nom, b_rob = need(G), need(0.8 * G)
    for name, g, b in (("nominal", G, b_nom), ("20% worse", 0.8 * G, b_rob)):
        r, act, xx = link_design(g, b)
        print(f"  5 Mbit/s with {name} gains: {b:.3f} W, channels "
              f"{[k + 1 for k in act]}, powers {xx.round(3)}")
    r, act, xx = link_design(0.8 * G, b_rob)
    print("  the robust design at nominal gains gives",
          round(float(np.sum(np.log2(1 + G * xx))), 3), "Mbit/s; rate "
          "per watt nominal", 5 / b_nom, "robust", 5 / b_rob)
    r, act, xx = link_design(G, b_nom)
    print("  the nominal design if gains are 20% worse:",
          round(float(np.sum(np.log2(1 + 0.8 * G * xx))), 3))
    rates = [np.log2(1 + G[k] * x[k]) for k in range(3)]
    print("  marginal rates at the 10 W design:",
          (G[:3] / ((1 + G[:3] * x[:3]) * LN2)).round(4), rates)

    # --- three objectives with a solver in the loop ---------------
    from in_practice import evolve, portfolio
    caps = (1.0, 0.5, 0.4, 0.3, 0.25)
    for target in (3, 4, 5, 6, 7, 8, 9):
        row = [portfolio(target, cap) for cap in caps]
        print("return", target, ["  --  " if r is None else f"{r[0]:6.2f}"
                                 for r in row])
    vol, w, p_ret, p_cap = portfolio(6.0, 0.3)
    print("target 6, cap 30%: volatility", round(vol, 3), "holdings",
          (100 * w).round(1), "price of return", round(p_ret, 2),
          "price of the cap", round(p_cap, 2))
    print("  variance", round(vol ** 2, 3), "; target 6.1:",
          round(portfolio(6.1, 0.3)[0] ** 2, 3), "predicted",
          round(vol ** 2 + 0.1 * p_ret, 3), "; cap 29%:",
          round(portfolio(6.0, 0.29)[0] ** 2, 3), "predicted",
          round(vol ** 2 + 0.01 * p_cap, 3))
    solves = sum(portfolio(t, c) is not None
                 for t in np.linspace(3, 9, 13) for c in caps)
    print("  feasible combinations on a 13 by 5 grid:", solves)

    # --- a front from a library -----------------------------------
    import time
    for gens in (20, 50, 200):
        for seed in (1, 2, 3):
            t0 = time.perf_counter()
            power, rate_found, _ = evolve(gens, seed=seed)
            secs = time.perf_counter() - t0
            inside = (power >= 2) & (power <= 20)
            exact = np.array([link_design(G, p)[0] for p in power[inside]])
            short = exact - rate_found[inside]
            print(f"NSGA-II, {gens} generations, seed {seed}: "
                  f"{inside.sum()} designs between 2 and 20 W, mean "
                  f"shortfall {short.mean():.3f}, worst {short.max():.3f} "
                  f"({100 * (short / exact).max():.1f}%), {secs:.1f} s")
