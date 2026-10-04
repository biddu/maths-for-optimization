"""Chapter 11: every number quoted in the chapter."""
import math

import numpy as np
from scipy.optimize import linprog

from from_scratch import corners, simplex, two_phase
from in_practice import (DEMAND, FEEDS, RATE, SUPPLY, blend, diagnose,
                         fat, fiber, protein, ship)

A = np.array([[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]])
b = np.array([40.0, 60.0, 30.0])
c = np.array([40.0, 30.0])


def workshop(saw=40.0, assembly=60.0, kits=30.0, profit=c):
    return simplex(profit, A, np.array([saw, assembly, kits]))


def klee_minty(n):
    cc = np.array([2.0 ** (n - j) for j in range(1, n + 1)])
    M = np.eye(n)
    for i in range(n):
        for j in range(i):
            M[i, j] = 2.0 ** (i - j + 1)
    return cc, M, np.array([5.0 ** i for i in range(1, n + 1)])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # --- the workshop ---------------------------------------------
    x, value, prices, path = workshop()
    print("corners:", [tuple(v) for v in corners(A, b)],
          "profits:", [float(c @ v) for v in corners(A, b)])
    print("simplex:", x, value, "prices", prices, "path",
          [tuple(p) for p in path])
    for saw in (25, 30, 39, 40, 41, 45, 46, 50):
        x, value, prices, _ = workshop(saw=saw)
        print(f"sawing {saw}: plan {x}, profit {value}, prices {prices}")
    for p1 in (25, 30, 35, 55, 60, 65):
        x, value, _, _ = workshop(profit=np.array([p1, 30.0]))
        print(f"desk profit {p1}: plan {x}, profit {value}")
    for name, col, p in (("chair", [1, 1, 0], 25.0),
                         ("stool", [0.5, 0.5, 0], 18.0)):
        A3 = np.column_stack([A, col])
        x, value, prices, _ = simplex(np.array([40.0, 30.0, p]), A3, b)
        print(name, "plan", x, "profit", value, "prices", prices)
    x, value, prices, path = workshop(kits=20)
    res = linprog(-c, A_ub=A, b_ub=[40, 60, 20])
    print("kits 20: plan", x, value, "our prices", prices, "HiGHS prices",
          -res.ineqlin.marginals, "pivots", len(path) - 1)
    for saw in (39, 41):
        print("  kits 20, sawing", saw, workshop(saw=saw, kits=20)[1])

    # --- a first corner when the origin is not feasible ----------
    A4, b4 = np.vstack([A, [-1.0, -1.0]]), np.append(b, -25.0)
    x, value, prices, path = two_phase(c, A4, b4)
    print("contract for 25:", x, value, prices, [tuple(p) for p in path])
    try:
        two_phase(c, A4, np.append(b, -45.0))
    except ValueError as err:
        print("contract for 45:", err)
    rows = np.array([-protein, fiber, -fat, [0, 0, 0, 1.0],
                     np.ones(4), -np.ones(4)])
    rhs = np.array([-20.0, 5.0, -3.5, 0.05, 1.0, -1.0])
    feed_cost = np.array([0.25, 0.45, 0.18, 1.10])
    x, value, prices, path = two_phase(-feed_cost, rows, rhs)
    print("blend by two_phase:", (100 * x).round(1), round(-1000 * value, 2),
          "prices", (1000 * prices).round(2), "pivots", len(path) - 1)
    rng = np.random.default_rng(1)
    wrong = infeasible = 0
    for _ in range(300):
        mm, nn = rng.integers(2, 7), rng.integers(2, 7)
        M = np.vstack([rng.normal(size=(mm, nn)).round(1), np.ones(nn)])
        h = np.append(3 * rng.normal(size=mm).round(1), 10.0)
        g = rng.normal(size=nn).round(1)
        ref = linprog(-g, A_ub=M, b_ub=h)
        try:
            ok = abs(two_phase(g, M, h, rule="bland")[1] + ref.fun) < 1e-6
        except ValueError:
            ok, infeasible = ref.status == 2, infeasible + 1
        wrong += not ok
    print("300 random problems against HiGHS: disagreements", wrong,
          "infeasible", infeasible)

    # --- how many pivots ---------------------------------------------
    rng = np.random.default_rng(0)
    for n in (5, 10, 20, 50, 100, 200):
        piv = []
        for _ in range(10):
            M = rng.uniform(0, 1, (n, n))
            piv.append(len(simplex(rng.uniform(0, 1, n), M,
                                   rng.uniform(1, 2, n) * n / 2)[3]) - 1)
        print(f"random {n} x {n}: pivots mean {np.mean(piv):.1f}, max "
              f"{max(piv)}; possible bases {math.comb(2 * n, n):.1e}")
    for n in (3, 5, 8, 10):
        cc, M, bb = klee_minty(n)
        print(f"Klee-Minty {n}: largest-coefficient "
              f"{len(simplex(cc, M, bb)[3]) - 1} pivots, Bland "
              f"{len(simplex(cc, M, bb, rule='bland')[3]) - 1}")
    cb = np.array([0.75, -20, 0.5, -6])
    Mb = np.array([[0.25, -8, -1, 9], [0.5, -12, -0.5, 3], [0, 0, 1, 0]])
    bb = np.array([0.0, 0, 1])
    try:
        simplex(cb, Mb, bb, max_pivots=10_000)
    except RuntimeError as err:
        print("Beale, largest coefficient:", err)
    T = simplex(cb, Mb, bb, max_pivots=6 * 3 + 1) if False else None
    x, value, _, path = simplex(cb, Mb, bb, rule="bland")
    print("Beale, Bland:", x, value, len(path) - 1, "pivots")

    # --- the blend ---------------------------------------------------
    res = blend()
    print("blend", dict(zip(FEEDS, (100 * res.x).round(1))), "cost",
          round(1000 * res.fun, 2), "per tonne")
    print("  protein, fiber, fat:", protein @ res.x, fiber @ res.x,
          fat @ res.x, "rule prices per tonne",
          (-1000 * res.ineqlin.marginals).round(2), "fish cap price",
          res.upper.marginals.round(3))
    for p in (20, 21, 22, 23):
        r = blend(min_protein=p)
        print(f"  protein {p}: cost per tonne {1000 * r.fun:.2f}, "
              f"mix {(100 * r.x).round(1)}")
    r = blend(min_protein=25)
    print("  protein 25:", r.status, r.message)
    mix, give = diagnose(min_protein=25)
    print("  diagnosis: mix", (100 * mix).round(1), "give", give.round(3))
    best = linprog(-protein, A_ub=np.array([fiber, -fat]), b_ub=[5, -3.5],
                   A_eq=np.ones((1, 4)), b_eq=[1],
                   bounds=[(0, None)] * 3 + [(0, 0.05)])
    print("  highest protein within the rules:", round(-best.fun, 2))
    r = linprog([-1, -1], A_ub=[[1, -1]], b_ub=[1])
    print("  an unbounded model:", r.status, r.message)

    # --- shipping ----------------------------------------------------
    status, plan, total, prices = ship()
    print("plan", np.array(plan), "cost", total, "routes used",
          int(np.sum(np.array(plan) > 1e-9)))
    print("warehouse prices", np.round(prices[0], 2), "city prices",
          np.round(prices[1], 2))
    print("dual value", np.dot(prices[0], SUPPLY) + np.dot(prices[1], DEMAND))
    print("demand at D + 10:", ship(demand=[80, 70, 110, 70])[2])
    print("supply at W1 + 10:", ship(supply=[130, 150, 90])[2])
    print("supply at W2 + 10:", ship(supply=[120, 160, 90])[2])
    status, plan, total, prices = ship(supply=[120, 100, 90])
    print("W2 cut to 100: status", status, "(0 is optimal, 2 infeasible)")
