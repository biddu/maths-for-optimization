"""Chapter 12: every number quoted in the chapter."""
import itertools
import json
import pathlib
import time

import numpy as np
from scipy.optimize import linprog

from from_scratch import branch_and_bound
from in_practice import JOBS, random_shop, schedule_cp, schedule_milp
from models import (CAPACITY, DEMAND, RATE, RENT, location_model,
                    random_sites)

HERE = pathlib.Path(__file__).resolve().parent
COST = np.array([[6.0, 5.0, 5.0, 3.0]])
GAIN = np.array([30.0, 24.0, 23.0, 9.0])

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # --- two machines: rounding fails -----------------------------
    A = np.array([[6.0, 5.0], [1.0, 3.0]])
    b = np.array([42.0, 19.0])
    c = np.array([9.0, 8.0])
    lp = linprog(-c, A_ub=A, b_ub=b)
    print("LP optimum", lp.x, -lp.fun)
    pts = [(i, j) for i in range(8) for j in range(8)
           if np.all(A @ [i, j] <= b)]
    ranked = sorted(pts, key=lambda p: -(c @ p))[:5]
    print("feasible whole points", len(pts), "best five",
          [(p, float(c @ p)) for p in ranked])
    for p in ((2, 6), (3, 5), (2, 5), (3, 6)):
        print(" rounded", p, "feasible", bool(np.all(A @ p <= b)),
              "profit", float(c @ p))
    x, value, solved, root = branch_and_bound(c, A, b, [10, 10])
    print("branch and bound", x, value, "subproblems", solved)

    # --- the knapsack of Chapter 1 ------------------------------------
    x, value, solved, root = branch_and_bound(GAIN, COST, [10.0], [1] * 4)
    print("knapsack", x, value, "subproblems", solved, "root", root)
    cuts = [[1, 1, 0, 0], [1, 0, 1, 0]]
    for k in range(3):
        rows = np.vstack([COST] + [np.array([r], float) for r in cuts[:k]])
        res = linprog(-GAIN, A_ub=rows, b_ub=[10.0] + [1.0] * k,
                      bounds=[(0, 1)] * 4)
        print(k, "cuts: bound", round(-res.fun, 2), res.x)

    # --- larger knapsacks ----------------------------------------------
    rng = np.random.default_rng(5)
    for n in (10, 20, 30, 40, 60):
        counts = []
        for _ in range(5):
            w = rng.integers(10, 60, n).astype(float)
            v = w + rng.integers(-5, 15, n)
            counts.append(branch_and_bound(v, w[None], [w.sum() // 3],
                                           [1] * n)[2])
        print(f"{n} items: subproblems {counts}, plans {2.0 ** n:.1e}")
    for n in (9, 11, 13, 15, 17):
        t0 = time.perf_counter()
        x, value, solved, root = branch_and_bound(
            np.ones(n), 2 * np.ones((1, n)), [n], [1] * n)
        print(f"{n} equal items: value {value}, bound {root}, "
              f"subproblems {solved}, plans {2 ** n}, "
              f"{time.perf_counter() - t0:.1f} s")

    # --- renting warehouses -----------------------------------------
    W, C = RATE.shape
    whole = range(W * C, W * C + W)
    best = min((RENT[list(s)].sum(), s) for r in range(1, 5)
               for s in itertools.combinations(range(4), r)
               if CAPACITY[list(s)].sum() >= DEMAND.sum())
    print("cheapest rent that covers demand", best)
    for name, kw in (("M = capacity", {}), ("M = 10000", {"big": 1e4}),
                     ("strong", {"strong": True})):
        cc, AA, bb, up = location_model(**kw)
        x, value, solved, root = branch_and_bound(cc, AA, bb, up, whole)
        print(f"{name}: cost {-value}, open {x[W * C:]}, subproblems "
              f"{solved}, first bound {-root:.1f}")
    print(x[:W * C].reshape(W, C))
    cc, AA, bb, up = location_model()
    res = linprog(-cc, A_ub=AA, b_ub=bb, bounds=list(zip(0 * up, up)))
    print("relaxation: open", res.x[W * C:], "cost", res.fun)
    sites = random_sites()
    W2, C2 = sites[3].shape
    for name, kw in (("M = capacity", {}), ("M = 10000", {"big": 1e4}),
                     ("strong", {"strong": True})):
        cc, AA, bb, up = location_model(*sites, **kw)
        t0 = time.perf_counter()
        x, value, solved, root = branch_and_bound(
            cc, AA, bb, up, range(W2 * C2, W2 * C2 + W2))
        print(f"10 sites, 15 cities, {name}: cost {-value}, subproblems "
              f"{solved}, first bound {-root:.1f}, rows {len(bb)}, "
              f"{time.perf_counter() - t0:.1f} s")

    # --- a big M and a tolerance -------------------------------------
    for M in (10.0, 1e3, 1e6, 1e7, 1e8):
        x, value, solved, root = branch_and_bound(
            [0.0, -100.0], np.array([[-1.0, 0.0], [1.0, -M]]),
            [-5.0, 0.0], [10, 1], integer=[1])
        print(f"M = {M:g}: ship {x[0]}, open {x[1]:g}, cost {-value:g}, "
              f"first bound {-root:g}")

    # --- scheduling ---------------------------------------------------
    value, bound, secs, plan, status = schedule_cp(JOBS)
    print("workshop schedule: CP-SAT", value, bound, status, plan)
    res = schedule_milp(JOBS)
    print("  MILP", res.fun, res.mip_dual_bound, "nodes", res.mip_node_count)
    shop = random_shop()
    value, bound, secs, plan, status = schedule_cp(shop)
    print(f"10 jobs, 5 machines: CP-SAT {value} (bound {bound}) in "
          f"{secs:.2f} s, {status}")
    history = []
    for limit in (0.5, 1, 2, 5, 10, 20, 40, 80):
        res = schedule_milp(shop, limit)
        history.append((limit, None if res.x is None else float(res.fun),
                        float(res.mip_dual_bound), int(res.mip_node_count)))
        print("  MILP limit", history[-1])
    (HERE / "gap_history.json").write_text(json.dumps(
        {"optimum": value, "cp_seconds": secs, "milp": history}))
