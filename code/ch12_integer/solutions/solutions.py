"""Chapter 12: checks for the exercise solutions."""
import itertools
import pathlib
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import branch_and_bound
from in_practice import JOBS, schedule_cp, schedule_milp

COST = np.array([6.0, 5.0, 5.0, 3.0])
GAIN = np.array([30.0, 24.0, 23.0, 9.0])

# Exercise 12.1: a small problem by listing.
A = np.array([[2.0, 3.0], [4.0, 1.0]])
b = np.array([12.0, 10.0])
c = np.array([3.0, 4.0])
lp = linprog(-c, A_ub=A, b_ub=b)
pts = [(i, j) for i in range(7) for j in range(7) if np.all(A @ [i, j] <= b)]
best = max(pts, key=lambda p: c @ p)
print("Ex 1: LP", lp.x, -lp.fun, "best whole", best, c @ best,
      "ties", [p for p in pts if c @ p == c @ best])


def relax(budget, fixed):
    """Fractional knapsack with some channels fixed on or off."""
    left = budget - sum(COST[j] for j, v in fixed.items() if v == 1)
    if left < 0:
        return None, None
    value = sum(GAIN[j] for j, v in fixed.items() if v == 1)
    x = dict(fixed)
    for j in np.argsort(-GAIN / COST):
        if j in fixed:
            continue
        x[j] = min(1.0, left / COST[j])
        left -= x[j] * COST[j]
        value += x[j] * GAIN[j]
    return value, x


def tree(budget, fixed=None, depth=0, best=[0.0]):
    fixed = fixed or {}
    value, x = relax(budget, fixed)
    label = ", ".join(f"x{j + 1}={v}" for j, v in fixed.items()) or "root"
    if value is None:
        print("  " * depth + f"[{label}] infeasible")
        return
    frac = [j for j, v in x.items() if 0 < v < 1]
    note = "" if frac else "  whole"
    cut = value <= best[0] and bool(frac)
    print("  " * depth + f"[{label}] bound {value:.1f}{note}"
          + ("  pruned" if cut else ""))
    if not frac:
        best[0] = max(best[0], value)
        return
    if cut:
        return
    for v in (1, 0):
        tree(budget, {**fixed, frac[0]: v}, depth + 1, best)


print("The tree of Section 12.4, for 10 watts:")
tree(10.0, best=[0.0])

# Exercises 12.2 and 12.3: the knapsack with 9 watts.
print("Ex 2: bound", relax(9.0, {}))
tree(9.0, best=[0.0])
print("Ex 3: branch and bound",
      branch_and_bound(GAIN, COST[None], [9.0], [1] * 4))

# Exercise 12.4: cover cuts for 9 watts.
covers = [s for r in (2, 3) for s in itertools.combinations(range(4), r)
          if COST[list(s)].sum() > 9
          and all(COST[list(t)].sum() <= 9
                  for t in itertools.combinations(s, len(s) - 1))]
print("Ex 4: minimal covers", [[j + 1 for j in s] for s in covers])
rows = [COST] + [np.isin(range(4), s).astype(float) for s in covers]
rhs = [9.0] + [len(s) - 1.0 for s in covers]
res = linprog(-GAIN, A_ub=np.array(rows), b_ub=rhs, bounds=[(0, 1)] * 4)
print("Ex 4: bound with all cover cuts", -res.fun, res.x)
res = linprog(-GAIN, A_ub=np.array(rows[:2]), b_ub=rhs[:2],
              bounds=[(0, 1)] * 4)
print("Ex 4: bound with the first cut", -res.fun, res.x)

# Exercise 12.8: the workshop with setup costs.
# Variables: desks, bookcases, make_desks, make_bookcases.
c = np.array([40.0, 30.0, -300.0, -500.0])
A = np.array([[1.0, 1, 0, 0], [2, 1, 0, 0], [0, 1, 0, 0],
              [1, 0, -30, 0], [0, 1, 0, -30]])
b = np.array([40.0, 60, 30, 0, 0])
print("Ex 8:", branch_and_bound(c, A, b, [40, 40, 1, 1]))
for setup in (100.0, 200.0, 300.0):
    c[2:] = [-300.0, -setup]
    print("Ex 8: bookcase setup", setup,
          branch_and_bound(c, A, b, [40, 40, 1, 1])[:2])

# Exercise 12.9: equal items.
for n in (11, 13):
    x, value, solved, root = branch_and_bound(
        np.ones(n), 2 * np.ones((1, n)), [n], [1] * n)
    print("Ex 9:", n, value, root, solved, 2 ** n)
x, value, solved, root = branch_and_bound(
    np.ones(13), np.vstack([2 * np.ones(13), np.ones(13)]), [13, 6],
    [1] * 13)
print("Ex 9: with the cut sum(x) <= 6:", value, root, solved)

# Exercise 12.10: a fifth job.
jobs = JOBS + [[(2, 2), (0, 3), (1, 2)]]
value, bound, secs, plan, status = schedule_cp(jobs)
res = schedule_milp(jobs)
print("Ex 10: CP-SAT", value, status, "MILP", round(res.fun, 6),
      "bench hours", sum(h for job in jobs for m, h in job if m == 1))
