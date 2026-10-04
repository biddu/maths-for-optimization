"""Chapter 8: every number quoted in the chapter."""
import time

import cvxpy as cp
import numpy as np

from applications import lasso_cvx, portfolio, svm, svm_dual
from data import ASSETS, STORES, TRIPS, cars, parts, standardize
from from_scratch import (best_response, dual_function, lasso,
                          solve_dual)

np.set_printoptions(precision=4, suppress=True)

# --- The dual of Version A -------------------------------------
for y in [(3, 0, 0), (2.5, 0.5, 0), (2, 1, 0), (2, 1.5, 0.5)]:
    ok = y[0] + y[1] >= 3 and y[0] + y[2] >= 2
    print("y =", y, "feasible", ok, "bound", 10*y[0] + 6*y[1] + 8*y[2])
y = cp.Variable(3, nonneg=True)
dual = cp.Problem(cp.Minimize(10 * y[0] + 6 * y[1] + 8 * y[2]),
                  [y[0] + y[1] >= 3, y[0] + y[2] >= 2])
dual.solve()
print("dual LP:", y.value.round(4), round(dual.value, 4))

# --- The dual of the power problem ------------------------------
g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
P = 10.0
rate = lambda x: np.sum(np.log2(1 + g * x))
for lam in [0.1, 0.2, 0.3, 0.3298, 0.4, 0.5, 0.8]:
    x = best_response(lam, g)
    feas = x * min(1.0, P / x.sum())
    print(f"lam {lam:6.4f}: wanted {x.sum():7.3f} W, bound "
          f"{-dual_function(lam, g, P):.4f}, feasible rate "
          f"{rate(feas):.4f}, gap "
          f"{-dual_function(lam, g, P) - rate(feas):.4f}")
lam = solve_dual(g, P)
print("best price", round(lam, 4), best_response(lam, g).round(3))
count, lo, hi = 0, 0.0, g.max() / np.log(2)
while hi - lo > 1e-10:
    lo, hi, count = (lo + hi) / 2, hi, count + 1
print("bisection steps", count)

rng = np.random.default_rng(0)
big = rng.exponential(1.0, 1000)
t0 = time.perf_counter(); lam = solve_dual(big, 100.0)
t1 = time.perf_counter()
p = cp.Variable(1000, nonneg=True)
prob = cp.Problem(cp.Maximize(cp.sum(cp.log1p(cp.multiply(big, p)))),
                  [cp.sum(p) <= 100])
t2 = time.perf_counter(); prob.solve(); t3 = time.perf_counter()
xb = best_response(lam, big)
print(f"1000 channels: bisection {1e3*(t1-t0):.1f} ms, CVXPY "
      f"{1e3*(t3-t2):.0f} ms, largest difference "
      f"{np.abs(xb - p.value).max():.5f}, used {(xb > 0).sum()}")

# --- A dual with a gap: the knapsack of Chapter 1 ---------------
cost = np.array([6.0, 5.0, 5.0, 3.0])
gain = np.array([30.0, 24.0, 23.0, 9.0])
bound = lambda lam, B: B*lam + np.maximum(gain - lam*cost, 0).sum()
for B in (10.0, 9.0):
    grid = np.linspace(0, 8, 8001)
    vals = np.array([bound(l, B) for l in grid])
    best = max(gain[list(s)].sum() for s in
               [[i for i in range(4) if m >> i & 1] for m in range(16)]
               if cost[list(s)].sum() <= B)
    print(f"knapsack {B}: dual bound {vals.min():.2f} at "
          f"{grid[vals.argmin()]:.2f}, best plan {best}")

# --- The depot as a cone program ---------------------------------
d = cp.Variable(2)
socp = cp.Problem(cp.Minimize(
    TRIPS @ cp.hstack([cp.norm(d - s) for s in STORES])))
socp.solve()
print("depot", d.value.round(4), round(socp.value, 3))

# --- Portfolio ----------------------------------------------------
print(ASSETS)
for target in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]:
    w, risk, price = portfolio(target)
    print(f"target {target:3.1f}: {np.abs(w).round(2)} vol "
          f"{risk:6.2f} price {price:7.2f}")
w6, r6, p6 = portfolio(6.0)
w61, r61, _ = portfolio(6.1)
print("variance 6.0:", round(r6**2, 3), "6.1:", round(r61**2, 3),
      "difference", round(r61**2 - r6**2, 3), "predicted",
      round(0.1 * p6, 3))

# --- Lasso --------------------------------------------------------
A_raw, price = cars()
A, b = standardize(A_raw), price - price.mean()
T_raw, t_price = cars(n=1000, seed=4)
T, tb = standardize(T_raw, A_raw), t_price - price.mean()
rmse = lambda x: np.sqrt(np.mean((T @ x - tb) ** 2))
ls = np.linalg.lstsq(A, b, rcond=None)[0]
print("least squares: first five", ls[:5].round(2), "largest junk",
      np.abs(ls[3:]).max().round(2), "train rmse",
      np.sqrt(np.mean((A @ ls - b) ** 2)).round(2),
      "test rmse", rmse(ls).round(2))
for lam in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
    x = lasso(A, b, lam)
    print(f"lam {lam:4.2f}: first three {x[:3].round(2)}, nonzero "
          f"{(x != 0).sum():2d}, junk nonzero {(x[3:] != 0).sum():2d},"
          f" train {np.sqrt(np.mean((A @ x - b) ** 2)):.2f}"
          f" test rmse {rmse(x):.2f}")
print("largest |A^T b|/n:", np.abs(A.T @ b).max() / len(b))
t0 = time.perf_counter(); xs = lasso(A, b, 0.3)
t1 = time.perf_counter(); xc = lasso_cvx(A, b, 0.3)
t2 = time.perf_counter()
print(f"ISTA {1e3*(t1-t0):.0f} ms, CVXPY {1e3*(t2-t1):.0f} ms, "
      f"largest difference {np.abs(xs - xc).max():.6f}, CVXPY "
      f"entries below 1e-6: {(np.abs(xc) < 1e-6).sum()}, exact "
      f"zeros {(xc == 0).sum()}")
truth = np.array([-1.5, -0.05, 0.04]) * A_raw[:, :3].std(axis=0)
print("true standardized coefficients", truth.round(2))

# --- SVM ----------------------------------------------------------
X, y = parts()
w, b0, alpha, value = svm(X, y)
a, dval = svm_dual(X, y)
sv = alpha > 1e-5
print("svm w", w.round(3), "b", round(float(b0), 3), "primal",
      round(value, 4), "dual", round(dval, 4))
print("margin width", round(2 / np.linalg.norm(w), 3),
      "support vectors", sv.sum(), "alphas", alpha[sv].round(3),
      "at bound", (alpha > 1 - 1e-5).sum())
print("w from multipliers", (X.T @ (alpha * y)).round(3),
      "misclassified", int(np.sum(y * (X @ w + b0) < 0)),
      "sum alpha*y", round(float(alpha @ y), 6))
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    w, b0, alpha, value = svm(X, y, C)
    print(f"C {C:6.2f}: width {2/np.linalg.norm(w):.3f}, support "
          f"vectors {(alpha > 1e-5).sum()}, misclassified "
          f"{int(np.sum(y * (X @ w + b0) < 0))}")

# --- Where it fails -----------------------------------------------
x = cp.Variable()
for name, prob in [
    ("infeasible", cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])),
    ("unbounded", cp.Problem(cp.Minimize(x), [x <= 0])),
    ("no Slater", cp.Problem(cp.Minimize(x), [cp.square(x) <= 0])),
]:
    try:
        prob.solve()
        print(name, prob.status, prob.value,
              prob.constraints[-1].dual_value)
    except Exception as err:
        print(name, "error:", type(err).__name__)
try:
    cp.Problem(cp.Minimize(cp.log(1 + cp.exp(x)))).solve()
except Exception as err:
    print(type(err).__name__, str(err)[:200])

# A solver that stops early: the SVM dual with C = 10.
import warnings
X, y = parts()
for solver in (cp.OSQP, cp.CLARABEL):
    a = cp.Variable(len(y))
    gain = cp.sum(a) - cp.sum_squares(X.T @ cp.multiply(a, y)) / 2
    prob = cp.Problem(cp.Maximize(gain),
                      [a >= 0, a <= 10, a @ y == 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=solver)
    print(solver, prob.status, round(prob.value, 4),
          prob.solver_stats.num_iters, "primal",
          round(svm(X, y, 10.0)[3], 4))

# A portfolio is only as good as its inputs.
from data import COV, RETURN
for label, k, value in [("as given", 4, 9.5), ("small 10.5", 4, 10.5)]:
    ret = RETURN.copy(); ret[k] = value
    w = cp.Variable(5, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, COV)),
                      [cp.sum(w) == 1, ret @ w >= 6.0])
    prob.solve()
    print(label, w.value.round(2), np.sqrt(prob.value).round(2))
