"""Chapter 8: checks for the exercise solutions."""
import pathlib
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from applications import lasso_cvx, svm, svm_dual
from data import COV, RETURN, cars, parts, standardize

np.set_printoptions(precision=4, suppress=True)

# Exercise 8.1: an LP and its dual.
x = cp.Variable(2, nonneg=True)
primal = cp.Problem(cp.Maximize(5 * x[0] + 4 * x[1]),
                    [x[0] + x[1] <= 8, x[0] <= 5])
primal.solve()
y = cp.Variable(2, nonneg=True)
dual = cp.Problem(cp.Minimize(8 * y[0] + 5 * y[1]),
                  [y[0] + y[1] >= 5, y[0] >= 4])
dual.solve()
print("Ex 1:", x.value.round(4), round(primal.value, 4),
      y.value.round(4), round(dual.value, 4))

# Exercise 8.2: q(lam) = lam - lam^2 / 4.
lam = np.linspace(0, 4, 401)
q = lam - lam ** 2 / 4
print("Ex 2: best lam", lam[q.argmax()], "value", q.max())

# Exercise 8.4: least norm on a plane.
a, b = np.array([1.0, 2.0, 2.0]), 9.0
z = cp.Variable(3)
plane = a @ z == b
prob = cp.Problem(cp.Minimize(cp.sum_squares(z) / 2), [plane])
prob.solve()
print("Ex 4:", z.value.round(4), round(prob.value, 4),
      "multiplier", round(float(plane.dual_value), 4))

# Exercise 8.5: the knapsack with 9 watts.
cost = np.array([6.0, 5.0, 5.0, 3.0])
gain = np.array([30.0, 24.0, 23.0, 9.0])
grid = np.linspace(0, 8, 8001)
vals = [9 * l + np.maximum(gain - l * cost, 0).sum() for l in grid]
plans = [[i for i in range(4) if m >> i & 1] for m in range(16)]
best = max((gain[s].sum(), s) for s in plans if cost[s].sum() <= 9)
print("Ex 5: bound", round(min(vals), 2), "at",
      grid[int(np.argmin(vals))], "best plan", best)

# Exercise 8.7: what the rule book accepts.
u, v = cp.Variable(), cp.Variable()
w3 = cp.Variable(3)
for name, e in [("square(u) + abs(v)", cp.square(u) + cp.abs(v)),
                ("sqrt(square(u))", cp.sqrt(cp.square(u))),
                ("u * v", u * v),
                ("quad_over_lin(u, v)", cp.quad_over_lin(u, v)),
                ("norm(w) <= 1", cp.norm(w3) <= 1),
                ("norm(w) >= 1", cp.norm(w3) >= 1)]:
    print("Ex 7:", name, e.is_dcp())

# Exercise 8.8: a portfolio with a cap on every holding.
for cap in (1.0, 0.3):
    w = cp.Variable(5, nonneg=True)
    enough = RETURN @ w >= 6.0
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, COV)),
                      [cp.sum(w) == 1, enough, w <= cap])
    prob.solve()
    print("Ex 8: cap", cap, w.value.round(3), "volatility",
          round(float(np.sqrt(prob.value)), 3), "price",
          round(float(enough.dual_value), 2))

# Exercise 8.9: ridge against the lasso.
A_raw, price = cars()
A, b = standardize(A_raw), price - price.mean()
T_raw, t_price = cars(n=1000, seed=4)
T, tb = standardize(T_raw, A_raw), t_price - price.mean()
rmse = lambda c: np.sqrt(np.mean((T @ c - tb) ** 2))
for weight in (0.01, 0.03, 0.1, 0.3):
    c = cp.Variable(40)
    fit = cp.sum_squares(A @ c - b) / (2 * len(b))
    cp.Problem(cp.Minimize(fit + weight * cp.sum_squares(c))).solve()
    print("Ex 9: ridge", weight, "nonzero",
          int(np.sum(np.abs(c.value) > 1e-6)), "test",
          round(float(rmse(c.value)), 2), c.value[:3].round(2))
c = lasso_cvx(A, b, 0.3)
print("Ex 9: lasso 0.3 nonzero", int(np.sum(np.abs(c) > 1e-6)),
      "test", round(float(rmse(c)), 2))

# Exercise 8.10: the classifier from its dual.
X, y = parts()
alpha, value = svm_dual(X, y)
w = X.T @ (alpha * y)
free = (alpha > 1e-4) & (alpha < 1 - 1e-4)
b0 = np.mean(y[free] - X[free] @ w)
print("Ex 10: w", w.round(3), "b", round(float(b0), 3),
      "free support vectors", int(free.sum()),
      "primal b", round(float(svm(X, y)[1]), 3))
new = np.array([4.5, 5.0])
print("Ex 10: score of (4.5, 5.0):", round(float(new @ w + b0), 3))
for C in (0.1, 10.0):
    alpha, _ = svm_dual(X, y, C)
    print("Ex 10: C", C, "support vectors",
          int(np.sum(alpha > 1e-5)))
