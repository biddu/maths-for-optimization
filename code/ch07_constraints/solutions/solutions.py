"""Chapter 7: numerical checks for the exercise solutions."""
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import rate, waterfill

# Exercise 7.2.
K = np.array([[2, 0, 0, 1, 1], [0, 2, 0, 1, 0], [0, 0, 2, 1, -1],
              [1, 1, 1, 0, 0], [1, 0, -1, 0, 0]], dtype=float)
print("Ex 2:", np.linalg.solve(K, [0, 0, 0, 3, 1]))

# Exercise 7.4.
for limit in (2, 4):
    res = minimize(lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2, [0, 0],
                   constraints=[{"type": "ineq",
                                 "fun": lambda x: limit - x[0] - x[1]}])
    print("Ex 4:", limit, res.x.round(4), res.multipliers)

# Exercise 7.6.
g = np.array([1.0, 0.25, 0.5])
for P in (10.0, 2.0):
    x, level = waterfill(g, P)
    print("Ex 6:", P, x, level, rate(x, g))

# Exercise 7.8: the can with lids that cost twice as much.
V = 330.0
r = (V / (4 * np.pi)) ** (1 / 3)
print("Ex 8:", r, 4 * r, np.pi * r * r * 4 * r)
res = minimize(lambda z: 4 * np.pi * z[0] ** 2 + 2 * np.pi * z[0] * z[1],
               [3.0, 10.0],
               constraints=[{"type": "eq", "fun":
                             lambda z: np.pi * z[0] ** 2 * z[1] - V}])
print("Ex 8:", res.x)

# Exercise 7.9: a quadratic penalty for an inequality.
f = lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2
gcon = lambda x: x[0] + x[1] - 2
for rho in (1, 10, 100, 1000):
    res = minimize(lambda x: f(x) + 0.5 * rho * max(0, gcon(x)) ** 2,
                   [0.0, 0.0], method="BFGS")
    print(f"Ex 9: rho {rho:5d}  x = {res.x.round(4)}  "
          f"violation {gcon(res.x):.4f}  rho*violation {rho * gcon(res.x):.4f}")

# Exercise 7.10: fifty channels.
rng = np.random.default_rng(7)
g = rng.uniform(0.05, 2.0, 50)
t0 = time.perf_counter()
x, level = waterfill(g, 20.0)
t1 = time.perf_counter()
res = minimize(lambda x: -rate(x, g), np.full(50, 0.4),
               bounds=[(0, None)] * 50,
               constraints=[{"type": "ineq", "fun": lambda x: 20 - x.sum()}],
               options={"maxiter": 500})
t2 = time.perf_counter()
print("Ex 10: active", int((x > 0).sum()), " level", round(level, 4),
      " rate", round(rate(x, g), 4))
print("Ex 10: SciPy rate", round(-res.fun, 4), " largest difference",
      round(np.abs(res.x - x).max(), 4), " steps", res.nit)
print(f"Ex 10: times {1e3 * (t1 - t0):.2f} ms and {1e3 * (t2 - t1):.0f} ms")
