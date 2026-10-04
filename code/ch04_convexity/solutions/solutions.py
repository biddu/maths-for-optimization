"""Chapter 4: numerical checks for the exercise solutions."""
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import from_scratch as fs

# Exercise 4.3: Hessian eigenvalues.
print("Ex 3a:", np.linalg.eigvalsh([[2, 1], [1, 2]]))
print("Ex 3b:", np.linalg.eigvalsh([[2, -3], [-3, 2]]))
print("Ex 3c:", np.linalg.eigvalsh([[1, 2], [2, 4]]))
print("Ex 3d:", np.linalg.eigvalsh([[0, 1], [1, 0]]))

# Exercise 4.8: the certificate for a quadratic.
A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([3.0, 0.0])
f = lambda x: 0.5 * x @ A @ x - b @ x
x_star = np.linalg.solve(A, b)
m = np.linalg.eigvalsh(A)[0]
g = A @ np.zeros(2) - b
print("Ex 8:", x_star, f(x_star), m, g @ g / (2 * m),
      f(np.zeros(2)) - f(x_star), np.linalg.norm(g) / m,
      np.linalg.norm(x_star))

# Exercise 4.9: the chord test on five more functions.
tests = [
    ("x^3", lambda x: x[0] ** 3, -2, 2, 1),
    ("max(x1, x2, 0)", lambda x: max(x[0], x[1], 0.0), -5, 5, 2),
    ("|x|^2 - |x|", lambda x: x @ x - np.linalg.norm(x), -2, 2, 2),
    ("log(1 + e^-x)", lambda x: np.logaddexp(0, -x[0]), -5, 5, 1),
    ("1/x", lambda x: 1 / x[0], 0.1, 5, 1),
]
for name, fun, low, high, dim in tests:
    print(f"Ex 9: {name:16s}",
          round(fs.chord_violation(fun, low, high, dim)[0], 4))

# Exercise 4.10: a tenth store.
fs.STORES = np.vstack([fs.STORES, [9.0, 9.0]])
fs.TRIPS = np.append(fs.TRIPS, 6.0)
rng = np.random.default_rng(5)
runs = [minimize(fs.one_depot, rng.uniform(0, 10, 2)) for _ in range(20)]
print("Ex 10:", runs[0].x.round(3), min(r.fun for r in runs),
      max(r.fun for r in runs))
options = {"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20_000}
costs = [minimize(fs.two_depots, rng.uniform(0, 10, 4),
                  method="Nelder-Mead", options=options).fun
         for _ in range(200)]
values, counts = np.unique(np.round(costs, 1), return_counts=True)
print("Ex 10:", dict(zip(values.tolist(), counts.tolist())))
