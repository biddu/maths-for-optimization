"""Chapter 6: numerical checks for the exercise solutions."""
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import conjugate_gradient, line_search
from problems import churn_matrix, logistic, smoother, temperatures

# Exercise 6.1.
print("Ex 1:", np.linalg.solve([[2.0, 1.0], [1.0, 2.0]], [3.0, 0.0]))

# Exercise 6.2: Newton on x - log x.
x = 0.5
for _ in range(4):
    x = 2 * x - x * x
    print("Ex 2:", x, 1 - x)

# Exercise 6.6.
for n in (1e4, 1e6):
    print(f"Ex 6: n={n:.0e}  Hessian {8 * n * n / 1e9:.1f} GB,"
          f"  solve {n ** 3 / 3:.1e} operations,"
          f"  L-BFGS (m=10) {8 * 20 * n / 1e6:.1f} MB")

# Exercise 6.7: conjugate gradient by hand.
A = np.array([[4.0, 1.0], [1.0, 3.0]])
x, res = conjugate_gradient(lambda v: A @ v, np.array([1.0, 2.0]))
print("Ex 7:", x, res, 1 / 11, 7 / 11)

# Exercise 6.8: how good is the BFGS estimate of the inverse Hessian?
f, grad, hess = logistic(*churn_matrix(standardized=True))
x, I, Hinv = np.zeros(3), np.eye(3), np.eye(3)
g = grad(x)
for k in range(1000):
    if np.linalg.norm(g) <= 1e-8:
        break
    d = -Hinv @ g
    s = line_search(f, x, g, d) * d
    g_new = grad(x + s)
    y = g_new - g
    if y @ s > 1e-12:
        if k == 0:
            Hinv = (y @ s) / (y @ y) * I
        rho = 1 / (y @ s)
        V = I - rho * np.outer(s, y)
        Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)
    x, g = x + s, g_new
true_inv = np.linalg.inv(hess(x))
print("Ex 8: steps", k)
print(Hinv.round(2))
print(true_inv.round(2))
print("relative error",
      np.linalg.norm(Hinv - true_inv) / np.linalg.norm(true_inv))

# Exercise 6.9: how much memory does L-BFGS need?
noisy, _ = temperatures()
matvec = smoother(100.0)
q = lambda v: 0.5 * v @ matvec(v) - noisy @ v
dq = lambda v: matvec(v) - noisy
for m in (1, 3, 10, 30):
    res = minimize(q, np.zeros(noisy.size), jac=dq, method="L-BFGS-B",
                   options={"maxcor": m, "gtol": 1e-6, "ftol": 1e-15,
                            "maxiter": 5000})
    print(f"Ex 9: memory {m:2d}: {res.nit} steps,"
          f" |g| = {np.linalg.norm(dq(res.x)):.1e}")

# Exercise 6.10: Newton on x^4.
x = 1.0
for _ in range(5):
    x = x - (4 * x ** 3) / (12 * x ** 2)
    print("Ex 10:", x)
