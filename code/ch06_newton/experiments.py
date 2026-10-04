"""Chapter 6: the tables quoted in the text."""
import numpy as np

from from_scratch import bfgs, conjugate_gradient, newton
from problems import (churn_matrix, logistic, rosenbrock, smoother,
                      temperatures)

# 1. Newton in one variable on exp(x) - 2x; minimizer log 2.
x = 0.0
for k in range(6):
    print(f"k={k}  x={x:.10f}  error={abs(x - np.log(2)):.1e}")
    x = x - (np.exp(x) - 2) / np.exp(x)

# 2. Pure Newton on sqrt(1 + x^2): the step is x -> -x^3.
f = lambda x: np.sqrt(1 + x[0] ** 2)
grad = lambda x: np.array([x[0] / np.sqrt(1 + x[0] ** 2)])
hess = lambda x: np.array([[(1 + x[0] ** 2) ** -1.5]])
for start in (0.5, 1.1):
    x, seq = start, []
    for _ in range(4):
        x = -x ** 3
        seq.append(float(f"{x:.4g}"))
    print("pure Newton from", start, seq)
for start in (1.1, 10.0):
    x, norms = newton(f, grad, hess, [start])
    print("damped Newton from", start, norms.size - 1, "steps")

# 3. The churn model: Newton and BFGS, raw and standardized columns.
for label, flag in (("raw", False), ("standardized", True)):
    f, grad, hess = logistic(*churn_matrix(flag))
    x, norms = newton(f, grad, hess, np.zeros(3))
    print(label, "Newton:", norms.size - 1, "steps;",
          " ".join(f"{v:.1e}" for v in norms))
    x, norms = bfgs(f, grad, np.zeros(3))
    print(label, "BFGS:", norms.size - 1, "steps", x.round(4))

# 4. Rosenbrock from (-1.2, 1).
f, grad, hess = rosenbrock()
start = [-1.2, 1.0]
print("Rosenbrock: damped Newton", newton(f, grad, hess, start)[1].size - 1,
      " pure Newton", newton(f, grad, hess, start, damped=False)[1].size - 1,
      " BFGS", bfgs(f, grad, start)[1].size - 1)

# 5. Smoothing 2,000 temperatures: conjugate gradient against
#    gradient descent on the same linear system.
noisy, clean = temperatures()
lam = 100.0
matvec = smoother(lam)
x, res = conjugate_gradient(matvec, noisy)
rms = lambda v: np.sqrt(np.mean(v ** 2))
print("CG steps:", res.size - 1, " error before", round(rms(noisy - clean), 3),
      " after", round(rms(x - clean), 3))
L = 1 + 4 * lam
z, steps, r0 = np.zeros(noisy.size), 0, np.linalg.norm(noisy)
while np.linalg.norm(noisy - matvec(z)) > 1e-8 * r0:
    z += (noisy - matvec(z)) / L
    steps += 1
print("gradient descent steps:", steps, " condition number about", L)

# 6. Newton heads for a saddle on the two-valley function of Chapter 3.
tv = lambda x: (x[0] ** 2 - 1) ** 2 + x[1] ** 2
tg = lambda x: np.array([4 * x[0] * (x[0] ** 2 - 1), 2 * x[1]])
th = lambda x: np.array([[12 * x[0] ** 2 - 4, 0.0], [0.0, 2.0]])
x = np.array([0.3, 0.5])
for _ in range(4):
    x = x - np.linalg.solve(th(x), tg(x))
print("pure Newton from (0.3, 0.5):", x)
x, norms = newton(tv, tg, th, [0.3, 0.5])
print("safeguarded Newton:", x.round(6), norms.size - 1, "steps")
