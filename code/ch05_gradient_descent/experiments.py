"""Chapter 5: the step-size and condition-number tables."""
import numpy as np

from from_scratch import gradient_descent


def bowl(kappa):
    f = lambda x: 0.5 * (x[0] ** 2 + kappa * x[1] ** 2)
    grad = lambda x: np.array([x[0], kappa * x[1]])
    return f, grad


# 1. Step sizes on the bowl with L = 10, started at (10, 1).
f, grad = bowl(10)
for step in (0.01, 0.05, 0.1, 0.19, 0.21, None):
    x, hist = gradient_descent(f, grad, [10.0, 1.0], step,
                               max_iter=2000)
    print(f"step {step}: {hist.size - 1} steps, f = {hist[-1]:.2e}")

# 2. Steps needed to cut f by a factor of a million, from (1, 1).
for kappa in (1, 10, 100, 1000):
    f, grad = bowl(kappa)
    _, fixed = gradient_descent(f, grad, [1.0, 1.0], 1 / kappa, tol=0,
                                max_iter=20_000)
    _, back = gradient_descent(f, grad, [1.0, 1.0], tol=0,
                               max_iter=20_000)
    target = 1e-6 * fixed[0]
    print(f"kappa {kappa:5d}: fixed {np.argmax(fixed <= target):5d}, "
          f"backtracking {np.argmax(back <= target):5d}, "
          f"bound {kappa * np.log(1e6):8.0f}")

# 3. Curvature of the churn loss at its minimizer, raw and scaled.
from scipy.optimize import minimize

from from_scratch import (churn_data, logistic_grad, logistic_loss,
                          standardize)

hours, months, y = churn_data()
print("subscribers:", y.size, " cancelled:", int((y > 0).sum()))
ones = np.ones(y.size)
for name, cols in (("raw", (hours, months)),
                   ("standardized",
                    (standardize(hours), standardize(months)))):
    A = np.column_stack([ones, *cols])
    res = minimize(lambda x: logistic_loss(x, A, y), np.zeros(3),
                   jac=lambda x: logistic_grad(x, A, y),
                   options={"gtol": 1e-9})
    p = 1 / (1 + np.exp(-(A @ res.x)))
    lam = np.linalg.eigvalsh((A.T * (p * (1 - p))) @ A / y.size)
    print(f"{name:13s} Hessian eigenvalues {lam.round(4)}  "
          f"ratio {lam[-1] / lam[0]:.1f}")
