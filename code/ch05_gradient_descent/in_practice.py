"""Chapter 5, in practice: who cancels? Logistic regression."""
import numpy as np
from scipy.optimize import minimize

from from_scratch import (churn_data, gradient_descent,
                          logistic_grad, logistic_loss,
                          standardize)

hours, months, y = churn_data()
ones = np.ones(y.size)
raw = np.column_stack([ones, hours, months])
scaled = np.column_stack([ones, standardize(hours),
                          standardize(months)])

for name, A in (("raw", raw), ("standardized", scaled)):
    f = lambda x: logistic_loss(x, A, y)
    grad = lambda x: logistic_grad(x, A, y)
    L = np.linalg.eigvalsh(A.T @ A)[-1] / (4 * y.size)
    start = np.zeros(3)
    _, fixed = gradient_descent(f, grad, start, step=1 / L,
                                max_iter=500_000)
    x, back = gradient_descent(f, grad, start)
    res = minimize(f, start, jac=grad)
    print(f"{name:13s} L = {L:8.3f}   fixed {fixed.size - 1}, "
          f"backtracking {back.size - 1}, BFGS {res.nit}")
    print("              coefficients", x.round(4),
          " loss", round(back[-1], 6))
