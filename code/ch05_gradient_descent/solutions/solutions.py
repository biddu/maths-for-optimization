"""Chapter 5: numerical checks for the exercise solutions."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import (churn_data, gradient_descent, logistic_grad,
                          logistic_loss)

# Exercise 5.1.
x = np.array([2.0, 1.0])
for _ in range(2):
    x = x - 0.25 * np.array([2 * x[0], 4 * x[1]])
    print("Ex 1:", x, x[0] ** 2 + 2 * x[1] ** 2)

# Exercise 5.4: the five cars, raw and centered ages.
ages = np.arange(1.0, 6.0)
for label, col in (("raw", ages), ("centered", ages - 3)):
    A = np.column_stack([np.ones(5), col])
    lam = np.linalg.eigvalsh(A.T @ A)
    print("Ex 4:", label, lam, lam[-1] / lam[0])

# Exercise 5.6.
print("Ex 6:", 50 * np.log(1e4), (1 - 1 / 50) ** 461)


# Exercise 5.8: the Rosenbrock function.
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


x, hist = gradient_descent(rosen, rosen_grad, [-1.2, 1.0])
print("Ex 8:", x, hist.size - 1, hist[-1])
H = np.array([[802.0, -400.0], [-400.0, 200.0]])
lam = np.linalg.eigvalsh(H)
print("Ex 8: Hessian at (1, 1):", lam, lam[-1] / lam[0])

# Exercise 5.9: months or years?
hours, months, y = churn_data()
ones = np.ones(y.size)
for label, col in (("months", months), ("years", months / 12)):
    A = np.column_stack([ones, hours, col])
    f = lambda x: logistic_loss(x, A, y)
    grad = lambda x: logistic_grad(x, A, y)
    x, hist = gradient_descent(f, grad, np.zeros(3))
    lam = np.linalg.eigvalsh(A.T @ A)
    print("Ex 9:", label, hist.size - 1, x.round(4),
          round(lam[-1] / lam[0]))
