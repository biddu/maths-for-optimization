"""Chapter 2: numerical checks for the exercise solutions."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import (grad_central, grad_rate, gradient_check,
                          hess_rate)


# Exercises 2.1 and 2.2.
def f1(x):
    return x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2 - 4 * x[0]


x = np.array([1.0, 1.0])
g = grad_central(f1, x)
print("Ex 1: gradient", g.round(6),
      "eigenvalues", np.linalg.eigvalsh([[2, 3], [3, 4]]).round(4))
print("Ex 2:", g @ np.array([3, 4]) / 5, np.linalg.norm(g))


# Exercise 2.3: three channels, equal marginal rates.
def rate3(x):
    return (np.log2(1 + x[0]) + np.log2(1 + x[1] / 4)
            + np.log2(1 + x[2] / 2))


print("Ex 3:", grad_central(rate3, np.array([14, 5, 11]) / 3).round(4),
      grad_central(rate3, np.array([5.0, 2.0, 3.0])).round(4))

# Exercise 2.4: Taylor approximations of x1 * exp(x2) at (1, 0).
print("Ex 4:", 1.1 * np.exp(0.2))

# Exercise 2.5: least squares.
A = np.array([[1.0, 0], [1, 1], [1, 2]])
b = np.array([1.0, 2, 2])
print("Ex 5:", A.T @ A, A.T @ b, np.linalg.solve(A.T @ A, A.T @ b))

# Exercise 2.6.
for M in ([[4, 1], [1, 4]], [[1, 3], [3, 1]], [[2, -2], [-2, 2]],
          [[-3, 0], [0, -1]]):
    print("Ex 6:", M, np.linalg.eigvalsh(M).round(6))


# Exercise 2.8: Hessian by differences of the gradient.
def hessian_central(grad, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    H = np.zeros((x.size, x.size))
    for j in range(x.size):
        step = np.zeros(x.size)
        step[j] = h
        H[:, j] = (grad(x + step) - grad(x - step)) / (2 * h)
    return (H + H.T) / 2


x = np.array([4.0, 6.0])
H = hessian_central(grad_rate, x)
print("Ex 8:", H, np.abs(H - hess_rate(x)).max())

# Exercise 2.9: forward-difference error for sin at x = 1.
for k in range(1, 16):
    h = 10.0 ** -k
    err = abs((np.sin(1 + h) - np.sin(1)) / h - np.cos(1))
    print(f"Ex 9: h=1e-{k:02d} error={err:.1e}")

# Exercise 2.10.
v = np.array([3.0, -4.0, 12.0])
print("Ex 10:", np.abs(v).sum(), np.linalg.norm(v), np.abs(v).max())
