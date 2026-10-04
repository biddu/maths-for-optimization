"""Chapter 3: numerical checks for the exercise solutions."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import (classify, normal_equations, standardize,
                          used_cars)

# Exercise 3.1.
H = np.array([[2.0, 1.0], [1.0, 2.0]])
x = np.linalg.solve(H, [3.0, 0.0])
f = x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 3 * x[0]
print("Ex 1:", x, f, classify(H))

# Exercise 3.2.
for p in ([0.0, 0.0], [1.0, 1.0]):
    H = np.array([[6 * p[0], -3.0], [-3.0, 6 * p[1]]])
    print("Ex 2:", p, np.linalg.eigvalsh(H), classify(H))

# Exercise 3.4.
t = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 4.0, 4.0])
A = np.column_stack([np.ones(4), t])
x = normal_equations(A, y)
print("Ex 4:", A.T @ A, A.T @ y, x, y - A @ x, A.T @ (y - A @ x))

# Exercise 3.6.
b = np.array([2.0, 3.0, 7.0, 8.0])
print("Ex 6:", b.mean(), ((b - b.mean()) ** 2).sum())

# Exercise 3.9: units and the condition number.
age, km, price = used_cars()
ones = np.ones(age.size)
for name, col in (("years", age), ("days", 365 * age),
                  ("standardized", standardize(age))):
    A = np.column_stack([ones, col])
    x = normal_equations(A, price)
    print(f"Ex 9: {name:13s} cond(A) = {np.linalg.cond(A):9.1f}  "
          f"cond(A^T A) = {np.linalg.cond(A.T @ A):12.1f}  x = {x}")

# Exercise 3.10: ridge regression.
Z = np.column_stack([standardize(age), standardize(km)])
yc = price - price.mean()
lam = np.linalg.eigvalsh(Z.T @ Z)
for mu in (0.0, 10.0):
    x = np.linalg.solve(Z.T @ Z + mu * np.eye(2), Z.T @ yc)
    print(f"Ex 10: mu={mu:4.1f}  x = {x}  "
          f"cond = {(lam[-1] + mu) / (lam[0] + mu):.1f}")
print("Ex 10: eigenvalues", lam)
