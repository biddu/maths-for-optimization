"""Chapter 3, in practice: NumPy and SciPy on the same tasks."""
import numpy as np
from scipy.optimize import minimize

from from_scratch import (classify, grad_two_valleys,
                          hess_two_valleys, normal_equations,
                          standardize, two_valleys, used_cars)

# Used cars: price against age and distance driven.
age, km, price = used_cars()
ones = np.ones(age.size)
A = np.column_stack([ones, age, km])
x, *_ = np.linalg.lstsq(A, price, rcond=None)
print("lstsq:  ", x)
print("normal: ", normal_equations(A, price))
print("cond(A) =", np.linalg.cond(A))
print("cond(A^T A) =", np.linalg.cond(A.T @ A))

# The same fit after standardizing the two columns.
Z = np.column_stack([ones, standardize(age), standardize(km)])
z, *_ = np.linalg.lstsq(Z, price, rcond=None)
print("standardized:", z, " cond(Z) =", np.linalg.cond(Z))

# A solver that stops at a saddle point.
for start in ([0.0, 1.0], [0.01, 1.0]):
    res = minimize(two_valleys, start, jac=grad_two_valleys)
    kind = classify(hess_two_valleys(res.x))
    print(start, "->", res.x.round(6), res.success, kind)
