"""Chapter 1, in practice: the same problems handed to SciPy."""
import numpy as np
from scipy.optimize import linprog, minimize

# Version A: a linear program.
# linprog minimizes, so the objective is negated.
res = linprog(c=[-3, -2], A_ub=[[1, 1]], b_ub=[10],
              bounds=[(0, 6), (0, 8)])
print("A:", res.x, -res.fun)


# Version B: a smooth nonlinear program, again as a minimization.
def neg_rate(x):
    return -(np.log2(1 + x[0]) + np.log2(1 + x[1] / 4))


power = {"type": "ineq", "fun": lambda x: 10 - x[0] - x[1]}
res = minimize(neg_rate, x0=[1.0, 1.0],
               bounds=[(0, 10), (0, 10)], constraints=[power])
print("B:", res.x, -res.fun, res.success)


# Where it fails: one function, two starting points, two answers.
def f(x):
    return x[0] ** 4 - 3 * x[0] ** 2 + x[0]


for start in (1.0, -1.0):
    res = minimize(f, x0=[start])
    print(f"start {start:+.1f}: x = {res.x[0]:+.4f}, "
          f"f = {res.fun:+.4f}, success = {res.success}")
