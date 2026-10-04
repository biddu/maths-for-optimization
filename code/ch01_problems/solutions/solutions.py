"""Chapter 1: numerical checks for the exercise solutions."""
import sys, pathlib
import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import grid_search, knapsack

# Exercise 1: the workshop.
res = linprog(c=[-70, -50], A_ub=[[4, 3], [2, 1]], b_ub=[240, 100],
              bounds=[(0, None), (0, None)])
print("Ex 1:", res.x, -res.fun)

# Exercise 2.
res = linprog(c=[-3, 1], A_ub=[[-1, -1], [1, 0]], b_ub=[-2, 4],
              A_eq=[[1, -2]], b_eq=[0], bounds=[(None, None)] * 2)
print("Ex 2:", res.x, -res.fun)

# Exercise 4.
res = linprog(c=[-5, -4], A_ub=[[1, 1]], b_ub=[8],
              bounds=[(0, 5), (0, None)])
print("Ex 4:", res.x, -res.fun)


# Exercise 7: three channels.
def three(x):
    return (np.log2(1 + x[0]) + np.log2(1 + x[1] / 4)
            + np.log2(1 + x[2] / 2))


for h in (1.0, 0.5):
    x, fx, n = grid_search(three, lambda x: x.sum() <= 10 + 1e-9,
                           [0, 0, 0], [10, 10, 10], h)
    print(f"Ex 7: h={h} x={x} f={fx:.4f} points={n}")
res = minimize(lambda x: -three(x), x0=[1, 1, 1], bounds=[(0, 10)] * 3,
               constraints=[{"type": "ineq",
                             "fun": lambda x: 10 - x.sum()}])
print("Ex 7 scipy:", res.x, -res.fun, " exact:", np.log2(0.125 * (17 / 3) ** 3))


# Exercise 8: multi-start.
def f(x):
    return x[0] ** 4 - 3 * x[0] ** 2 + x[0]


ends = [minimize(f, x0=[s]).x[0] for s in np.linspace(-2, 2, 21)]
print("Ex 8: left", sum(e < 0 for e in ends),
      "right", sum(e > 0 for e in ends),
      [round(float(s),1) for s,e in zip(np.linspace(-2,2,21),ends) if e<0])

# Exercise 9: the knapsack at three budgets.
for b in (9, 10, 11):
    print("Ex 9:", b, knapsack([6, 5, 5, 3], [30, 24, 23, 9], b))
