"""Chapter 11: checks for the exercise solutions."""
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments import A, b, klee_minty
from from_scratch import corners, simplex
from in_practice import DEMAND, RATE, ship

# Exercise 11.1: the same problem, two ways.
res = linprog([2, 3], A_ub=[[-1, -1]], b_ub=[-2], A_eq=[[1, -1]],
              b_eq=[1], bounds=[(0, None), (None, None)])
print("Ex 1: as stated", res.x, res.fun)
res = linprog([2, 3, -3], A_ub=[[-1, -1, 1], [1, -1, 1], [-1, 1, -1]],
              b_ub=[-2, 1, -1])
print("Ex 1: standard form", res.x, res.fun)

# Exercises 11.2 to 11.4: a second small problem.
A2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
b2 = np.array([8.0, 6.0, 16.0])
c2 = np.array([2.0, 3.0])
print("Ex 2:", [(tuple(v), float(c2 @ v)) for v in corners(A2, b2)])
x, value, prices, path = simplex(c2, A2, b2)
print("Ex 3:", [tuple(p) for p in path], value, "prices", prices,
      "dual value", prices @ b2)

# Exercise 11.5: the profit on desks.
for p in (55.0, 65.0):
    x, value, _, _ = simplex(np.array([p, 30.0]), A, b)
    print("Ex 5: desk profit", p, x, value)

# Exercise 11.6: a chair and a stool.
for name, col, p in (("chair", [1, 1, 0], 25.0),
                     ("stool", [0.5, 0.5, 0], 18.0)):
    x, value, _, _ = simplex(np.array([40.0, 30.0, p]),
                             np.column_stack([A, col]), b)
    print("Ex 6:", name, x, value)

# Exercise 11.8: corners against pivots.
rng = np.random.default_rng(3)
M = rng.uniform(0, 1, (8, 8))
rhs = rng.uniform(1, 2, 8) * 4
cost = rng.uniform(0, 1, 8)
found = corners(M, rhs)
x, value, _, path = simplex(cost, M, rhs)
print("Ex 8: systems tried", math.comb(16, 8), "corners", len(found),
      "best by enumeration", (found @ cost).max().round(6),
      "simplex", value.round(6), "pivots", len(path) - 1)

# Exercise 11.9: the twisted cube.
cc, Mk, bk = klee_minty(8)
print("Ex 9:", len(simplex(cc, Mk, bk)[3]) - 1,
      len(simplex(cc, Mk, bk, rule="bland")[3]) - 1)
res = linprog(-cc, A_ub=Mk, b_ub=bk)
print("Ex 9: HiGHS iterations", res.nit, "value", -res.fun)

# Exercise 11.10: shipping with changes.
print("Ex 10a:", ship(demand=[80, 70, 110, 70])[2])
supply = [120, 100, 90]
cost = np.array(RATE, dtype=float).ravel()
c_all = np.concatenate([cost, 20 * np.ones(4)])    # 4 shortfalls
A_ub = np.zeros((7, 16))
for i in range(3):
    A_ub[i, 4 * i:4 * i + 4] = 1                   # supply rows
for j in range(4):
    A_ub[3 + j, j:12:4] = -1                       # demand rows
    A_ub[3 + j, 12 + j] = -1
b_ub = np.concatenate([supply, -np.array(DEMAND, dtype=float)])
res = linprog(c_all, A_ub=A_ub, b_ub=b_ub)
print("Ex 10b: plan", res.x[:12].reshape(3, 4), "short", res.x[12:],
      "shipping cost", cost @ res.x[:12])
