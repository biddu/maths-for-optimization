"""Chapter 7: the tables quoted in the text."""
import numpy as np
from scipy.optimize import linprog

from from_scratch import (penalty_method, project_budget,
                          projected_gradient, rate, waterfill)

# 1. The can.
V = 330.0
r = (V / (2 * np.pi)) ** (1 / 3)
best_area = lambda v: 6 * np.pi * (v / (2 * np.pi)) ** (2 / 3)
print(f"can: r = {r:.4f}, h = {2 * r:.4f}, area = {best_area(V):.3f},"
      f" nu = {-2 / r:.4f}")
print("     one more cubic centimeter costs",
      round(best_area(V + 1) - best_area(V), 4), "; predicted", round(2 / r, 4))

# 2. Five channels.
g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
x, level = waterfill(g, 10.0)
lam = 1 / (level * np.log(2))
print("water-filling:", x, " level", level, " rate", round(rate(x, g), 4))
print("  lambda =", round(lam, 4), " mu_5 =", round(lam - g[4] / np.log(2), 4))
for extra in (0.1, 1.0):
    gain = rate(waterfill(g, 10.0 + extra)[0], g) - rate(x, g)
    print(f"  {extra} more watts: gain {gain:.4f}, predicted {lam * extra:.4f}")

# 3. Version A of Chapter 1: what is each limit worth?
for total, cap in ((10, 6), (11, 6), (10, 7)):
    res = linprog([-3, -2], A_ub=[[1, 1]], b_ub=[total],
                  bounds=[(0, cap), (0, 8)])
    print(f"LP: total {total}, cap {cap}: x = {res.x}, rate = {-res.fun}")

# 4. The penalty method on the can.
f = lambda z: 2 * np.pi * z[0] ** 2 + 2 * np.pi * z[0] * z[1]
h = lambda z: np.pi * z[0] ** 2 * z[1] - V
_, rows = penalty_method(f, h, [3.0, 10.0], [0.01, 0.1, 1, 10, 100, 1000])
for rho, z, viol, mult, steps in rows:
    print(f"rho {rho:7.2f}: r = {z[0]:.4f}, h = {z[1]:.4f}, "
          f"violation {viol:9.4f}, rho*h = {mult:.4f}, steps {steps}")

# 5. Projected gradient on the five channels.
grad = lambda x: -g / ((1 + g * x) * np.log(2))
L = (g ** 2).max() / np.log(2)
xp, steps = projected_gradient(grad, lambda v: project_budget(v, 10.0),
                               np.full(5, 2.0), 1 / L)
print("projected gradient:", xp.round(5), steps, "steps")
