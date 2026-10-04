"""Chapter 8, in practice: convex programs in CVXPY."""
import cvxpy as cp
import numpy as np

from data import STORES, TRIPS

# A linear program: Version A of Chapter 1, with its prices.
x = cp.Variable(2)
limits = [x[0] + x[1] <= 10, x[0] <= 6, x[1] <= 8, x >= 0]
lp = cp.Problem(cp.Maximize(3 * x[0] + 2 * x[1]), limits)
lp.solve()
print("LP:", x.value.round(4), round(lp.value, 4),
      [round(float(c.dual_value), 4) for c in limits[:3]])

# A second-order cone program: the depot of Chapter 4.
d = cp.Variable(2)
legs = [cp.norm(d - s) for s in STORES]
socp = cp.Problem(cp.Minimize(TRIPS @ cp.hstack(legs)))
socp.solve()
print("depot:", d.value.round(2), round(socp.value, 2))

# The five channels of Chapter 7, with the price of power.
g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
p = cp.Variable(5, nonneg=True)
budget = cp.sum(p) <= 10
rate = cp.sum(cp.log1p(cp.multiply(g, p))) / np.log(2)
chan = cp.Problem(cp.Maximize(rate), [budget])
chan.solve()
print("channels:", p.value.round(3), round(chan.value, 4),
      round(float(budget.dual_value), 4))

# The rule book: what CVXPY accepts and what it refuses.
t = cp.Variable()
bad = cp.log(1 + cp.exp(t))
good = cp.logistic(t)
print("log(1 + exp(t)) accepted:", bad.is_dcp())
print("logistic(t) accepted:    ", good.is_dcp())
