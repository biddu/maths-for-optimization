"""Chapter 11, in practice: a blend and a shipping plan."""
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog

# --- Blending: the cheapest feed that meets the specification ---
FEEDS = ["maize", "soya meal", "wheat bran", "fish meal"]
cost = np.array([0.25, 0.45, 0.18, 1.10])      # euro per kg
protein = np.array([9.0, 44.0, 15.0, 60.0])    # percent
fiber = np.array([2.5, 6.0, 11.0, 1.0])        # percent
fat = np.array([4.0, 1.5, 4.0, 9.0])           # percent


def blend(min_protein=20.0, max_fiber=5.0, min_fat=3.5,
          max_fish=0.05):
    """Fractions of each feed in the cheapest legal blend."""
    A_ub = np.array([-protein, fiber, -fat])   # rows are <= rules
    b_ub = np.array([-min_protein, max_fiber, -min_fat])
    limits = [(0, None)] * 3 + [(0, max_fish)]
    return linprog(cost, A_ub=A_ub, b_ub=b_ub,
                   A_eq=np.ones((1, 4)), b_eq=[1.0],
                   bounds=limits)


def diagnose(min_protein=20.0, max_fiber=5.0, min_fat=3.5,
             max_fish=0.05):
    """Smallest total relaxation of the three rules (in percent
    points) that makes the specification possible."""
    A_ub = np.array([-protein, fiber, -fat])
    A_ub = np.hstack([A_ub, -np.eye(3)])       # one give per rule
    b_ub = np.array([-min_protein, max_fiber, -min_fat])
    A_eq = np.array([[1.0, 1, 1, 1, 0, 0, 0]])
    limits = [(0, None)] * 3 + [(0, max_fish)] + [(0, None)] * 3
    res = linprog([0, 0, 0, 0, 1, 1, 1], A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=[1.0], bounds=limits)
    return res.x[:4], res.x[4:]


# --- Transportation: three warehouses, four cities --------------
SUPPLY = [120, 150, 90]
DEMAND = [80, 70, 110, 60]
RATE = [[4, 6, 9, 5],                          # euro per pallet
        [7, 3, 4, 8],
        [6, 5, 2, 7]]


def ship(supply=SUPPLY, demand=DEMAND, rate=RATE):
    """The cheapest plan, its cost, and the dual prices."""
    solver = pywraplp.Solver.CreateSolver("GLOP")
    W, C = range(len(supply)), range(len(demand))
    x = [[solver.NumVar(0, solver.infinity(), f"x{i}{j}")
          for j in C] for i in W]
    stock = [solver.Add(sum(x[i][j] for j in C) <= supply[i])
             for i in W]
    needs = [solver.Add(sum(x[i][j] for i in W) >= demand[j])
             for j in C]
    solver.Minimize(sum(rate[i][j] * x[i][j]
                        for i in W for j in C))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return status, None, None, None
    plan = [[x[i][j].solution_value() for j in C] for i in W]
    prices = ([row.dual_value() for row in stock],
              [row.dual_value() for row in needs])
    return status, plan, solver.Objective().Value(), prices


if __name__ == "__main__":
    res = blend()
    print("blend:", dict(zip(FEEDS, res.x.round(4))),
          "cost per kg", round(res.fun, 4))
    print("  prices of the rules:", res.ineqlin.marginals.round(4),
          "of the total:", res.eqlin.marginals.round(4))
    status, plan, total, prices = ship()
    print("shipping:", np.array(plan), "cost", total)
    print("  warehouse prices:", np.round(prices[0], 3),
          "city prices:", np.round(prices[1], 3))
