"""Chapter 14: checks for the exercise solutions."""
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments import CARS, reliability, workshop
from from_scratch import (G, LN2, at_price, link_design, pareto, waterfill,
                          weighted_sum, with_limit)

# Exercise 14.1: six laptops, price ($100) and weight (kg).
laptops = np.array([[9, 2.4], [12, 1.6], [11, 2.0], [15, 1.2],
                    [14, 1.7], [18, 1.1]])
print("Ex 1: not dominated:", [k + 1 for k in pareto(laptops)])

# Exercise 14.2: the cars with a year worth 1.5 thousand.
print("Ex 2: totals", (CARS @ [1, 1.5]).round(2), "car",
      weighted_sum(CARS, [1, 1.5]) + 1)
print("Ex 2: car 4 wins for a year worth between",
      (12.5 - 9) / (7 - 4), "and", (16 - 12.5) / (4 - 2))

# Exercise 14.3: predicting along the front.
x5, r5 = waterfill(G, 5.0)
lam = 1 / ((x5 + 1 / G)[x5 > 0].max() * LN2)
print("Ex 3: rate at 5 W", r5, "price", lam, "prediction for 6 W",
      r5 + lam, "actual", waterfill(G, 6.0)[1])

# Exercise 14.4: the point for a price of 1.
x, rate = at_price(G, 1.0)
print("Ex 4: level", 1 / LN2, "powers", x.round(3), "total", x.sum(),
      "rate", rate)

# Exercise 14.5: three options.
pts = np.array([[1.0, 6.0], [3.0, 5.0], [6.0, 1.0]])
found = {weighted_sum(pts, [w, 1 - w]) for w in np.linspace(0, 1, 1001)}
print("Ex 5: not dominated", pareto(pts), "found by weights", found)

# Exercise 14.6: closest to the ideal car.
ideal = CARS.min(axis=0)
raw = np.linalg.norm(CARS - ideal, axis=1)
scaled = (CARS - ideal) / (CARS.max(axis=0) - ideal)
print("Ex 6: ideal", ideal, "raw distances", raw.round(2), "car",
      raw.argmin() + 1)
print("Ex 6: scaled distances", np.linalg.norm(scaled, axis=1).round(3),
      "car", np.linalg.norm(scaled, axis=1).argmin() + 1)
months = CARS * [1, 12]
print("Ex 6: age in months: car",
      np.linalg.norm(months - months.min(axis=0), axis=1).argmin() + 1)

# Exercise 14.7: the margin for 90 percent.
for margin in (0.07, 0.08, 0.082, 0.085, 0.09):
    plan, profit = workshop(margin)
    print("Ex 7: margin", margin, "reliability",
          round(reliability(plan), 3), "profit", round(profit))

# Exercise 14.8: the on-off channels of Chapter 1.
cost = np.array([6.0, 5.0, 5.0, 3.0])
gain = np.array([30.0, 24.0, 23.0, 9.0])
plans = [s for r in range(5) for s in itertools.combinations(range(4), r)]
table = np.array([[cost[list(s)].sum(), -gain[list(s)].sum()]
                  for s in plans])
keep = pareto(table)
hull = sorted({weighted_sum(table, [w, 1]) for w in np.linspace(0, 20, 20001)})
for k in keep:
    print("Ex 8:", [j + 1 for j in plans[k]], "power", table[k, 0],
          "rate", -table[k, 1], "found by weights" if k in hull else "")

# Exercise 14.9: many objectives.
rng = np.random.default_rng(0)
for d in (2, 4, 6, 8, 10):
    print("Ex 9:", d, len(pareto(rng.uniform(size=(1000, d)))))

# Exercise 14.10: a hungrier circuit.
for circuit in (0.5, 1.0, 2.0):
    rate, active, x = link_design(G, 10.0, circuit)
    print("Ex 10: circuit", circuit, "rate", round(rate, 4), "channels",
          [k + 1 for k in active], "powers", x.round(3))
