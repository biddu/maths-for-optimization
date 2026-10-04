"""Chapter 4, in practice: what convexity buys from a solver."""
import numpy as np
from scipy.optimize import minimize

from from_scratch import STORES, TRIPS, one_depot, two_depots

rng = np.random.default_rng(5)

# One depot, twenty random starting points.
runs = [minimize(one_depot, rng.uniform(0, 10, 2))
        for _ in range(20)]
costs = np.array([r.fun for r in runs])
best = runs[0].x
print("one depot :", best.round(3), costs.min(), costs.max())

# The pulls of the stores balance at the optimum.
away = STORES - best
pull = away / np.linalg.norm(away, axis=1)[:, None]
print("net pull  :", (TRIPS @ pull).round(5))

# Two depots, two hundred random starting points.
options = {"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20_000}
costs = [minimize(two_depots, rng.uniform(0, 10, 4),
                  method="Nelder-Mead", options=options).fun
         for _ in range(200)]
rounded = np.round(costs, 1)
values, counts = np.unique(rounded, return_counts=True)
for value, count in zip(values, counts):
    print(f"two depots: cost {value:5.1f}, {count:3d} runs")
