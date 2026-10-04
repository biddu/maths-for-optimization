"""Chapter 13: checks for the exercise solutions."""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from from_scratch import (anneal, nearest_neighbor, nelder_mead,
                          tour_length, two_opt)
from in_practice import cities

# Exercise 13.1: one step of Nelder-Mead.
f = lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2
for p in ((0, 0), (1, 0), (0, 1), (1, 1), (1.5, 1.5)):
    print("Ex 1: f", p, "=", f(p))

# Exercise 13.2: four stops on a square.
P = np.array([[0.0, 0], [1, 1], [1, 0], [0, 1]])
D = np.linalg.norm(P[:, None] - P[None], axis=2)
print("Ex 2:", tour_length(np.array([0, 1, 2, 3]), D),
      tour_length(np.array([0, 2, 1, 3]), D))

# Exercise 13.3: acceptance probabilities.
print("Ex 3:", [math.exp(-5 / T) for T in (10, 1, 0.1)])

# Exercise 13.4: counting tours.
print("Ex 4:", math.factorial(9) // 2, f"{math.factorial(39) / 2:.2e}")

# Exercise 13.6: luck.
print("Ex 6:", 1 - 0.9 ** 30, 1 - 0.99 ** 30,
      math.log(0.05) / math.log(0.99))

# Exercise 13.8: many starts.
P, D = cities(40, 13)
ends = [tour_length(two_opt(np.random.default_rng(s).permutation(40), D),
                    D) for s in range(20)]
print("Ex 8: best", min(ends), "mean", np.mean(ends), "worst", max(ends),
      "within 1% of 550.4:", sum(e < 555.9 for e in ends))

# Exercise 13.9: how slowly to cool.
nn = nearest_neighbor(D)
for steps in (2_000, 20_000, 200_000):
    vals = [tour_length(anneal(nn, D, steps=steps, seed=s), D)
            for s in range(5)]
    print("Ex 9:", steps, "steps: mean", np.mean(vals))

# Exercise 13.10: Nelder-Mead and the number of variables.
for n in (2, 5, 10, 20, 40):
    x, value, evals = nelder_mead(lambda x: x @ x, np.ones(n), tol=1e-12,
                                  max_evals=50_000)
    print("Ex 10: n =", n, "evaluations", evals, "final value", value)
