"""Chapter 9: checks for the exercise solutions."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments import A, d, gap, y
from from_scratch import running_mean_sgd, sgd

# Exercise 9.1: three data points.
a = np.array([1.0, 2.0, 6.0])
g = 0.0 - a
print("Ex 1: gradients", g, "mean", g.mean(), "variance", g.var())

# Exercise 9.2: step 1/k gives the running mean.
calls = np.random.default_rng(2).normal(6, 2, 2000)
path = running_mean_sgd(calls, lambda k: 1 / (k + 1))
means = np.cumsum(calls) / np.arange(1, 2001)
print("Ex 2: largest difference", np.abs(path - means).max())

# Exercise 9.3: the floor and the forgetting time.
for t in (0.1, 0.01):
    k = int(np.ceil(np.log(0.01) / np.log(1 - t)))
    print("Ex 3: step", t, "spread", np.sqrt(t * 4 / (2 - t)).round(3),
          "steps to forget 99%", k)
rng = np.random.default_rng(0)
ends = [running_mean_sgd(rng.normal(6, 2, 400), 0.1)[-1]
        for _ in range(20_000)]
print("Ex 3: simulated spread", np.std(ends).round(3))

# Exercise 9.8: three constant steps.
sigma2 = 1.891
for step in (0.1, 0.02, 0.005):
    _, snaps = sgd(A, y, np.zeros(d), step, epochs=5, every=10_000)
    print("Ex 8: step", step, "floor",
          np.mean([gap(s) for s in snaps[-10:]]).round(4),
          "predicted", round(step * sigma2 / 4, 4))

# Exercise 9.9: averaging the iterates.
x, snaps = sgd(A, y, np.zeros(d), 0.02, epochs=3, every=100)
print("Ex 9: last iterate", gap(x).round(5), "average of last pass",
      gap(np.mean(snaps[-1000:], axis=0)).round(5))

# Exercise 9.10: sorted data.
order = np.argsort(y, kind="stable")
for shuffle in (False, True):
    x, _ = sgd(A[order], y[order], np.zeros(d), 0.02, epochs=3,
               shuffle=shuffle)
    print("Ex 10: shuffle", shuffle, "gap", gap(x).round(4),
          "predicts cancel for", np.mean(A @ x > 0).round(3))
