"""Chapter 10: checks for the exercise solutions."""
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments import bowl, raw_churn, steps_to
from from_scratch import adam, momentum
from in_practice import OPTIMIZERS, evaluate, network, train

# Exercise 10.1: three steps of momentum by hand.
path = momentum(lambda x: x, [1.0], 0.1, 0.5, 3)
print("Ex 1:", path.ravel())

# Exercise 10.3: kappa = 400.
f, grad = bowl(400)
r = 20.0
beta = ((r - 1) / (r + 1)) ** 2
goal = f(np.ones(2)) / 1e6
gd = momentum(grad, [1, 1], 1 / 400, 0.0, 20_000)
hb = momentum(grad, [1, 1], 4 / (r + 1) ** 2, beta, 2000)
print("Ex 3: beta", round(beta, 3), "GD", steps_to(gd, f, goal),
      "momentum", steps_to(hb, f, goal))

# Exercises 10.4 and 10.5: Adam's first step.
f, grad, best, L = raw_churn()
g = grad(np.zeros(3))
print("Ex 4: gradient", g.round(3), "first step",
      adam(grad, np.zeros(3), 0.03, 1)[1])
print("Ex 5: uncorrected ratio", 0.1 / math.sqrt(0.001))

# Exercise 10.6: the loss multiplied by 100.
big = lambda x: 100 * grad(x)
a1 = adam(grad, np.zeros(3), 0.03, 200)[-1]
a2 = adam(big, np.zeros(3), 0.03, 200)[-1]
print("Ex 6: largest difference after 200 steps",
      np.abs(a1 - a2).max())

# Exercise 10.7: one tanh unit.
w, a, y = 0.5, 2.0, 0.0
h = math.tanh(w * a)
print("Ex 7: h", round(h, 4), "f", round((h - y) ** 2, 4),
      "df/dw", round(2 * (h - y) * (1 - h ** 2) * a, 4))

# Exercise 10.8: every weight zero.
model = network()
for p in model.parameters():
    torch.nn.init.zeros_(p)
h = train(model, torch.optim.Adam(model.parameters(), lr=0.01))
moved = [float(p.detach().abs().max()) for p in model.parameters()]
print("Ex 8: loss", round(h[-1], 4), "largest entry per tensor",
      [round(m, 4) for m in moved])

# Exercise 10.9: two step sizes, two optimizers.
for name in ("SGD", "Adam"):
    for t in (0.001, 0.3):
        model = network()
        h = train(model, OPTIMIZERS[name](model.parameters(), t))
        print("Ex 9:", name, t, round(h[-1], 3),
              round(evaluate(model)[1], 3))

# Exercise 10.10: Adam's bursts, and a decaying step.
path = adam(grad, np.zeros(3), 0.03, 20_000)
gaps = np.array([f(p) - best for p in path])
print("Ex 10: constant step: gap at step 1000", gaps[1000],
      "largest gap after step 1000", gaps[1000:].max(),
      "at step", 1000 + int(gaps[1000:].argmax()))


def adam_decay(grad, x0, step, steps, b1=0.9, b2=0.999, eps=1e-8):
    x = np.array(x0, dtype=float)
    m, s = np.zeros_like(x), np.zeros_like(x)
    out = [x.copy()]
    for k in range(1, steps + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        t = step / math.sqrt(1 + k / 100)
        x = x - t * (m / (1 - b1 ** k)) / (np.sqrt(s / (1 - b2 ** k)) + eps)
        out.append(x.copy())
    return np.array(out)


path = adam_decay(grad, np.zeros(3), 0.03, 20_000)
gaps = np.array([f(p) - best for p in path])
print("Ex 10: decaying step: first below 1e-6 at",
      int(np.argmax(gaps <= 1e-6)), "largest gap after step 2000",
      gaps[2000:].max())
