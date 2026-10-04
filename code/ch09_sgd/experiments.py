"""Chapter 9: every number quoted in the chapter."""
import time

import numpy as np
from scipy.optimize import minimize

from from_scratch import (gradient, loss, running_mean_sgd, sgd,
                          subscribers)

A_all, y_all = subscribers()
A, y = A_all[:100_000], y_all[:100_000]
T, ty = A_all[100_000:], y_all[100_000:]
n, d = A.shape
ref = minimize(loss, np.zeros(d), jac=gradient, args=(A, y),
               method="BFGS", options={"gtol": 1e-10})
x_star, f_star = ref.x, ref.fun
gap = lambda x: loss(x, A, y) - f_star
L = np.linalg.eigvalsh(A.T @ A / (4 * n)).max()
DECAY = lambda k: 0.1 / (1 + k / 1000)

if __name__ == "__main__":
    acc = lambda x, A, y: np.mean(np.sign(A @ x) == y)
    print("cancelled:", (y > 0).mean(), "f(0)", loss(np.zeros(d), A, y))
    print("f*", round(f_star, 5), "test loss", round(loss(x_star, T, ty), 5),
          "accuracy", acc(x_star, A, y), acc(x_star, T, ty),
          "BFGS passes", ref.nfev, "L", round(L, 4))
    P = 1 / (1 + np.exp(ty * (T @ x_star)))
    per = np.logaddexp(0, -ty * (T @ x_star))
    print("std error of test loss", per.std() / np.sqrt(ty.size))

    # --- the average of a stream -------------------------------
    calls = np.random.default_rng(2).normal(6, 2, 2000)
    print("mean of the 2000 calls", calls.mean().round(3))
    for name, st in (("0.1", 0.1), ("0.01", 0.01),
                     ("1/k", lambda k: 1 / (k + 1))):
        p = running_mean_sgd(calls, st)
        print(f"step {name}: x after 50, 200, 2000 calls "
              f"{p[49]:.3f} {p[199]:.3f} {p[-1]:.3f}; spread over "
              f"the last 1000: {p[1000:].std():.3f}")
    print("predicted spreads", np.sqrt(0.1 * 4 / 1.9).round(3),
          np.sqrt(0.01 * 4 / 1.99).round(3))

    # --- gradient descent --------------------------------------
    x, first = np.zeros(d), None
    for ep in range(1, 101):
        x -= gradient(x, A, y) / L
        if first is None and gap(x) <= 0.003:
            first = ep
        if ep in (1, 5, 10, 20, 50, 100):
            print("GD pass", ep, "gap", round(gap(x), 5),
                  "test loss", round(loss(x, T, ty), 4))
    print("GD passes to reach 0.003:", first)

    # --- gradient noise ------------------------------------------
    for pt, name in ((np.zeros(d), "x = 0"), (x_star, "x*")):
        p = 1 / (1 + np.exp(y * (A @ pt)))
        G = -(A * (y * p)[:, None])
        g = G.mean(axis=0)
        print(name, "sigma^2", ((G - g) ** 2).sum(axis=1).mean().round(3),
              "|full gradient|^2", (g @ g).round(4))
    sigma2 = ((G - g) ** 2).sum(axis=1).mean()

    # --- constant steps and the floor ---------------------------
    for step in (0.1, 0.02, 0.005):
        x, snaps = sgd(A, y, np.zeros(d), step, epochs=5, every=10_000)
        g = [gap(s) for s in snaps]
        print(f"step {step}: gap after 0.1, 1, 5 passes {g[0]:.4f} "
              f"{g[9]:.4f} {g[49]:.4f}; mean over last pass "
              f"{np.mean(g[-10:]):.4f}; predicted floor "
              f"{step * sigma2 / 4:.4f}")

    # --- a decreasing step ---------------------------------------
    x, snaps = sgd(A, y, np.zeros(d), DECAY, epochs=5, every=1000)
    g = np.array([gap(s) for s in snaps])
    print("decay: gap after 0.1, 0.5, 1, 2, 5 passes",
          [f"{g[i]:.5f}" for i in (9, 49, 99, 199, 499)],
          "first below 0.003 after", (np.argmax(g <= 0.003) + 1) * 1000,
          "examples; test loss", round(loss(x, T, ty), 4),
          "accuracy", acc(x, T, ty))

    # --- mini-batches ----------------------------------------------
    rng = np.random.default_rng(1)
    full = gradient(np.zeros(d), A, y)
    for b in (1, 4, 16, 64, 256, 1024):
        v = np.mean([np.sum((gradient(np.zeros(d), A[r], y[r]) - full) ** 2)
                     for r in (rng.choice(n, b, replace=False)
                               for _ in range(4000))])
        print(f"batch {b}: variance {v:.4f}, times b {v * b:.2f}")
    for b in (1, 8, 64, 512, 4096):
        sched = lambda k, b=b: min(b * DECAY(k * b), 1 / L)
        per = -(-n // b)
        every = max(1, per // 50)
        t0 = time.perf_counter()
        x, snaps = sgd(A, y, np.zeros(d), sched, batch=b, epochs=4,
                       every=every)
        secs = (time.perf_counter() - t0) / 4
        g = np.array([gap(s) for s in snaps])
        seen = (np.argmax(g <= 0.003) + 1) * every * b
        print(f"batch {b}: {per} updates and {secs:.3f} s per pass; "
              f"{seen} examples and {secs * seen / n:.4f} s to reach "
              f"0.003; gap after 4 passes {g[-1]:.5f}")

    # --- where it fails ---------------------------------------------
    order = np.argsort(y, kind="stable")
    for shuffle in (False, True):
        x, snaps = sgd(A[order], y[order], np.zeros(d), 0.02, epochs=3,
                       shuffle=shuffle)
        print("sorted data, shuffle", shuffle, "gap", round(gap(x), 4),
              "accuracy", acc(x, T, ty), "intercept", x[0].round(2),
              "true", x_star[0].round(2))
    x, snaps = sgd(A, y, np.zeros(d), 1.0, epochs=2, every=10_000)
    print("step 1.0: gaps", [round(gap(s), 2) for s in snaps[::4]])
