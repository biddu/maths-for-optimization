"""Chapter 13: every number quoted in the chapter."""
import importlib.util
import pathlib
import time

import numpy as np
from scipy.optimize import (differential_evolution, dual_annealing,
                            minimize)

from from_scratch import (anneal, bayes_search, genetic, nearest_neighbor,
                          nelder_mead, reverse_gain, tour_length, two_opt)
from in_practice import (BOX_HIGH, BOX_LOW, cities, exact_tour, route,
                         save_cache, validation_loss)

HERE = pathlib.Path(__file__).resolve().parent
rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def branin(x):
    """A standard two-variable test function; minimum 0.3979."""
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return ((x[1] - b * x[0] ** 2 + c * x[0] - 6) ** 2
            + 10 * (1 - t) * np.cos(x[0]) + 10)


def clipped(f):
    """Keep a search inside the box by clipping its points."""
    return lambda x: f(np.clip(x, BOX_LOW, BOX_HIGH))


def searches(seed):
    """Thirty evaluations each, four ways. Returns the points."""
    rng = np.random.default_rng(seed)
    low, high = np.array(BOX_LOW), np.array(BOX_HIGH)
    out = {}
    X = rng.uniform(low, high, (30, 2))
    out["random"] = (X, np.array([validation_loss(x) for x in X]))
    trail = []
    f = lambda x: trail.append(np.clip(x, low, high)) or \
        validation_loss(np.clip(x, low, high))
    nelder_mead(f, rng.uniform(low, high), size=1.0, tol=0, max_evals=30)
    T = np.array(trail[:30])
    out["Nelder-Mead"] = (T, np.array([validation_loss(x) for x in T]))
    out["Bayesian"] = bayes_search(validation_loss, low, high, evals=30,
                                   seed=seed)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # --- Nelder-Mead on a curved valley ----------------------------
    x, value, evals = nelder_mead(rosen, [-1.2, 1.0])
    print("Nelder-Mead:", x, value, evals, "evaluations")
    res = minimize(rosen, [-1.2, 1.0], method="BFGS")
    print("BFGS with differences:", res.x, res.fun, res.nfev)
    rng = np.random.default_rng(0)
    best = np.inf
    for k in range(157):
        best = min(best, rosen(rng.uniform(-2, 2, 2)))
    print("157 random points in [-2, 2]^2: best", best)
    for sd in (1e-6, 1e-4, 1e-2):
        rng = np.random.default_rng(0)
        noisy = lambda x: rosen(x) + sd * rng.normal()
        x, _, _ = nelder_mead(noisy, [-1.2, 1.0], tol=0, max_evals=600)
        res = minimize(noisy, [-1.2, 1.0], method="BFGS")
        print(f"noise {sd:g}: Nelder-Mead ends at f = {rosen(x):.2e}, "
              f"BFGS at f = {rosen(res.x):.2f} after {res.nfev}")

    # --- forty stops --------------------------------------------------
    P, D = cities(40, 13)
    rng = np.random.default_rng(1)
    rand = [tour_length(rng.permutation(40), D) for _ in range(1000)]
    print("random tours: mean", np.mean(rand), "best of 1000", min(rand))
    nn = nearest_neighbor(D)
    t0 = time.perf_counter(); local = two_opt(nn, D)
    t1 = time.perf_counter()
    print("nearest neighbor", tour_length(nn, D), "then 2-opt",
          tour_length(local, D), f"{t1 - t0:.3f} s")
    left = sum(reverse_gain(local, D, i, j) < -1e-12
               for i in range(1, 39) for j in range(i + 1, 40))
    print("improving reversals left:", left)
    starts = [tour_length(two_opt(np.random.default_rng(s).permutation(40),
                                  D), D) for s in range(20)]
    print("2-opt from 20 random tours: best", min(starts), "mean",
          np.mean(starts), "worst", max(starts))
    t0 = time.perf_counter()
    sa = [tour_length(anneal(nn, D, seed=s), D) for s in range(10)]
    t1 = time.perf_counter()
    ga = [tour_length(genetic(D, seed=s), D) for s in range(10)]
    t2 = time.perf_counter()
    print(f"annealing, 10 seeds: best {min(sa):.1f} mean {np.mean(sa):.1f}"
          f" worst {max(sa):.1f}, {(t1 - t0) / 10:.1f} s each")
    print(f"genetic, 10 seeds: best {min(ga):.1f} mean {np.mean(ga):.1f}"
          f" worst {max(ga):.1f}, {(t2 - t1) / 10:.1f} s each")
    print("exact:", exact_tour(D))
    print("routing library:", tour_length(route(D, 5), D))
    for hot in (0.01, 1.0, None, 500.0):
        vals = [tour_length(anneal(nn, D, hot=hot, seed=s), D)
                for s in range(5)]
        print("annealing, starting temperature", hot or D.mean() / 2,
              "mean", np.mean(vals))
    for steps in (2_000, 20_000, 200_000):
        vals = [tour_length(anneal(nn, D, steps=steps, seed=s), D)
                for s in range(5)]
        print("annealing,", steps, "steps: mean", np.mean(vals))

    # --- two hundred stops -----------------------------------------
    P, D = cities(200, 14)
    nn = nearest_neighbor(D)
    for name, fn in (("2-opt", lambda: two_opt(nn, D)),
                     ("annealing, 200,000 steps", lambda: anneal(nn, D)),
                     ("annealing, 2,000,000 steps",
                      lambda: anneal(nn, D, steps=2_000_000)),
                     ("genetic", lambda: genetic(D, generations=1000)),
                     ("routing library, 10 s", lambda: route(D, 10))):
        t0 = time.perf_counter()
        tour = fn()
        print(f"200 stops, {name}: {tour_length(tour, D):.1f} in "
              f"{time.perf_counter() - t0:.1f} s")
    print("200 stops, nearest neighbor", tour_length(nn, D))
    print("200 stops, exact method, 60 s:", exact_tour(D, 60))

    # --- a standard test function ----------------------------------
    low, high = np.array([-5.0, 0.0]), np.array([10.0, 15.0])
    grid = [branin([a, b]) for a in np.linspace(-5, 10, 5)
            for b in np.linspace(0, 15, 6)]
    print("test function: minimum", branin([np.pi, 2.275]),
          "grid of 30:", min(grid))
    for budget in (30, 60):
        rows = {"random": [], "Nelder-Mead": [], "Bayesian": []}
        for seed in range(10):
            rng = np.random.default_rng(seed)
            rows["random"].append(min(
                branin(x) for x in rng.uniform(low, high, (budget, 2))))
            x0 = np.random.default_rng(100 + seed).uniform(low, high)
            rows["Nelder-Mead"].append(nelder_mead(
                branin, x0, size=2.0, tol=0, max_evals=budget)[1])
            rows["Bayesian"].append(bayes_search(
                branin, low, high, evals=budget, seed=seed)[1].min())
        print(budget, "evaluations:", {k: (round(np.mean(v), 3),
                                           round(max(v), 3))
                                       for k, v in rows.items()})

    # --- tuning two settings -----------------------------------------
    steps = np.linspace(BOX_LOW[0], BOX_HIGH[0], 15)
    decays = np.linspace(BOX_LOW[1], BOX_HIGH[1], 17)
    t0 = time.perf_counter()
    fine = np.array([[validation_loss((a, b)) for b in decays]
                     for a in steps])
    print("fine grid: 255 trainings,", round(time.perf_counter() - t0),
          "s; best", fine.min(), "at",
          steps[fine.argmin() // 17], decays[fine.argmin() % 17])
    grid = [(a, b) for a in np.linspace(BOX_LOW[0], BOX_HIGH[0], 5)
            for b in np.linspace(BOX_LOW[1], BOX_HIGH[1], 6)]
    gv = [validation_loss(g) for g in grid]
    print("grid of 30: best", min(gv), "at", grid[int(np.argmin(gv))])
    table = {"random": [], "Nelder-Mead": [], "Bayesian": []}
    for seed in range(5):
        for name, (X, y) in searches(seed).items():
            table[name].append((y.min(), y[:15].min()))
    for name, rows in table.items():
        full = [r[0] for r in rows]
        half = [r[1] for r in rows]
        print(f"{name}: best after 30 {np.round(full, 3)} mean "
              f"{np.mean(full):.3f}; after 15 mean {np.mean(half):.3f}")
    noise = [validation_loss((-2.0, -0.5), seed=s) for s in range(8)]
    print("same setting, 8 seeds:", np.round(noise, 3), "spread",
          np.std(noise).round(3))
    save_cache()

    # --- a convex problem all along ----------------------------------
    spec = importlib.util.spec_from_file_location(
        "ch08", HERE.parent / "ch08_duality" / "data.py")
    ch08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch08)
    A_raw, price = ch08.cars()
    A, b = ch08.standardize(A_raw), price - price.mean()
    loss = lambda x: np.mean((A @ x - b) ** 2) / 2
    t0 = time.perf_counter()
    exact = np.linalg.lstsq(A, b, rcond=None)[0]
    t1 = time.perf_counter()
    print(f"least squares: loss {loss(exact):.4f} in "
          f"{1e3 * (t1 - t0):.2f} ms")
    for name, run in (
        ("Nelder-Mead", lambda: minimize(
            loss, np.zeros(40), method="Nelder-Mead",
            options={"maxfev": 20_000, "maxiter": 20_000})),
        ("differential evolution", lambda: differential_evolution(
            loss, [(-10, 10)] * 40, maxiter=100, seed=0, polish=False)),
        ("dual annealing", lambda: dual_annealing(
            loss, [(-10, 10)] * 40, maxfun=20_000, seed=0,
            no_local_search=True)),
    ):
        t0 = time.perf_counter()
        res = run()
        print(f"{name}: loss {res.fun:.4f} after {res.nfev} evaluations, "
              f"{time.perf_counter() - t0:.1f} s")
    rng = np.random.default_rng(0)
    print("20,000 random points: best loss",
          min(loss(rng.uniform(-10, 10, 40)) for _ in range(20_000)))
