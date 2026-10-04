"""Chapter 13, from scratch: four methods that use no gradient."""
import numpy as np


def nelder_mead(f, x0, size=0.5, tol=1e-8, max_evals=5000):
    """Minimize f by moving a simplex of n + 1 points.

    Returns the best point, its value and the evaluations used."""
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    pts += [pts[0] + size * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        center = np.mean(pts[:-1], axis=0)   # all but the worst
        step = center - pts[-1]
        trial = center + step                # reflect the worst
        f_trial = f(trial); evals += 1
        if f_trial < vals[0]:                # very good: go on
            far = center + 2 * step
            f_far = f(far); evals += 1
            if f_far < f_trial:
                trial, f_trial = far, f_far
        elif f_trial >= vals[-2]:            # poor: come back
            trial = center - 0.5 * step
            f_trial = f(trial); evals += 1
            if f_trial >= vals[-1]:          # still bad: shrink
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
                evals += n
                continue
        pts[-1], vals[-1] = trial, f_trial
    best = int(np.argmin(vals))
    return pts[best], vals[best], evals


def tour_length(tour, D):
    """Length of the closed tour through the cities in order."""
    return D[tour, np.roll(tour, -1)].sum()


def nearest_neighbor(D, start=0):
    """Always go to the closest city not yet visited."""
    tour, left = [start], set(range(len(D))) - {start}
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt); left.remove(nxt)
    return np.array(tour)


def reverse_gain(tour, D, i, j):
    """Change in length if the stretch tour[i..j] is reversed."""
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % len(tour)]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def two_opt(tour, D):
    """Reverse stretches of the tour until no reversal helps."""
    tour, n = tour.copy(), len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if reverse_gain(tour, D, i, j) < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def anneal(tour, D, steps=200_000, hot=None, cold=1e-3, seed=0):
    """Simulated annealing with random reversals."""
    rng = np.random.default_rng(seed)
    tour, n = tour.copy(), len(tour)
    hot = D.mean() / 2 if hot is None else hot
    length = tour_length(tour, D)
    best, best_length = tour.copy(), length
    for k in range(steps):
        T = hot * (cold / hot) ** (k / steps)    # cooling
        i, j = sorted(rng.integers(1, n, 2))
        if i == j:
            continue
        gain = reverse_gain(tour, D, i, j)
        if gain < 0 or rng.random() < np.exp(-gain / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            length += gain
            if length < best_length:
                best, best_length = tour.copy(), length
    return best


def crossover(mom, dad, rng):
    """Keep a stretch of one parent; fill in the other's order."""
    n = len(mom)
    i, j = sorted(rng.integers(0, n, 2))
    kept = set(mom[i:j].tolist())
    rest = [c for c in dad if c not in kept]
    return np.array(rest[:i] + mom[i:j].tolist() + rest[i:])


def genetic(D, size=100, generations=500, mutate=0.3, seed=0):
    """A genetic algorithm for the tour: select, cross, mutate."""
    rng = np.random.default_rng(seed)
    n = len(D)
    crowd = [rng.permutation(n) for _ in range(size)]
    for _ in range(generations):
        crowd.sort(key=lambda t: tour_length(t, D))
        parents = crowd[:size // 2]              # the fitter half
        children = []
        while len(children) < size - len(parents):
            a, b = rng.integers(0, len(parents), 2)
            child = crossover(parents[a], parents[b], rng)
            if rng.random() < mutate:        # reverse a stretch
                i, j = sorted(rng.integers(0, n, 2))
                child[i:j + 1] = child[i:j + 1][::-1]
            children.append(child)
        crowd = parents + children
    return min(crowd, key=lambda t: tour_length(t, D))


def bayes_search(f, low, high, evals=25, start=5, scale=0.25,
                 noise=1e-3, seed=0):
    """Bayesian optimization in a box, with a Gaussian-process
    model and the expected-improvement rule."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    low, high = np.asarray(low, float), np.asarray(high, float)
    unit = lambda X: (np.atleast_2d(X) - low) / (high - low)
    apart = lambda A, B: ((A[:, None] - B[None]) ** 2).sum(axis=2)
    kernel = lambda A, B: np.exp(-apart(A, B) / (2 * scale ** 2))
    X = rng.uniform(low, high, (start, len(low)))
    y = np.array([f(x) for x in X])
    while len(y) < evals:
        mean, spread = y.mean(), y.std() + 1e-12
        K = kernel(unit(X), unit(X)) + noise * np.eye(len(y))
        cand = rng.uniform(low, high, (2000, len(low)))
        k = kernel(unit(cand), unit(X))
        mu = mean + k @ np.linalg.solve(K, y - mean)
        var = 1 - (k * np.linalg.solve(K, k.T).T).sum(axis=1)
        sd = spread * np.sqrt(np.maximum(var, 1e-12))
        z = (y.min() - mu) / sd
        gain = (y.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
        x = cand[np.argmax(gain)]                # most promising
        X, y = np.vstack([X, x]), np.append(y, f(x))
    return X, y
