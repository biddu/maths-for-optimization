"""Chapter 4, from scratch: evidence against convexity."""
import numpy as np

# Nine stores in three towns: position (km), deliveries per week.
STORES = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 3.0],
                   [8.0, 1.0], [9.0, 2.0], [7.0, 3.0],
                   [4.0, 8.0], [5.0, 9.0], [6.0, 8.0]])
TRIPS = np.array([3.0, 2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0])


def one_depot(x):
    """Weekly distance driven from a single depot at x."""
    return TRIPS @ np.linalg.norm(STORES - x, axis=1)


def two_depots(z):
    """Depots at z[:2] and z[2:]; each store uses the nearer."""
    d1 = np.linalg.norm(STORES - z[:2], axis=1)
    d2 = np.linalg.norm(STORES - z[2:], axis=1)
    return TRIPS @ np.minimum(d1, d2)


def chord_violation(f, low, high, dim, trials=10_000,
                    seed=0):
    """Largest amount by which f rises above one of its chords."""
    rng = np.random.default_rng(seed)
    worst, witness = 0.0, None
    for _ in range(trials):
        x = rng.uniform(low, high, dim)
        y = rng.uniform(low, high, dim)
        t = rng.uniform()
        gap = f(t * x + (1 - t) * y) - (t * f(x) + (1 - t) * f(y))
        if gap > worst:
            worst, witness = gap, (x, y, t)
    return worst, witness


def hessian(f, x, h=1e-4):
    """Hessian by central differences of the function itself."""
    n = x.size
    H = np.zeros((n, n))
    E = h * np.eye(n)
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                       - f(x - E[i] + E[j]) + f(x - E[i] - E[j]))
    return H / (4 * h * h)


def least_curvature(f, low, high, dim, trials=2_000,
                    seed=0):
    """Smallest Hessian eigenvalue found at random points."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(low, high, (trials, dim))
    return min(np.linalg.eigvalsh(hessian(f, p))[0]
               for p in points)


if __name__ == "__main__":
    print("one depot :", chord_violation(one_depot, 0, 10, 2)[0])
    worst, (x, y, t) = chord_violation(two_depots, 0, 10, 4)
    print("two depots:", worst)
    print("  witness:", x.round(2), y.round(2), round(t, 3))
