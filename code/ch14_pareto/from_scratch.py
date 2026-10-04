"""Chapter 14, from scratch: fronts, weights and limits."""
import itertools

import numpy as np

LN2 = np.log(2.0)
G = np.array([2.0, 1.0, 0.5, 0.25, 0.1])     # the five channels


def pareto(points):
    """Indices of the points not dominated by any other.

    Every column is a cost: smaller is better."""
    points = np.asarray(points, dtype=float)
    keep = []
    for i, p in enumerate(points):
        no_worse = np.all(points <= p, axis=1)
        better = np.any(points < p, axis=1)
        if not np.any(no_worse & better):
            keep.append(i)
    return keep


def waterfill(g, P):
    """Best powers for total power P (Chapter 7), and the rate."""
    floors = np.sort(1 / g)
    for k in range(len(g), 0, -1):
        level = (P + floors[:k].sum()) / k
        if level > floors[k - 1]:
            break
    x = np.maximum(level - 1 / g, 0)
    return x, np.sum(np.log2(1 + g * x))


def at_price(g, lam):
    """Powers that maximize rate - lam * power (Chapter 8)."""
    x = np.maximum(1 / (lam * LN2) - 1 / g, 0)
    return x, np.sum(np.log2(1 + g * x))


def weighted_sum(points, weights):
    """The point that minimizes a weighted sum of the costs."""
    totals = np.asarray(points) @ np.asarray(weights)
    return int(np.argmin(totals))


def with_limit(points, limit):
    """Least first cost among points whose second is <= limit."""
    points = np.asarray(points, dtype=float)
    allowed = np.flatnonzero(points[:, 1] <= limit)
    if allowed.size == 0:
        return None
    return int(allowed[np.argmin(points[allowed, 0])])


def link_design(g, budget, circuit=0.5):
    """Capstone: choose which channels to switch on, and their
    powers, when each active channel also draws circuit watts.

    Returns the best rate, the active set and the powers."""
    best = (0.0, (), np.zeros(len(g)))
    for size in range(1, len(g) + 1):
        for active in itertools.combinations(range(len(g)), size):
            power = budget - circuit * size
            if power <= 0:
                continue
            idx = list(active)
            x_part, rate = waterfill(g[idx], power)
            if rate > best[0]:
                x = np.zeros(len(g))
                x[idx] = x_part
                best = (rate, active, x)
    return best
