"""Chapter 7, from scratch: three ways to handle constraints."""
import numpy as np
from scipy.optimize import minimize


def rate(x, g):
    """Total rate of channels with qualities g at powers x."""
    return np.sum(np.log2(1 + g * x))


def waterfill(g, P):
    """Solve the KKT conditions of the power problem exactly."""
    floors = 1 / np.asarray(g, dtype=float)
    order = np.argsort(floors)
    for k in range(floors.size, 0, -1):
        level = (P + floors[order[:k]].sum()) / k
        if level > floors[order[k - 1]]:
            break
    return np.maximum(level - floors, 0), level


def penalty_method(f, h, x0, rhos):
    """Minimize f subject to h(x) = 0 by growing penalties."""
    x, rows = np.asarray(x0, dtype=float), []
    for rho in rhos:
        res = minimize(lambda z: f(z) + 0.5 * rho * h(z) ** 2, x,
                       method="BFGS", options={"gtol": 1e-8})
        x = res.x
        rows.append((rho, x.copy(), h(x), rho * h(x), res.nit))
    return x, rows


def project_budget(v, P):
    """Nearest point to v with entries >= 0 that sum to P."""
    u = np.sort(v)[::-1]
    excess = (np.cumsum(u) - P) / np.arange(1, v.size + 1)
    k = np.nonzero(u > excess)[0][-1]
    return np.maximum(v - excess[k], 0)


def projected_gradient(grad, project, x0, step, tol=1e-9,
                       max_iter=100_000):
    """Gradient steps, each followed by a projection."""
    x = project(np.asarray(x0, dtype=float))
    for k in range(max_iter):
        x_new = project(x - step * grad(x))
        if np.linalg.norm(x_new - x) <= tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter
