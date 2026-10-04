"""Chapter 8, from scratch: a dual solved by bisection, and
the lasso solved by soft-thresholding."""
import numpy as np

LN2 = np.log(2.0)


def best_response(lam, g):
    """Powers that minimize the Lagrangian at price lam."""
    return np.maximum(1 / (lam * LN2) - 1 / g, 0)


def dual_function(lam, g, P):
    """q(lam): never above minus the best rate."""
    x = best_response(lam, g)
    return -np.sum(np.log2(1 + g * x)) + lam * (x.sum() - P)


def solve_dual(g, P, tol=1e-10):
    """Maximize q by bisection on its slope, sum(x) - P."""
    lo, hi = 0.0, g.max() / LN2
    while hi - lo > tol:
        lam = (lo + hi) / 2
        if best_response(lam, g).sum() > P:
            lo = lam      # too much power wanted: raise price
        else:
            hi = lam
    return (lo + hi) / 2


def soft_threshold(v, t):
    """Move every entry of v toward zero by t, stopping at 0."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)


def lasso(A, b, lam, steps=2000):
    """Minimize (1/2n)|Ax - b|^2 + lam |x|_1."""
    n = len(b)
    L = np.linalg.eigvalsh(A.T @ A / n).max()
    x = np.zeros(A.shape[1])
    for _ in range(steps):
        grad = A.T @ (A @ x - b) / n
        x = soft_threshold(x - grad / L, lam / L)
    return x


if __name__ == "__main__":
    g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
    lam = solve_dual(g, 10.0)
    print("price", round(lam, 4), "powers",
          best_response(lam, g).round(3),
          "bound", round(-dual_function(lam, g, 10.0), 4))
