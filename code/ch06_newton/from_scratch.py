"""Chapter 6, from scratch: Newton, BFGS, conjugate gradient."""
import numpy as np


def line_search(f, x, g, d, alpha=0.3, beta=0.5):
    """Backtrack from t = 1 along a descent direction d."""
    t, fx, slope = 1.0, f(x), g @ d
    while f(x + t * d) > fx + alpha * t * slope:
        t *= beta
    return t


def newton_direction(H, g):
    """Solve H d = -g, shifting H until positive definite."""
    shift = 0.0
    while True:
        try:
            L = np.linalg.cholesky(H + shift * np.eye(g.size))
            return -np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:
            shift = max(2 * shift, 1e-3)


def newton(f, grad, hess, x0, tol=1e-8, max_iter=100,
           damped=True):
    """Newton's method; returns the point and gradient norms."""
    x = np.asarray(x0, dtype=float)
    norms = []
    for k in range(max_iter):
        g = grad(x)
        norms.append(np.linalg.norm(g))
        if norms[-1] <= tol:
            break
        d = newton_direction(hess(x), g)
        t = line_search(f, x, g, d) if damped else 1.0
        x = x + t * d
    return x, np.array(norms)


def bfgs(f, grad, x0, tol=1e-8, max_iter=1000):
    """BFGS with an inverse-Hessian estimate Hinv."""
    x = np.asarray(x0, dtype=float)
    n = x.size
    I, Hinv = np.eye(n), np.eye(n)
    g = grad(x)
    norms = [np.linalg.norm(g)]
    for k in range(max_iter):
        if norms[-1] <= tol:
            break
        d = -Hinv @ g
        t = line_search(f, x, g, d)
        s = t * d
        g_new = grad(x + s)
        y = g_new - g
        if y @ s > 1e-12:
            if k == 0:
                Hinv = (y @ s) / (y @ y) * I
            rho = 1 / (y @ s)
            V = I - rho * np.outer(s, y)
            Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)
        x, g = x + s, g_new
        norms.append(np.linalg.norm(g))
    return x, np.array(norms)


def conjugate_gradient(matvec, b, tol=1e-8, max_iter=None):
    """Solve A x = b, A positive definite, given v -> A v."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    residuals = [np.sqrt(rr)]
    for k in range(max_iter or b.size):
        if residuals[-1] <= tol * residuals[0]:
            break
        Ap = matvec(p)
        step = rr / (p @ Ap)
        x += step * p
        r -= step * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
        residuals.append(np.sqrt(rr))
    return x, np.array(residuals)
