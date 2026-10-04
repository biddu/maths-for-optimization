"""Chapter 6: the test problems used throughout the chapter."""
import numpy as np


def churn_matrix(standardized=False, n=200, seed=4):
    """The subscriber data of Chapter 5 as a matrix A and labels y."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 20, n)
    months = rng.uniform(1, 48, n)
    score = 2.0 - 0.25 * hours - 0.05 * months
    cancelled = rng.uniform(size=n) < 1 / (1 + np.exp(-score))
    if standardized:
        hours = (hours - hours.mean()) / hours.std()
        months = (months - months.mean()) / months.std()
    A = np.column_stack([np.ones(n), hours, months])
    return A, np.where(cancelled, 1.0, -1.0)


def logistic(A, y):
    """Loss, gradient and Hessian of logistic regression on (A, y)."""
    def f(x):
        return np.mean(np.logaddexp(0, -y * (A @ x)))

    def grad(x):
        p = 1 / (1 + np.exp(y * (A @ x)))
        return -(A.T @ (y * p)) / y.size

    def hess(x):
        p = 1 / (1 + np.exp(-(A @ x)))
        return (A.T * (p * (1 - p))) @ A / y.size

    return f, grad, hess


def rosenbrock():
    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        return np.array([
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2)])

    def hess(x):
        return np.array([
            [2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]],
            [-400 * x[0], 200.0]])

    return f, grad, hess


def temperatures(n=2000, seed=8):
    """A noisy daily temperature record and the curve behind it."""
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    clean = 10 + 8 * np.sin(2 * np.pi * days / 365)
    return clean + rng.normal(0, 2, n), clean


def smoother(lam):
    """The map v -> (I + lam D^T D) v, without forming the matrix."""
    def matvec(v):
        d = np.diff(v)
        out = v.copy()
        out[:-1] -= lam * d
        out[1:] += lam * d
        return out
    return matvec
