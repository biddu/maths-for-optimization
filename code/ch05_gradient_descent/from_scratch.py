"""Chapter 5, from scratch: gradient descent with backtracking."""
import numpy as np


def backtracking(f, x, g, t, alpha=0.3, beta=0.5):
    """Halve the step until it gives a sufficient decrease."""
    fx, gg = f(x), g @ g
    while f(x - t * g) > fx - alpha * t * gg:
        t *= beta
    return t


def gradient_descent(f, grad, x0, step=None, tol=1e-6,
                     max_iter=100_000):
    """Minimize f from x0; step=None means backtracking."""
    x = np.asarray(x0, dtype=float)
    t = 1.0 if step is None else step
    history = [f(x)]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            break
        if step is None:
            t = backtracking(f, x, g, 2 * t)
        x = x - t * g
        history.append(f(x))
    return x, np.array(history)


def churn_data(n=200, seed=4):
    """Synthetic subscribers: hours watched, months, cancelled?"""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 20, n)
    months = rng.uniform(1, 48, n)
    score = 2.0 - 0.25 * hours - 0.05 * months
    cancelled = rng.uniform(size=n) < 1 / (1 + np.exp(-score))
    return hours, months, np.where(cancelled, 1.0, -1.0)


def logistic_loss(x, A, y):
    """Average of log(1 + exp(-y_i a_i^T x)) over the data."""
    return np.mean(np.logaddexp(0, -y * (A @ x)))


def logistic_grad(x, A, y):
    p = 1 / (1 + np.exp(y * (A @ x)))
    return -(A.T @ (y * p)) / y.size


def standardize(col):
    return (col - col.mean()) / col.std()


if __name__ == "__main__":
    f = lambda x: 0.5 * (x[0] ** 2 + 10 * x[1] ** 2)
    grad = lambda x: np.array([x[0], 10 * x[1]])
    for step in (0.01, 0.05, 0.1, 0.19, 0.21, None):
        x, hist = gradient_descent(f, grad, [10.0, 1.0], step,
                                   max_iter=2000)
        print(step, hist.size - 1, hist[-1])
