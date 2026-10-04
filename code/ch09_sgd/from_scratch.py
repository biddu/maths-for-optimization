"""Chapter 9, from scratch: stochastic gradient descent."""
import numpy as np


def subscribers(n=120_000, d=20, seed=9):
    """Synthetic subscribers: d standardized features, and
    +1 if the subscriber cancelled, -1 otherwise."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(d, d)) / np.sqrt(d)
    Z = rng.normal(size=(n, d)) @ (np.eye(d) + mix)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    A = np.column_stack([np.ones(n), Z])
    truth = np.concatenate([[-1.0], rng.normal(0, 0.7, d)])
    p = 1 / (1 + np.exp(-A @ truth))
    return A, np.where(rng.uniform(size=n) < p, 1.0, -1.0)


def loss(x, A, y):
    """Average of log(1 + exp(-y_i a_i^T x)) over the rows."""
    return np.mean(np.logaddexp(0, -y * (A @ x)))


def gradient(x, A, y):
    """Gradient of the average loss over the rows given."""
    p = 1 / (1 + np.exp(y * (A @ x)))
    return -(A.T @ (y * p)) / y.size


def sgd(A, y, x0, step, batch=1, epochs=1, seed=0, every=None,
        shuffle=True):
    """Mini-batch SGD, reshuffling the data every epoch.

    step is a number, or a function of the update count k.
    Returns the final x and a list of snapshots of x."""
    rng = np.random.default_rng(seed)
    x, k, n = np.array(x0, dtype=float), 0, y.size
    every = every or -(-n // batch)
    snapshots = []
    for _ in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch):
            rows = order[start:start + batch]
            t = step(k) if callable(step) else step
            x -= t * gradient(x, A[rows], y[rows])
            k += 1
            if k % every == 0:
                snapshots.append(x.copy())
    return x, snapshots


def running_mean_sgd(a, step, x0=0.0):
    """SGD on (1/2)(x - a_i)^2, one pass through the stream a."""
    x, path = x0, []
    for k, a_k in enumerate(a):
        t = step(k) if callable(step) else step
        x -= t * (x - a_k)
        path.append(x)
    return np.array(path)
