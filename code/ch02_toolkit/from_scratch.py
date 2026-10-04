"""Chapter 2, from scratch: formula and difference gradients."""
import numpy as np

LN2 = np.log(2)


def rate(x):
    """Rate of the two-channel transmitter of Chapter 1."""
    return np.log2(1 + x[0]) + np.log2(1 + x[1] / 4)


def grad_rate(x):
    """Gradient of the rate, from the formula."""
    return np.array([1 / (1 + x[0]), 1 / (4 + x[1])]) / LN2


def hess_rate(x):
    """Hessian of the rate, from the formula."""
    d = [1 / (1 + x[0]) ** 2, 1 / (4 + x[1]) ** 2]
    return -np.diag(d) / LN2


def grad_forward(f, x, h=1e-8):
    """Forward differences: n + 1 evaluations of f."""
    x = np.asarray(x, dtype=float)
    g, fx = np.zeros(x.size), f(x)
    for i in range(x.size):
        step = np.zeros(x.size)
        step[i] = h
        g[i] = (f(x + step) - fx) / h
    return g


def grad_central(f, x, h=1e-5):
    """Central differences: 2n evaluations of f."""
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.size)
    for i in range(x.size):
        step = np.zeros(x.size)
        step[i] = h
        g[i] = (f(x + step) - f(x - step)) / (2 * h)
    return g


def gradient_check(f, grad, x, h=1e-5):
    """Relative gap between a coded gradient and differences."""
    exact, approx = grad(x), grad_central(f, x, h)
    gap = np.linalg.norm(exact - approx)
    scale = max(np.linalg.norm(exact), np.linalg.norm(approx))
    return gap / scale


def grad_rate_wrong(x):
    """A gradient with a chain-rule slip in the second entry."""
    return np.array([1 / (1 + x[0]), 1 / (1 + x[1] / 4)]) / LN2


if __name__ == "__main__":
    x = np.array([4.0, 6.0])
    print("rate      ", rate(x))
    print("gradient  ", grad_rate(x))
    print("forward   ", grad_forward(rate, x))
    print("central   ", grad_central(rate, x))
    print("check ok  ", gradient_check(rate, grad_rate, x))
    print("check bad ", gradient_check(rate, grad_rate_wrong, x))
