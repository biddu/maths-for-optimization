"""Chapter 10, from scratch: momentum and adaptive steps."""
import numpy as np


def momentum(grad, x0, step, beta, steps, nesterov=False):
    """Gradient descent with a velocity that remembers."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x - step * beta * v) if nesterov else grad(x)
        v = beta * v + g
        x = x - step * v
        path.append(x.copy())
    return np.array(path)


def adagrad(grad, x0, step, steps, eps=1e-8):
    """Divide each coordinate by the root of its past squares."""
    x = np.array(x0, dtype=float)
    s = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x)
        s += g * g
        x = x - step * g / (np.sqrt(s) + eps)
        path.append(x.copy())
    return np.array(path)


def rmsprop(grad, x0, step, steps, beta2=0.99, eps=1e-8):
    """AdaGrad with a fading memory of the squares."""
    x = np.array(x0, dtype=float)
    s = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x)
        s = beta2 * s + (1 - beta2) * g * g
        x = x - step * g / (np.sqrt(s) + eps)
        path.append(x.copy())
    return np.array(path)


def adam(grad, x0, step, steps, beta1=0.9, beta2=0.999,
         eps=1e-8, decay=0.0):
    """Adam; decay > 0 gives AdamW."""
    x = np.array(x0, dtype=float)
    m, s = np.zeros_like(x), np.zeros_like(x)
    path = [x.copy()]
    for k in range(1, steps + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** k)
        s_hat = s / (1 - beta2 ** k)
        x = x - step * decay * x
        x = x - step * m_hat / (np.sqrt(s_hat) + eps)
        path.append(x.copy())
    return np.array(path)
