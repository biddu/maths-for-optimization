"""Chapter 10, from scratch: a small network and its gradient."""
import numpy as np


def start(width=16, seed=0):
    """Small random weights for 2 inputs, `width` hidden units."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, 4 * width + 1)


def unpack(theta, width=16):
    """Split the parameter vector into W1, b1, w2, b2."""
    W1 = theta[:2 * width].reshape(2, width)
    b1 = theta[2 * width:3 * width]
    w2 = theta[3 * width:4 * width]
    return W1, b1, w2, theta[-1]


def loss_and_grad(theta, X, y, width=16):
    """Logistic loss of the network, and its gradient."""
    W1, b1, w2, b2 = unpack(theta, width)
    H = np.tanh(X @ W1 + b1)            # forward: hidden layer
    z = H @ w2 + b2                     # forward: one score each
    loss = np.mean(np.logaddexp(0, -y * z))
    dz = -y / (1 + np.exp(y * z)) / y.size   # backward from here
    dw2, db2 = H.T @ dz, dz.sum()
    dH = np.outer(dz, w2)
    dA = dH * (1 - H ** 2)              # back through tanh
    dW1, db1 = X.T @ dA, dA.sum(axis=0)
    grad = np.concatenate([dW1.ravel(), db1, dw2, [db2]])
    return loss, grad


def batch_gradient(X, y, batch=64, seed=0):
    """A function that returns the gradient on a fresh batch."""
    rng = np.random.default_rng(seed)

    def grad(theta):
        rows = rng.integers(0, y.size, batch)
        return loss_and_grad(theta, X[rows], y[rows])[1]
    return grad
