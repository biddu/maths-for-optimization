"""Chapter 10: machine parts that no straight line separates."""
import numpy as np


def parts(n=3000, seed=10, noise=0.05):
    """Two sensor readings per part; +1 sound, -1 faulty.

    A part is sound when both readings are near normal, so the
    sound parts form an island surrounded by faulty ones. A
    fraction `noise` of the labels is flipped."""
    rng = np.random.default_rng(seed)
    Z = rng.normal([5.0, 5.0], [2.0, 2.0], (n, 2))
    u, v = (Z[:, 0] - 5) / 2.6, (Z[:, 1] - 5) / 2.0
    inside = u ** 2 + v ** 2 + 0.6 * u * v < 1.0
    y = np.where(inside, 1.0, -1.0)
    flip = rng.uniform(size=n) < noise
    return Z, np.where(flip, -y, y)
