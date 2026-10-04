"""Chapter 3, from scratch: stationary points, least squares."""
import numpy as np


def classify(H, tol=1e-8):
    """Name a stationary point from its Hessian's eigenvalues."""
    lam = np.linalg.eigvalsh(H)
    if lam[0] > tol:
        return "local minimum"
    if lam[-1] < -tol:
        return "local maximum"
    if lam[0] < -tol and lam[-1] > tol:
        return "saddle point"
    return "inconclusive"


def normal_equations(A, b):
    """Least squares by solving (A^T A) x = A^T b."""
    return np.linalg.solve(A.T @ A, A.T @ b)


def standardize(col):
    """Shift a column to mean 0, scale it to deviation 1."""
    return (col - col.mean()) / col.std()


def used_cars(n=200, seed=3):
    """Synthetic cars: age (yr), distance (km), price ($1000)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 12, n)
    km = np.clip(15000 * age + rng.normal(0, 8000, n), 0, None)
    noise = rng.normal(0, 1.5, n)
    price = 30 - 1.5 * age - 0.05 * km / 1000 + noise
    return age, km, price


def two_valleys(x):
    """A function with two minima and a saddle between them."""
    return (x[0] ** 2 - 1) ** 2 + x[1] ** 2


def grad_two_valleys(x):
    return np.array([4 * x[0] * (x[0] ** 2 - 1), 2 * x[1]])


def hess_two_valleys(x):
    return np.array([[12 * x[0] ** 2 - 4, 0.0], [0.0, 2.0]])


if __name__ == "__main__":
    # Five cars by hand: age (years), price ($1000).
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([18.0, 15.0, 14.0, 11.0, 9.0])
    A = np.column_stack([np.ones(5), t])
    x = normal_equations(A, y)
    r = y - A @ x
    print("A^T A =", (A.T @ A).tolist())
    print("A^T b =", (A.T @ y).tolist())
    print("fit:", x, " residuals:", r.round(4))
    print("A^T r:", (A.T @ r).round(10))
    print("sum of squares:", r @ r)
    for p in ([1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]):
        print(p, classify(hess_two_valleys(np.array(p))))
