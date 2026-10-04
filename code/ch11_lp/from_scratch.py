"""Chapter 11, from scratch: the simplex method on a tableau."""
import numpy as np


def pivot(T, basis, row, col):
    """Rewrite the table with column col solved for in row."""
    T[row] /= T[row, col]
    for i in range(len(T)):
        if i != row:
            T[i] -= T[i, col] * T[row]
    basis[row] = col


def improve(T, basis, rule="largest", max_pivots=100_000):
    """Pivot until the last row offers no gain.

    Returns the points visited. rule is "largest" (coefficient)
    or "bland"."""
    m = len(T) - 1
    visited = []
    for _ in range(max_pivots):
        gain = -T[m, :-1]                # profit per unit entered
        if gain.max() <= 1e-9:
            return visited               # no improvement left
        if rule == "bland":
            col = int(np.argmax(gain > 1e-9))
        else:
            col = int(np.argmax(gain))
        rows = [i for i in range(m) if T[i, col] > 1e-9]
        if not rows:
            raise ValueError("the problem is unbounded")
        tie = (lambda i: basis[i]) if rule == "bland" else int
        row = min(rows,
                  key=lambda i: (T[i, -1] / T[i, col], tie(i)))
        pivot(T, basis, row, col)
        x = np.zeros(T.shape[1] - 1)
        x[basis] = T[:m, -1]
        visited.append(x)
    raise RuntimeError("no optimum after max_pivots pivots")


def simplex(c, A, b, rule="largest", max_pivots=100_000):
    """Maximize c @ x subject to A @ x <= b and x >= 0 (b >= 0).

    Returns x, the value, the shadow prices and the corners
    visited."""
    m, n = A.shape
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n], T[:m, n:n + m], T[:m, -1] = A, np.eye(m), b
    T[m, :n] = -np.asarray(c, dtype=float)
    basis = list(range(n, n + m))        # start at the slacks
    visited = improve(T, basis, rule, max_pivots)
    path = [np.zeros(n)] + [x[:n] for x in visited]
    return path[-1], T[m, -1], T[m, n:n + m], path


def two_phase(c, A, b, rule="largest"):
    """The same problem with no condition on the signs of b."""
    m, n = A.shape
    if np.min(b) >= 0:
        return simplex(c, A, b, rule)
    T = np.zeros((m + 1, n + m + 2))     # one more column, for x0
    T[:m, :n], T[:m, n:n + m], T[:m, -1] = A, np.eye(m), b
    T[:m, n + m] = -1                    # rules may give by x0
    T[m, n + m] = 1                      # phase 1: maximize -x0
    basis = list(range(n, n + m))
    pivot(T, basis, int(np.argmin(b)), n + m)
    visited = improve(T, basis, rule)
    if T[m, -1] < -1e-9:
        raise ValueError("the problem is infeasible")
    if n + m in basis:                   # x0 is zero: swap it out
        row = basis.index(n + m)
        col = int(np.argmax(np.abs(T[row, :n + m]) > 1e-9))
        pivot(T, basis, row, col)
    T = np.delete(T, n + m, axis=1)      # phase 2: real profit
    T[m] = 0
    T[m, :n] = -np.asarray(c, dtype=float)
    for i, j in enumerate(basis):
        T[m] -= T[m, j] * T[i]
    visited += improve(T, basis, rule)
    path = [np.zeros(n)] + [x[:n] for x in visited]
    return path[-1], T[m, -1], T[m, n:n + m], path


def corners(A, b):
    """Every corner of {x >= 0, A @ x <= b}, by brute force."""
    from itertools import combinations
    m, n = A.shape
    G = np.vstack([A, -np.eye(n)])       # all rules: G x <= h
    h = np.concatenate([b, np.zeros(n)])
    found = []
    for rows in combinations(range(m + n), n):
        M = G[list(rows)]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, h[list(rows)])
        if np.all(G @ x <= h + 1e-9):
            found.append(x)
    return np.unique(np.round(found, 9), axis=0)


if __name__ == "__main__":
    A = np.array([[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]])
    b = np.array([40.0, 60.0, 30.0])
    x, value, prices, path = simplex([40.0, 30.0], A, b)
    print(x, value, prices, [tuple(p) for p in path])
    print(corners(A, b))
    A4 = np.vstack([A, [-1.0, -1.0]])    # at least 25 items in all
    b4 = np.append(b, -25.0)
    x, value, prices, path = two_phase([40.0, 30.0], A4, b4)
    print(x, value, prices, [tuple(p) for p in path])
