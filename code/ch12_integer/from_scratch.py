"""Chapter 12, from scratch: branch and bound."""
import heapq
import math

import numpy as np
from scipy.optimize import linprog


def branch_and_bound(c, A, b, upper, integer=None, tol=1e-6):
    """Maximize c @ x subject to A @ x <= b, 0 <= x <= upper,
    with the listed variables whole numbers (all, if not given).

    Returns the best x, its value, the number of subproblems
    solved, and the bound from the first relaxation."""
    n = len(c)
    integer = range(n) if integer is None else integer
    best_x, best_value, solved, root = None, -math.inf, 0, None
    start = [(0.0, float(u)) for u in upper]
    queue = [(-math.inf, 0, start)]          # (-bound, id, boxes)
    while queue:
        bound, _, boxes = heapq.heappop(queue)
        if -bound <= best_value + tol:
            continue                         # cannot beat best
        res = linprog(-np.asarray(c), A_ub=A, b_ub=b,
                      bounds=boxes)
        solved += 1
        if res.status != 0:
            continue                         # no feasible point
        value = -res.fun
        root = value if root is None else root
        if value <= best_value + tol:
            continue                         # bound too low
        frac = [j for j in integer
                if abs(res.x[j] - round(res.x[j])) > tol]
        if not frac:                         # whole: a new best
            best_x, best_value = res.x.round(9), value
            continue
        j = max(frac,
                key=lambda k: abs(res.x[k] - round(res.x[k])))
        lo, hi = boxes[j]
        down, up = list(boxes), list(boxes)
        down[j] = (lo, math.floor(res.x[j]))
        up[j] = (math.ceil(res.x[j]), hi)
        heapq.heappush(queue, (-value, 2 * solved, down))
        heapq.heappush(queue, (-value, 2 * solved + 1, up))
    return best_x, best_value, solved, root


if __name__ == "__main__":
    cost = np.array([[6.0, 5.0, 5.0, 3.0]])
    rate = np.array([30.0, 24.0, 23.0, 9.0])
    print(branch_and_bound(rate, cost, [10.0], [1, 1, 1, 1]))
    A = np.array([[6.0, 5.0], [1.0, 3.0]])
    print(branch_and_bound([9.0, 8.0], A, [42.0, 19.0], [10, 10]))
