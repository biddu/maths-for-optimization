"""Chapter 1, from scratch: grid search and enumeration."""
import itertools
import numpy as np

TOL = 1e-9  # slack for floating-point comparisons


def grid_search(f, feasible, lower, upper, h):
    """Maximize f on a grid of spacing h over the feasible set."""
    axes = [np.arange(lo, hi + h / 2, h)
            for lo, hi in zip(lower, upper)]
    best_x, best_f, count = None, -np.inf, 0
    for point in itertools.product(*axes):
        x = np.array(point)
        count += 1
        if feasible(x) and f(x) > best_f:
            best_x, best_f = x, f(x)
    return best_x, best_f, count


def linear(x):
    """Version A: rate proportional to power."""
    return 3 * x[0] + 2 * x[1]


def feasible_a(x):
    return (x[0] + x[1] <= 10 + TOL and x[0] <= 6 + TOL
            and x[1] <= 8 + TOL)


def shannon(x):
    """Version B: Shannon rate, channel gains 1 and 1/4."""
    return np.log2(1 + x[0]) + np.log2(1 + x[1] / 4)


def feasible_b(x):
    return x[0] + x[1] <= 10 + TOL


def knapsack(costs, values, budget):
    """Version C: try every subset of channels."""
    best_set, best_value = (), 0
    n = len(costs)
    for r in range(n + 1):
        for subset in itertools.combinations(range(n), r):
            cost = sum(costs[i] for i in subset)
            value = sum(values[i] for i in subset)
            if cost <= budget and value > best_value:
                best_set, best_value = subset, value
    return best_set, best_value


if __name__ == "__main__":
    for name, f, feas in (("A", linear, feasible_a),
                          ("B", shannon, feasible_b)):
        for h in (1.0, 0.1, 0.01):
            x, fx, n = grid_search(f, feas, [0, 0], [10, 10], h)
            print(f"{name}  h={h:<5} x=({x[0]:.2f}, {x[1]:.2f})  "
                  f"f={fx:.6f}  points={n}")
    costs, values = [6, 5, 5, 3], [30, 24, 23, 9]
    for budget in (9, 10, 11):
        s, v = knapsack(costs, values, budget)
        print(f"C  budget={budget}  "
              f"channels={[i + 1 for i in s]}  value={v}")
