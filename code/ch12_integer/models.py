"""Chapter 12: modeling with binary variables."""
import numpy as np

# Four candidate warehouses and the four cities of Chapter 11.
CAPACITY = np.array([200.0, 250.0, 180.0, 220.0])
RENT = np.array([520.0, 640.0, 380.0, 560.0])     # euro per week
DEMAND = np.array([80.0, 70.0, 110.0, 60.0])
RATE = np.array([[4.0, 6.0, 9.0, 5.0],            # euro per pallet
                 [7.0, 3.0, 4.0, 8.0],
                 [6.0, 5.0, 2.0, 7.0],
                 [5.0, 4.0, 6.0, 3.0]])


def location_model(capacity=CAPACITY, rent=RENT, demand=DEMAND,
                   rate=RATE, big=None, strong=False):
    """Which warehouses to rent, and what to ship from each.

    Variables: shipments x[i, j] row by row, then open[i] (0/1).
    Returns c, A, b, upper for `maximize c @ z, A @ z <= b`."""
    W, C = rate.shape
    n = W * C + W
    link = capacity if big is None else np.full(W, float(big))
    rows, rhs = [], []
    for i in range(W):                    # ship only from open sites
        row = np.zeros(n)
        row[i * C:(i + 1) * C] = 1
        row[W * C + i] = -link[i]
        rows.append(row); rhs.append(0.0)
        if big is not None:               # the real capacity
            row = np.zeros(n)
            row[i * C:(i + 1) * C] = 1
            rows.append(row); rhs.append(capacity[i])
    for j in range(C):                    # meet every demand
        row = np.zeros(n)
        row[j:W * C:C] = -1
        rows.append(row); rhs.append(-demand[j])
    if strong:                            # x[i, j] <= demand[j] open[i]
        for i in range(W):
            for j in range(C):
                row = np.zeros(n)
                row[i * C + j] = 1
                row[W * C + i] = -demand[j]
                rows.append(row); rhs.append(0.0)
    c = -np.concatenate([rate.ravel(), rent])
    upper = np.concatenate([np.full(W * C, demand.sum()), np.ones(W)])
    return c, np.array(rows), np.array(rhs), upper


def random_sites(W=10, C=15, seed=1):
    rng = np.random.default_rng(seed)
    demand = rng.integers(20, 80, C).astype(float)
    capacity = rng.integers(200, 400, W).astype(float)
    rent = rng.integers(300, 700, W).astype(float)
    rate = rng.integers(2, 12, (W, C)).astype(float)
    return capacity, rent, demand, rate
