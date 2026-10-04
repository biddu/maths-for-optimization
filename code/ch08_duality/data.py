"""Chapter 8: the data sets used in the chapter."""
import numpy as np

# Chapter 4's nine stores: position (km), deliveries per week.
STORES = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 3.0],
                   [8.0, 1.0], [9.0, 2.0], [7.0, 3.0],
                   [4.0, 8.0], [5.0, 9.0], [6.0, 8.0]])
TRIPS = np.array([3.0, 2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0])

# Five asset classes. Invented, round numbers: yearly return
# (percent), yearly volatility (percent), and correlations.
ASSETS = ["cash", "gov. bonds", "corp. bonds",
          "large shares", "small shares"]
RETURN = np.array([2.0, 3.5, 5.0, 7.5, 9.5])
VOL = np.array([0.5, 5.0, 8.0, 16.0, 24.0])
CORR = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.6, -0.1, -0.15],
                 [0.0, 0.6, 1.0, 0.4, 0.4],
                 [0.0, -0.1, 0.4, 1.0, 0.8],
                 [0.0, -0.15, 0.4, 0.8, 1.0]])
COV = CORR * np.outer(VOL, VOL)

def standardize(M, ref=None):
    """Scale columns using the mean and deviation of ref."""
    ref = M if ref is None else ref
    return (M - ref.mean(axis=0)) / ref.std(axis=0)


def cars(n=60, extra=37, seed=3):
    """Synthetic cars: 3 useful columns, then useless ones.

    Columns: age (yr), distance (1000 km), engine power (kW),
    then `extra` columns unrelated to the price ($1000)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 12, n)
    km = np.clip(15 * age + rng.normal(0, 8, n), 0, None)
    power = rng.normal(100, 25, n)
    noise = rng.normal(0, 1.5, n)
    price = (30 - 1.5 * age - 0.05 * km
             + 0.04 * (power - 100) + noise)
    junk = rng.normal(0, 1, (n, extra))
    return np.column_stack([age, km, power, junk]), price


def parts(n=40, seed=11):
    """Two sensor readings per part; +1 sound, -1 faulty."""
    rng = np.random.default_rng(seed)
    half = n // 2
    sound = rng.normal([3.0, 4.0], [0.9, 0.9], (half, 2))
    faulty = rng.normal([5.5, 6.0], [1.0, 1.0], (half, 2))
    X = np.vstack([sound, faulty])
    y = np.concatenate([np.ones(half), -np.ones(half)])
    return X, y
