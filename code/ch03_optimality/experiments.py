"""Chapter 3: the three numerical experiments quoted in the text."""
import numpy as np

from from_scratch import normal_equations, standardize, used_cars

# 1. How rare is a positive definite matrix?
rng = np.random.default_rng(1)
for n in (1, 2, 3, 4, 5):
    count, trials = 0, 100_000
    for _ in range(trials):
        M = rng.normal(size=(n, n))
        count += np.linalg.eigvalsh((M + M.T) / 2)[0] > 0
    print(f"n={n}: fraction positive definite = {count / trials:.5f}")

# 2. Two correlated columns: which numbers are stable?
age, km, price = used_cars()
ones = np.ones(age.size)
Z = np.column_stack([ones, standardize(age), standardize(km)])
print("correlation:", np.corrcoef(age, km)[0, 1])
print("eigenvalues of Z^T Z:", np.linalg.eigvalsh(Z.T @ Z))
rng = np.random.default_rng(11)
both, alone = [], []
for _ in range(1000):
    noisy = price + rng.normal(0, 0.3, price.size)
    both.append(np.linalg.lstsq(Z, noisy, rcond=None)[0])
    alone.append(np.linalg.lstsq(Z[:, :2], noisy, rcond=None)[0])
both, alone = np.array(both), np.array(alone)
print("spread of age, km coefficients:", both[:, 1:].std(axis=0))
print("spread of their sum:", (both[:, 1] + both[:, 2]).std())
print("spread of age coefficient alone:", alone[:, 1].std())
print("age-only fit:", np.linalg.lstsq(Z[:, :2], price, rcond=None)[0])

# 3. Normal equations against lstsq on two nearly equal columns.
for delta in (1e-1, 1e-3, 1e-5, 1e-7, 1e-8):
    A = np.array([[1.0, 1.0], [delta, 0.0], [0.0, delta]])
    truth = np.array([1.0, 2.0])
    b = A @ truth
    stable = np.linalg.lstsq(A, b, rcond=None)[0]
    try:
        err = np.abs(normal_equations(A, b) - truth).max()
        normal = f"{err:.1e}"
    except np.linalg.LinAlgError:
        normal = "singular"
    print(f"delta={delta:.0e}  cond(A)={np.linalg.cond(A):.1e}  "
          f"normal: {normal}  lstsq: {np.abs(stable - truth).max():.1e}")
