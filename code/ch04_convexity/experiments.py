"""Chapter 4: the gallery, the hidden double well, the certificate."""
import numpy as np

from from_scratch import (chord_violation, least_curvature, one_depot,
                          two_depots)

gallery = [
    ("x^2", lambda x: x[0] ** 2, -5, 5, 1),
    ("|x|", lambda x: abs(x[0]), -5, 5, 1),
    ("exp(x)", lambda x: np.exp(x[0]), -5, 5, 1),
    ("log(e^x1 + e^x2)", lambda x: np.logaddexp(x[0], x[1]), -5, 5, 2),
    ("one depot", one_depot, 0, 10, 2),
    ("sin(x)", lambda x: np.sin(x[0]), 0, 10, 1),
    ("x^4 - 3x^2 + x", lambda x: x[0] ** 4 - 3 * x[0] ** 2 + x[0],
     -2, 2, 1),
    ("sqrt|x|", lambda x: np.sqrt(abs(x[0])), -5, 5, 1),
    ("x1 * x2", lambda x: x[0] * x[1], -5, 5, 2),
    ("two depots", two_depots, 0, 10, 4),
]
for name, f, low, high, dim in gallery:
    worst, _ = chord_violation(f, low, high, dim)
    print(f"{name:18s} largest violation {worst:10.4f}")


# A double well too small to see.
def tiny_well(x):
    return x[0] ** 4 - 0.002 * x[0] ** 2


print("tiny well, chords   :", chord_violation(tiny_well, -5, 5, 1)[0])
print("tiny well, curvature:", least_curvature(tiny_well, -5, 5, 1))
print("x1^2 + x1 x2 + x2^2 :",
      least_curvature(lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2,
                      -5, 5, 2))
print("x1 * x2             :",
      least_curvature(lambda x: x[0] * x[1], -5, 5, 2))

# The strong-convexity certificate on the five cars of Chapter 3,
# with age measured from 3 years (centered) and from 0 (raw).
ages = np.arange(1.0, 6.0)
b = np.array([18.0, 15.0, 14.0, 11.0, 9.0])
for label, col, x in (("centered", ages - 3, np.array([13.0, -2.0])),
                      ("raw", ages, np.array([19.0, -2.0]))):
    A = np.column_stack([np.ones(5), col])
    m = np.linalg.eigvalsh(A.T @ A)[0]
    x_star = np.linalg.solve(A.T @ A, A.T @ b)
    g = A.T @ (A @ x - b)
    gap = 0.5 * (np.sum((A @ x - b) ** 2)
                 - np.sum((A @ x_star - b) ** 2))
    print(f"{label:9s} m={m:.4f}  |g|={np.linalg.norm(g):.4f}  "
          f"bound={g @ g / (2 * m):.4f}  true gap={gap:.4f}  "
          f"distance bound={np.linalg.norm(g) / m:.4f}  "
          f"true distance={np.linalg.norm(x - x_star):.4f}")
