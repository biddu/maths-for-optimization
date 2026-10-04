"""Chapter 7, in practice: SciPy's constrained solver."""
import numpy as np
from scipy.optimize import minimize

from from_scratch import rate, waterfill

# The can: least surface area for a volume of 330 cm^3.
area = lambda z: 2 * np.pi * z[0] ** 2 + 2 * np.pi * z[0] * z[1]
volume = {"type": "eq",
          "fun": lambda z: np.pi * z[0] ** 2 * z[1] - 330}
res = minimize(area, [3.0, 10.0], constraints=[volume])
print("can:", res.x.round(4), round(res.fun, 3), res.multipliers)

# Five channels, 10 watts: maximize the rate.
g = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
power = {"type": "ineq", "fun": lambda x: 10 - x.sum()}
res = minimize(lambda x: -rate(x, g), np.full(5, 2.0),
               bounds=[(0, None)] * 5, constraints=[power])
print("channels:", res.x.round(3), round(-res.fun, 4),
      res.multipliers)
print("exact:   ", waterfill(g, 10.0)[0])
