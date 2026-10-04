"""Chapter 6, in practice: four SciPy methods on two problems."""
import numpy as np
from scipy.optimize import minimize

from problems import churn_matrix, logistic, rosenbrock

cases = [("churn (raw)", logistic(*churn_matrix()), np.zeros(3)),
         ("Rosenbrock", rosenbrock(), np.array([-1.2, 1.0]))]

for name, (f, grad, hess), start in cases:
    print(name)
    for method in ("BFGS", "L-BFGS-B", "CG", "Newton-CG"):
        h = hess if method == "Newton-CG" else None
        res = minimize(f, start, jac=grad, hess=h, method=method)
        gnorm = np.linalg.norm(grad(res.x))
        print(f"  {method:10s} steps {res.nit:3d}"
              f"  f evals {res.nfev:3d}  g evals {res.njev:3d}"
              f"  final |g| {gnorm:.1e}")
