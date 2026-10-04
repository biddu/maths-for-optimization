"""Chapter 2, in practice: the same checks in SciPy and NumPy."""
import numpy as np
from scipy.optimize import approx_fprime, check_grad

from from_scratch import rate, grad_rate, grad_rate_wrong

x = np.array([4.0, 6.0])

# Finite-difference gradient, and the gap to a coded gradient.
print(approx_fprime(x, rate))
print(check_grad(rate, grad_rate, x))
print(check_grad(rate, grad_rate_wrong, x))

# Is a symmetric matrix positive definite?
A = np.array([[2.0, 1.0], [1.0, 2.0]])
B = np.array([[1.0, 2.0], [2.0, 1.0]])
print(np.linalg.eigvalsh(A), np.linalg.eigvalsh(B))


def is_positive_definite(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


print(is_positive_definite(A), is_positive_definite(B))
