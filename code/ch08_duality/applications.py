"""Chapter 8, in practice: three convex programs that matter."""
import cvxpy as cp
import numpy as np

from data import COV, RETURN, cars, parts, standardize


def portfolio(target):
    """Least-variance mix with expected return >= target."""
    w = cp.Variable(5, nonneg=True)
    enough = RETURN @ w >= target
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, COV)),
                      [cp.sum(w) == 1, enough])
    prob.solve()
    return w.value, np.sqrt(prob.value), float(enough.dual_value)


def lasso_cvx(A, b, lam):
    """The lasso, stated as it is written on paper."""
    x = cp.Variable(A.shape[1])
    fit = cp.sum_squares(A @ x - b) / (2 * len(b))
    cp.Problem(cp.Minimize(fit + lam * cp.norm1(x))).solve()
    return x.value


def svm(X, y, C=1.0):
    """Soft-margin classifier, with its multipliers."""
    w, b = cp.Variable(2), cp.Variable()
    slack = cp.Variable(len(y), nonneg=True)
    margin = cp.multiply(y, X @ w + b) >= 1 - slack
    cost = cp.sum_squares(w) / 2 + C * cp.sum(slack)
    prob = cp.Problem(cp.Minimize(cost), [margin])
    prob.solve()
    return w.value, b.value, margin.dual_value, prob.value


def svm_dual(X, y, C=1.0):
    """The dual of the same problem: one variable per point."""
    a = cp.Variable(len(y))
    gain = cp.sum(a) - cp.sum_squares(X.T @ cp.multiply(a, y)) / 2
    prob = cp.Problem(cp.Maximize(gain),
                      [a >= 0, a <= C, a @ y == 0])
    prob.solve(solver=cp.CLARABEL)
    return a.value, prob.value


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    w, risk, price = portfolio(6.0)
    print("portfolio:", w.round(3))
    print("  volatility", round(risk, 3), "price", round(price, 3))

    A, price = cars()
    A, b = standardize(A), price - price.mean()
    print("lasso:", lasso_cvx(A, b, 0.3).round(2)[:6])

    X, y = parts()
    w, b0, alpha, value = svm(X, y)
    print("svm: w", w.round(3), "b", round(float(b0), 3),
          "value", round(value, 4),
          "support vectors", int(np.sum(alpha > 1e-5)))
    a, dual_value = svm_dual(X, y)
    print("  dual value", round(dual_value, 4),
          "w from dual", (X.T @ (a * y)).round(3))
