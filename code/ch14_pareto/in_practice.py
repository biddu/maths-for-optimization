"""Chapter 14, in practice: fronts from two libraries."""
import importlib.util
import pathlib

import cvxpy as cp
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from from_scratch import G

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ch08", HERE.parent / "ch08_duality" / "data.py")
ch08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch08)           # the assets of Chapter 8


def portfolio(target, cap):
    """Least volatility with expected return >= target and no
    holding above cap. Returns None if that is impossible."""
    w = cp.Variable(5, nonneg=True)
    enough = ch08.RETURN @ w >= target
    spread = w <= cap
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, ch08.COV)),
                      [cp.sum(w) == 1, enough, spread])
    prob.solve()
    if prob.status != "optimal":
        return None
    return (np.sqrt(prob.value), w.value,
            float(enough.dual_value),
            float(np.sum(spread.dual_value)))


class Link(Problem):
    """The capstone as a black box: five transmit powers in,
    total power and (minus) rate out."""

    def __init__(self, circuit=0.5, most=5.0):
        super().__init__(n_var=5, n_obj=2, xl=0.0, xu=most)
        self.circuit = circuit

    def _evaluate(self, X, out, *args, **kwargs):
        on = X > 0.05                   # below this: switched off
        X = np.where(on, X, 0.0)
        power = X.sum(axis=1) + self.circuit * on.sum(axis=1)
        rate = np.log2(1 + G * X).sum(axis=1)
        out["F"] = np.column_stack([power, -rate])


def evolve(generations=200, size=100, seed=1, circuit=0.5):
    """A whole front from one run of NSGA-II."""
    result = minimize(Link(circuit), NSGA2(pop_size=size),
                      ("n_gen", generations), seed=seed,
                      verbose=False)
    power, rate = result.F[:, 0], -result.F[:, 1]
    order = np.argsort(power)
    return power[order], rate[order], result.X[order]


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print(portfolio(6.0, 1.0)[:1], portfolio(6.0, 0.3)[:1])
    power, rate, X = evolve()
    print(len(power), "designs from", power.min().round(2), "to",
          power.max().round(2), "watts")
