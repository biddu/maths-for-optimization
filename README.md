# Maths for Optimization: companion code

Code for the book *Maths for Optimization: From Gradients to Solvers*
(Mathematics for Everything, Book 6) by Avishek Nag.

Every listing printed in the book is taken from a file in this repository, and
every number quoted in the text is produced by one of these scripts. All data
sets are synthetic and generated from fixed seeds.

## Layout

One folder per chapter under `code/`:

| Folder | Chapter |
|---|---|
| `ch01_problems` | 1. What an Optimization Problem Is |
| `ch02_toolkit` | 2. The Toolkit: Calculus and Linear Algebra |
| `ch03_optimality` | 3. Optimality Conditions Without Constraints |
| `ch04_convexity` | 4. Convexity |
| `ch05_gradient_descent` | 5. Gradient Descent |
| `ch06_newton` | 6. Newton and Quasi-Newton Methods |
| `ch07_constraints` | 7. Constraints: Lagrange Multipliers and KKT |
| `ch08_duality` | 8. Duality and Convex Programs |
| `ch09_sgd` | 9. Stochastic Gradient Descent |
| `ch10_adam` | 10. Momentum, Adam and Training Neural Networks |
| `ch11_lp` | 11. Linear Programming |
| `ch12_integer` | 12. Integer Programming |
| `ch13_heuristics` | 13. When There Is No Gradient |
| `ch14_pareto` | 14. Many Objectives: Tradeoffs and Pareto Fronts |

In each folder:

- `from_scratch.py`: the algorithms of the chapter in plain NumPy.
- `in_practice.py`: the same problems with SciPy, CVXPY, PyTorch or OR-Tools.
- `experiments.py`: produces the tables and the numbers quoted in the text.
- `figures.py`: draws the chapter's figures into `manuscript/figures/`.
- `solutions/solutions.py`: numerical checks for the exercise solutions.

Some chapters have further files (`data.py`, `network.py`, `models.py`,
`applications.py`) and small cached results (`*.json`).

## Running

    pip install -r requirements.txt
    cd code/ch05_gradient_descent
    python from_scratch.py
    python experiments.py

Run each script from inside its chapter folder. A few chapters import data
from an earlier one, so keep the folder structure as it is.

## What to expect

Results that depend only on arithmetic and fixed seeds reproduce exactly.
Timings will differ on your machine. Results obtained under a wall-clock limit
(the routing and exact-tour runs in Chapter 13, the scheduling comparison in
Chapter 12) can also differ, because a faster machine gets further in the same
time.

The printed edition was produced with Python 3.13, NumPy 2.5, SciPy 1.18,
CVXPY 1.9, PyTorch 2.14, OR-Tools 9.15 and pymoo 0.6.

## License

MIT. See `LICENSE`.
