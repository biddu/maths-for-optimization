"""Chapter 10: every number quoted in the chapter."""
import importlib.util
import math
import pathlib
import time

import numpy as np
import torch
from scipy.optimize import minimize

from data import parts
from from_scratch import adagrad, adam, momentum, rmsprop
from in_practice import (BEST, OPTIMIZERS, X_test, X_train, Y_test,
                         Y_train, evaluate, loss_fn, network, train)
from network import batch_gradient, loss_and_grad, start

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ch05", HERE.parent / "ch05_gradient_descent" / "from_scratch.py")
ch05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch05)


def bowl(kappa):
    f = lambda x: 0.5 * (x[0] ** 2 + kappa * x[1] ** 2)
    grad = lambda x: np.array([x[0], kappa * x[1]])
    return f, grad


def steps_to(path, f, target):
    v = np.array([f(p) for p in path])
    hit = v <= target
    return int(np.argmax(hit)) if hit.any() else None


def raw_churn():
    hours, months, y = ch05.churn_data()
    A = np.column_stack([np.ones(200), hours, months])
    f = lambda x: ch05.logistic_loss(x, A, y)
    grad = lambda x: ch05.logistic_grad(x, A, y)
    best = minimize(f, np.zeros(3), jac=grad, method="BFGS",
                    options={"gtol": 1e-12}).fun
    L = np.linalg.eigvalsh(A.T @ A / 800).max()
    return f, grad, best, L


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # --- momentum on the bowl -----------------------------------
    for kappa in (10, 100, 1000, 10_000):
        f, grad = bowl(kappa)
        r = math.sqrt(kappa)
        gd = momentum(grad, [1, 1], 1 / kappa, 0.0, 30_000)
        hb = momentum(grad, [1, 1], 4 / (r + 1) ** 2,
                      ((r - 1) / (r + 1)) ** 2, 3000)
        ne = momentum(grad, [1, 1], 1 / kappa, (r - 1) / (r + 1), 3000,
                      nesterov=True)
        goal = f(np.ones(2)) / 1e6
        print(f"kappa {kappa}: GD {steps_to(gd, f, goal)}, heavy ball "
              f"{steps_to(hb, f, goal)} (beta "
              f"{((r - 1) / (r + 1)) ** 2:.3f}), Nesterov "
              f"{steps_to(ne, f, goal)}")
    f, grad = bowl(100)
    for beta in (0.0, 0.5, 0.9, 0.99):
        p = momentum(grad, [1, 1], 1 / 100, beta, 30_000)
        print("kappa 100, step 1/L, beta", beta, "steps",
              steps_to(p, f, f(np.ones(2)) / 1e6))

    # --- momentum and noise: the call lengths of Chapter 9 ------
    rng = np.random.default_rng(0)
    ends = {"plain 0.01": [], "plain 0.1": [], "momentum 0.01": []}
    for _ in range(4000):
        a = rng.normal(6, 2, 1500)
        for name, t, beta in (("plain 0.01", 0.01, 0.0),
                              ("plain 0.1", 0.1, 0.0),
                              ("momentum 0.01", 0.01, 0.9)):
            x = v = 0.0
            for a_k in a:
                v = beta * v + (x - a_k)
                x -= t * v
            ends[name].append(x)
    print({k: round(float(np.std(v)), 3) for k, v in ends.items()})

    # --- adaptive steps on the raw subscriber data --------------
    f, grad, best, L = raw_churn()
    runs = {
        "GD, step 1/L": momentum(grad, np.zeros(3), 1 / L, 0, 100_000),
        "momentum 0.9": momentum(grad, np.zeros(3), 1 / L, 0.9, 20_000),
        "momentum 0.99": momentum(grad, np.zeros(3), 1 / L, 0.99, 20_000),
        "AdaGrad 0.3": adagrad(grad, np.zeros(3), 0.3, 20_000),
        "RMSProp 0.003": rmsprop(grad, np.zeros(3), 0.003, 20_000),
        "Adam 0.03": adam(grad, np.zeros(3), 0.03, 20_000),
        "Adam 0.1": adam(grad, np.zeros(3), 0.1, 20_000),
    }
    for name, p in runs.items():
        gaps = np.array([f(q) - best for q in p])
        print(f"{name}: steps to 1e-3 {steps_to(p, f, best + 1e-3)}, "
              f"to 1e-6 {steps_to(p, f, best + 1e-6)}, final gap "
              f"{gaps[-1]:.2e}")
    g0 = grad(np.zeros(3))
    first = adam(grad, np.zeros(3), 0.03, 1)[1]
    print("gradient at 0", g0, "first Adam step", first)
    print("first GD step", -g0 / L)

    # --- Adam against torch.optim.Adam ---------------------------
    hours, months, y = ch05.churn_data()
    A = torch.tensor(np.column_stack([np.ones(200), hours, months]))
    yt = torch.tensor(y)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.03)
    for _ in range(500):
        opt.zero_grad()
        torch.nn.functional.softplus(-yt * (A @ w)).mean().backward()
        opt.step()
    ours = adam(grad, np.zeros(3), 0.03, 500)[-1]
    print("ours", ours, "torch", w.detach().numpy(), "difference",
          np.abs(ours - w.detach().numpy()).max())

    # --- backpropagation ------------------------------------------
    Z, yy = parts()
    Z = (Z - Z[:2000].mean(axis=0)) / Z[:2000].std(axis=0)
    Xn, yn, Xt, yt2 = Z[:2000], yy[:2000], Z[2000:], yy[2000:]
    theta = start()
    loss0, g = loss_and_grad(theta, Xn, yn)
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e[i] = 1e-6
        fd[i] = (loss_and_grad(theta + e, Xn, yn)[0]
                 - loss_and_grad(theta - e, Xn, yn)[0]) / 2e-6
    tt = torch.tensor(theta, requires_grad=True)
    W1, b1 = tt[:32].reshape(2, 16), tt[32:48]
    z = torch.tanh(torch.tensor(Xn) @ W1 + b1) @ tt[48:64] + tt[64]
    torch.nn.functional.softplus(-torch.tensor(yn) * z).mean().backward()
    print("parameters", theta.size, "loss", round(loss0, 4),
          "backprop vs differences", np.abs(g - fd).max(),
          "vs torch", np.abs(g - tt.grad.numpy()).max())
    t0 = time.perf_counter()
    for _ in range(200):
        loss_and_grad(theta, Xn, yn)
    t1 = time.perf_counter()
    for _ in range(200):
        np.mean(np.logaddexp(0, -yn * (np.tanh(Xn @ theta[:32].reshape(2, 16) + theta[32:48]) @ theta[48:64] + theta[64])))
    t2 = time.perf_counter()
    print("gradient costs", round((t1 - t0) / (t2 - t1), 2),
          "loss evaluations; central differences cost", 2 * theta.size)

    # --- the NumPy network trained with the NumPy Adam ----------
    path = adam(batch_gradient(Xn, yn), theta, 0.01, 60 * 32)
    acc = lambda th: np.mean(np.sign(np.tanh(Xt @ unp(th)[0] + unp(th)[1]) @ unp(th)[2] + unp(th)[3]) == yt2)
    from network import unpack as unp
    print("NumPy network: loss", round(loss_and_grad(path[-1], Xn, yn)[0], 4),
          "test accuracy", acc(path[-1]))

    # --- a straight line cannot do it ------------------------------
    torch.manual_seed(0)
    line = torch.nn.Linear(2, 1)
    train(line, torch.optim.Adam(line.parameters(), lr=0.01), epochs=30)
    print("linear model: test loss and accuracy", evaluate(line))
    model = network()
    print("network parameters", sum(p.numel() for p in model.parameters()))

    # --- the sweep over step sizes ---------------------------------
    steps = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    for name, make in OPTIMIZERS.items():
        row = []
        for t in steps:
            model = network()
            row.append(train(model, make(model.parameters(), t))[-1])
        print(name, [round(v, 3) for v in row])

    # --- ten random starts ------------------------------------------
    finals = []
    for seed in range(10):
        model = network(seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        finals.append((train(model, opt, seed=seed)[-1], evaluate(model)[1]))
    print("ten starts: losses", [round(a, 3) for a, _ in finals],
          "accuracies", [round(b, 3) for _, b in finals])

    # --- bad starts ---------------------------------------------------
    for value in (0.0, 0.5):
        model = network()
        for p in model.parameters():
            torch.nn.init.constant_(p, value)
        h = train(model, torch.optim.Adam(model.parameters(), lr=0.01))
        print("all weights", value, "final loss", round(h[-1], 4),
              "accuracy", evaluate(model)[1])

    # --- schedules -------------------------------------------------
    total = 60 * 32
    cosine = lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total))
    rules = {"constant": None, "cosine": cosine,
             "warm-up + cosine": lambda k: min((k + 1) / 160, cosine(k))}
    for t in (0.01, 0.1, 0.3):
        for name, rule in rules.items():
            out = []
            for seed in range(3):
                model = network(seed)
                opt = torch.optim.Adam(model.parameters(), lr=t)
                sch = (torch.optim.lr_scheduler.LambdaLR(opt, rule)
                       if rule else None)
                out.append(train(model, opt, schedule=sch, seed=seed)[-1])
            print(f"Adam {t} {name}: {[round(v, 3) for v in out]} "
                  f"mean {np.mean(out):.3f}")

    # --- weight decay with little data ----------------------------
    Xs, Ys = X_train[:150], Y_train[:150]
    for decay in (0.0, 0.03, 0.1, 0.3, 1.0):
        out = []
        for seed in range(3):
            model = network(seed, width=64)
            opt = torch.optim.AdamW(model.parameters(), lr=0.01,
                                    weight_decay=decay)
            h = train(model, opt, epochs=600, seed=seed, X=Xs, Y=Ys)
            size = math.sqrt(sum((p ** 2).sum().item()
                                 for p in model.parameters()))
            out.append((h[-1], *evaluate(model), size))
        print("AdamW decay", decay, "train loss %.3f, test loss %.3f, "
              "test accuracy %.3f, weight norm %.1f"
              % tuple(np.mean(out, axis=0)))
