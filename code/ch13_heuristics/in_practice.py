"""Chapter 13, in practice: library tools and a real black box."""
import json
import pathlib
import sys

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch10_adam"))
from data import parts  # noqa: E402  (the machine parts of Chapter 10)


def cities(n, seed):
    """n random stops in a 100 km square, and their distances."""
    P = np.random.default_rng(seed).uniform(0, 100, (n, 2))
    return P, np.linalg.norm(P[:, None] - P[None], axis=2)


def route(D, seconds=10):
    """A good tour from the OR-Tools routing library."""
    n = len(D)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    model = pywrapcp.RoutingModel(manager)
    meters = lambda i, j: int(round(1000 * D[manager.IndexToNode(i),
                                             manager.IndexToNode(j)]))
    model.SetArcCostEvaluatorOfAllVehicles(
        model.RegisterTransitCallback(meters))
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search.time_limit.seconds = seconds
    plan = model.SolveWithParameters(search)
    tour, index = [], model.Start(0)
    while not model.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = plan.Value(model.NextVar(index))
    return np.array(tour)


def exact_tour(D, seconds=60):
    """The exact method of Chapter 12: value, bound and status."""
    n = len(D)
    model = cp_model.CpModel()
    arcs, cost = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                used = model.NewBoolVar(f"a{i}_{j}")
                arcs.append((i, j, used))
                cost.append(int(round(1000 * D[i, j])) * used)
    model.AddCircuit(arcs)
    model.Minimize(sum(cost))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return (solver.ObjectiveValue() / 1000,
            solver.BestObjectiveBound() / 1000,
            solver.WallTime(), solver.StatusName(status))


# --- A black box: train a network, report its validation loss ---
Z, LABEL = parts()
Z = (Z - Z[:120].mean(axis=0)) / Z[:120].std(axis=0)
X_TRAIN = torch.tensor(Z[:120], dtype=torch.float32)
Y_TRAIN = torch.tensor(LABEL[:120], dtype=torch.float32)
X_VALID = torch.tensor(Z[1000:2000], dtype=torch.float32)
Y_VALID = torch.tensor(LABEL[1000:2000], dtype=torch.float32)
CACHE_FILE = HERE / "hp_cache.json"
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
BOX_LOW, BOX_HIGH = [-4.0, -3.0], [-0.5, 1.0]


def validation_loss(setting, seed=0):
    """Train with step 10**setting[0] and weight decay
    10**setting[1]; return the loss on 1000 held-back parts."""
    key = f"{setting[0]:.6f},{setting[1]:.6f},{seed}"
    if key in CACHE:
        return CACHE[key]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=10.0 ** setting[0],
                            weight_decay=10.0 ** setting[1])
    loss_fn = torch.nn.SoftMarginLoss()
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(200):
        order = torch.randperm(120, generator=shuffle)
        for i in range(0, 120, 32):
            rows = order[i:i + 32]
            opt.zero_grad()
            loss_fn(net(X_TRAIN[rows]).squeeze(1),
                    Y_TRAIN[rows]).backward()
            opt.step()
    with torch.no_grad():
        value = loss_fn(net(X_VALID).squeeze(1), Y_VALID).item()
    value = min(value, 2.0) if np.isfinite(value) else 2.0
    CACHE[key] = value
    return value


def save_cache():
    CACHE_FILE.write_text(json.dumps(CACHE))
