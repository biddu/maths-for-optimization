"""Chapter 12, in practice: one schedule, two solvers."""
import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp

# Each job is a list of (machine, hours), in the order required.
# Machines: 0 saw, 1 assembly bench, 2 paint booth.
JOBS = [[(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)],
        [(0, 2), (1, 3), (2, 1)]]


def random_shop(jobs=10, machines=5, seed=12):
    """A larger shop: every job visits every machine once."""
    rng = np.random.default_rng(seed)
    return [[(int(m), int(rng.integers(1, 10)))
             for m in rng.permutation(machines)]
            for _ in range(jobs)]


def schedule_cp(jobs, seconds=60.0):
    """Shortest schedule by constraint programming (CP-SAT)."""
    horizon = sum(h for job in jobs for _, h in job)
    model = cp_model.CpModel()
    on_machine, ends, starts = {}, [], {}
    for j, job in enumerate(jobs):
        previous_end = 0
        for k, (machine, hours) in enumerate(job):
            start = model.NewIntVar(0, horizon, f"s{j}_{k}")
            end = model.NewIntVar(0, horizon, f"e{j}_{k}")
            task = model.NewIntervalVar(start, hours, end,
                                        f"t{j}_{k}")
            model.Add(start >= previous_end)   # order in the job
            on_machine.setdefault(machine, []).append(task)
            starts[j, k], previous_end = start, end
        ends.append(previous_end)
    for tasks in on_machine.values():
        model.AddNoOverlap(tasks)              # one job at a time
    finish = model.NewIntVar(0, horizon, "finish")
    model.AddMaxEquality(finish, ends)
    model.Minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    plan = {key: solver.Value(var) for key, var in starts.items()}
    return (solver.ObjectiveValue(), solver.BestObjectiveBound(),
            solver.WallTime(), plan, solver.StatusName(status))


def schedule_milp(jobs, seconds=60.0):
    """The same problem as a mixed-integer linear program."""
    ops = [(j, k) for j, job in enumerate(jobs)
           for k in range(len(job))]
    index = {op: i for i, op in enumerate(ops)}
    hours = {(j, k): jobs[j][k][1] for j, k in ops}
    big = float(sum(hours.values()))
    pairs = [(a, b) for a in ops for b in ops if a < b
             and jobs[a[0]][a[1]][0] == jobs[b[0]][b[1]][0]]
    n = len(ops) + 1 + len(pairs)        # starts, finish, switches
    rows, low, high = [], [], []

    def add(terms, lo, hi):
        row = np.zeros(n)
        for col, coef in terms:
            row[col] = coef
        rows.append(row); low.append(lo); high.append(hi)

    finish = len(ops)
    for j, job in enumerate(jobs):
        for k in range(1, len(job)):         # order within the job
            add([(index[j, k], 1), (index[j, k - 1], -1)],
                hours[j, k - 1], np.inf)
        last = len(job) - 1
        add([(finish, 1), (index[j, last], -1)], hours[j, last],
            np.inf)
    for p, (a, b) in enumerate(pairs):       # a first, or b first
        y = finish + 1 + p                   # y = 1 means a first
        add([(index[b], 1), (index[a], -1), (y, -big)],
            hours[a] - big, np.inf)
        add([(index[a], 1), (index[b], -1), (y, big)],
            hours[b], np.inf)
    cost = np.zeros(n)
    cost[finish] = 1
    whole = np.zeros(n)
    whole[finish + 1:] = 1
    upper = np.full(n, big)
    upper[finish + 1:] = 1
    rules = LinearConstraint(np.array(rows), low, high)
    return milp(cost, constraints=rules, integrality=whole,
                bounds=Bounds(0, upper),
                options={"time_limit": seconds})


if __name__ == "__main__":
    value, bound, secs, plan, status = schedule_cp(JOBS)
    print("CP-SAT:", value, bound, round(secs, 3), status, plan)
    res = schedule_milp(JOBS)
    print("MILP:", res.fun, res.mip_dual_bound, res.mip_node_count)
