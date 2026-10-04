"""Chapter 10, in practice: one network, five optimizers."""
import torch

from data import parts

Z, y = parts()
Z = (Z - Z[:2000].mean(axis=0)) / Z[:2000].std(axis=0)
X = torch.tensor(Z, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)
X_train, Y_train = X[:2000], Y[:2000]
X_test, Y_test = X[2000:], Y[2000:]
loss_fn = torch.nn.SoftMarginLoss()


def network(seed=0, width=16):
    """Two inputs, two hidden layers of tanh units, one score."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1))


def train(model, opt, epochs=60, batch=64, schedule=None,
          seed=0, X=X_train, Y=Y_train):
    """The loop of Chapter 9; returns the loss after each pass."""
    shuffle = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(Y), generator=shuffle)
        for i in range(0, len(Y), batch):
            rows = order[i:i + batch]
            opt.zero_grad()
            loss_fn(model(X[rows]).squeeze(1), Y[rows]).backward()
            opt.step()
            if schedule:
                schedule.step()
        with torch.no_grad():
            history.append(loss_fn(model(X).squeeze(1), Y).item())
    return history


def evaluate(model, X=X_test, Y=Y_test):
    """Loss and fraction predicted correctly on held-back data."""
    with torch.no_grad():
        z = model(X).squeeze(1)
        right = ((z > 0) == (Y > 0)).float().mean().item()
        return loss_fn(z, Y).item(), right


OPTIMIZERS = {
    "SGD": lambda p, t: torch.optim.SGD(p, lr=t),
    "momentum": lambda p, t: torch.optim.SGD(p, lr=t,
                                             momentum=0.9),
    "RMSProp": lambda p, t: torch.optim.RMSprop(p, lr=t),
    "Adam": lambda p, t: torch.optim.Adam(p, lr=t),
}
BEST = {"SGD": 0.3, "momentum": 0.1, "RMSProp": 0.01,
        "Adam": 0.01}

if __name__ == "__main__":
    for name, make in OPTIMIZERS.items():
        model = network()
        opt = make(model.parameters(), BEST[name])
        h = train(model, opt)
        print(f"{name:9s} step {BEST[name]:5.2f}: loss after 5, "
              f"20, 60 passes {h[4]:.3f} {h[19]:.3f} {h[59]:.3f};"
              f" test accuracy {evaluate(model)[1]:.3f}")
