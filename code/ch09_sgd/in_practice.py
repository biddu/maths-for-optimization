"""Chapter 9, in practice: the same training loop in PyTorch."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from from_scratch import subscribers

A, y = subscribers()
X = torch.tensor(A[:100_000, 1:], dtype=torch.float32)
Y = torch.tensor(y[:100_000], dtype=torch.float32)
torch.manual_seed(0)

model = torch.nn.Linear(20, 1)        # 20 weights and a bias
loss_fn = torch.nn.SoftMarginLoss()   # mean of log(1 + exp(-y z))
opt = torch.optim.SGD(model.parameters(), lr=1.0)
rule = lambda k: min(6.4 / (1 + 0.064 * k), 1.38)
schedule = torch.optim.lr_scheduler.LambdaLR(opt, rule)
batches = DataLoader(TensorDataset(X, Y), batch_size=64,
                     shuffle=True)

for epoch in range(3):
    for xb, yb in batches:
        opt.zero_grad()              # forget the last gradient
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()              # gradient of this batch
        opt.step()                   # x <- x - t * gradient
        schedule.step()              # shrink t
    with torch.no_grad():
        total = loss_fn(model(X).squeeze(1), Y).item()
    print("pass", epoch + 1, "loss", round(total, 5))
