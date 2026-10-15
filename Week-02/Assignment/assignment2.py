import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ---------- 1. Synthetic two-class dataset (two moons: NOT linearly separable) ----------
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


def tensors(a, b):
    return torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32).reshape(-1, 1)


X_train, y_train = tensors(X_train, y_train)
X_val, y_val = tensors(X_val, y_val)
X_test, y_test = tensors(X_test, y_test)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")


# ---------- 2. Models and metrics ----------
def make_model(kind, hidden=0):
    if kind == "linear":
        return nn.Linear(2, 1)                      # logistic regression: a straight line
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def predict(model, X):
    with torch.no_grad():
        return (torch.sigmoid(model(X)) > 0.5).float()


def accuracy(model, X, y):
    return (predict(model, X) == y).float().mean().item()


def precision_recall(model, X, y):
    pred = predict(model, X)
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


loss_fn = nn.BCEWithLogitsLoss()


def train(kind, hidden, lr, epochs=300):
    torch.manual_seed(42)                            # same start for a fair comparison
    model = make_model(kind, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())
    return model, train_losses, val_losses


# ---------- 3. Experiments: linear model vs neural network, hidden size and learning rate ----------
LRS = [0.001, 0.01, 0.1]
HIDDEN_SIZES = [2, 4, 16, 64]

runs = []
for lr in LRS:
    model, tl, vl = train("linear", 0, lr)
    runs.append(dict(name="Linear", hidden=0, lr=lr, model=model, tl=tl, vl=vl))
for hidden in HIDDEN_SIZES:
    for lr in LRS:
        model, tl, vl = train("nn", hidden, lr)
        runs.append(dict(name="Neural net", hidden=hidden, lr=lr, model=model, tl=tl, vl=vl))

for r in runs:
    r["train_acc"] = accuracy(r["model"], X_train, y_train)
    r["val_acc"] = accuracy(r["model"], X_val, y_val)
    r["test_acc"] = accuracy(r["model"], X_test, y_test)
    r["val_loss"] = r["vl"][-1]

# ---------- 4. Comparison table ----------
print(f"\n{'Model':<12}{'Hidden':>7}{'LR':>8}{'Train acc':>11}{'Val acc':>9}{'Test acc':>10}")
for r in runs:
    print(f"{r['name']:<12}{r['hidden']:>7}{r['lr']:>8}{r['train_acc']:>11.3f}{r['val_acc']:>9.3f}{r['test_acc']:>10.3f}")

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "hidden_size", "learning_rate", "train_acc", "val_acc", "test_acc"])
    for r in runs:
        writer.writerow([r["name"], r["hidden"], r["lr"], round(r["train_acc"], 4),
                         round(r["val_acc"], 4), round(r["test_acc"], 4)])

# ---------- 5. Pick the best of each type using VALIDATION data, then report TEST data ----------
def best(name):
    return max((r for r in runs if r["name"] == name), key=lambda r: (r["val_acc"], -r["val_loss"]))


best_linear, best_nn = best("Linear"), best("Neural net")
for label, r in [("Best linear", best_linear), ("Best neural net", best_nn)]:
    p, rec = precision_recall(r["model"], X_test, y_test)
    print(f"\n{label}: hidden={r['hidden']}, lr={r['lr']}")
    print(f"  val acc={r['val_acc']:.3f}, test acc={r['test_acc']:.3f}, "
          f"test precision={p:.3f}, test recall={rec:.3f}")

# ---------- 6. Graphs ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, r, title in [(axes[0], best_linear, "Linear model"), (axes[1], best_nn, "Neural network")]:
    ax.plot(r["tl"], label="Train")
    ax.plot(r["vl"], label="Validation")
    ax.set_title(f"{title} (lr={r['lr']})")
    ax.set_xlabel("Epoch")
    ax.legend()
axes[0].set_ylabel("BCE loss")
plt.tight_layout()
plt.savefig("assignment2_loss_curves.png")
plt.close()


def plot_boundary(model, ax, title):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    ax.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(), c=y_test.flatten().numpy(),
               cmap="RdBu", edgecolors="k", s=20)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_boundary(best_linear["model"], axes[0], f"Linear model (test acc {best_linear['test_acc']:.2f})")
plot_boundary(best_nn["model"], axes[1], f"Neural network, {best_nn['hidden']} hidden (test acc {best_nn['test_acc']:.2f})")
plt.tight_layout()
plt.savefig("assignment2_decision_boundaries.png")
plt.close()

print("\nSaved results.csv, assignment2_loss_curves.png and assignment2_decision_boundaries.png")