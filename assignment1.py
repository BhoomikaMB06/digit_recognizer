import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ---------- Part 2: Tensor operations ----------
print("=== Tensor operations ===")
a = torch.tensor([[1, 2, 3], [4, 5, 6]])       # creation, shape (2, 3)
b = torch.arange(12).reshape(3, 4)             # creation + reshaping, shape (3, 4)
z = torch.zeros(2, 2)
r = torch.rand(2, 3)
print("a:", a, "shape:", a.shape)
print("b:", b, "shape:", b.shape)
print("zeros:", z)
print("random:", r)

flat = b.reshape(12)                           # reshape to 1-D
back = flat.reshape(2, 6)                      # reshape to (2, 6)
print("flattened:", flat)
print("reshaped to (2, 6):", back)

print("element-wise a * 2:", a * 2)
print("matrix multiply a @ b:", a.float() @ b.float())   # (2,3) @ (3,4) -> (2,4)

print("b[1, 2] =", b[1, 2])                    # one element
print("b[:, 0] =", b[:, 0])                    # first column
print("b[0:2, 1:3] =", b[0:2, 1:3])            # slice

# ---------- Part 3: Synthetic regression data ----------
# True relationship: y = 3x + 2 + noise
X = torch.rand(1000, 1) * 2 - 1
y = 3 * X + 2 + torch.randn(1000, 1) * 0.3

X_train, y_train = X[:700], y[:700]
X_val, y_val = X[700:850], y[700:850]
X_test, y_test = X[850:], y[850:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)


def train(lr, epochs=50):
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_losses.append(total / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())

    with torch.no_grad():
        test_loss = loss_fn(model(X_test), y_test).item()
    return model, train_losses, val_losses, test_loss


# ---------- Parts 4 and 5: two learning rates, loss plots ----------
results = {}
for lr in [0.001, 0.1]:
    model, tl, vl, test_loss = train(lr)
    results[lr] = (model, tl, vl, test_loss)
    print(f"lr={lr}: final train={tl[-1]:.4f}, final val={vl[-1]:.4f}, test={test_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, lr in zip(axes, results):
    _, tl, vl, _ = results[lr]
    ax.plot(tl, label="Train")
    ax.plot(vl, label="Validation")
    ax.set_title(f"Learning rate = {lr}")
    ax.set_xlabel("Epoch")
    ax.legend()
axes[0].set_ylabel("MSE loss")
plt.tight_layout()
plt.savefig("assignment1_loss_curves.png")
plt.close()

# Actual vs predicted for the better learning rate (0.1)
model = results[0.1][0]
xs, order = torch.sort(X_test[:, 0])
with torch.no_grad():
    pred = model(X_test)[order]
plt.figure(figsize=(6, 4))
plt.scatter(X_test.numpy(), y_test.numpy(), s=10, label="Actual")
plt.plot(xs.numpy(), pred.numpy(), color="red", label="Predicted (lr=0.1)")
plt.legend()
plt.title("Actual vs predicted")
plt.savefig("assignment1_actual_vs_predicted.png")
plt.close()

print("Saved assignment1_loss_curves.png and assignment1_actual_vs_predicted.png")