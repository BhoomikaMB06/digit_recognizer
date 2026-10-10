import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)


def true_fn(x):
    return 3 * x**2 + 2 * x + 5


# ---------- Generate data ----------
X = torch.rand(1000, 1) * 6 - 3                      # inputs between -3 and 3
y = true_fn(X) + torch.randn(1000, 1) * 0.5          # add a little noise

X_train, y_train = X[:700], y[:700]
X_val, y_val = X[700:850], y[700:850]
X_test, y_test = X[850:], y[850:]

# ---------- Model ----------
model = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Train ----------
train_losses, val_losses = [], []
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_losses.append(loss_fn(model(X_val), y_val).item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}")

# ---------- Test on unseen data ----------
model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(X_test), y_test).item()
print(f"Test MSE: {test_loss:.4f}")

new_x = torch.tensor([[-2.5], [0.0], [1.5], [2.8]])
with torch.no_grad():
    new_pred = model(new_x)
print("\nPredictions for unseen inputs:")
for xi, pi in zip(new_x, new_pred):
    print(f"x={xi.item():5.1f}  predicted={pi.item():7.3f}  true={true_fn(xi).item():7.3f}")

# ---------- Loss curve ----------
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE loss (log scale)")
plt.title("Loss curve")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

# ---------- Actual vs predicted ----------
xs, order = torch.sort(X_test[:, 0])
with torch.no_grad():
    pred_sorted = model(X_test)[order]
plt.figure(figsize=(6, 4))
plt.scatter(X_test.numpy(), y_test.numpy(), s=10, label="Actual (test data)")
plt.plot(xs.numpy(), pred_sorted.numpy(), color="red", label="Predicted")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Actual vs predicted: y = 3x² + 2x + 5")
plt.legend()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nSaved loss_curve.png and actual_vs_predicted.png")