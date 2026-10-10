# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

print(torch.__version__)

# ---- CHANGE THIS VALUE: run once with 0.001, then again with 0.1 ----
lr = 0.1

# %%
transform = transforms.ToTensor()
full_train = datasets.MNIST('data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('data', train=False, download=True, transform=transform)

train_set, val_set = random_split(full_train, [50000, 10000])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

# %%
images, labels = next(iter(train_loader))
plt.figure()
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(labels[i].item())
    plt.axis('off')
plt.savefig('sample_digits.png')
plt.close()

# %%
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# %%
epochs = 10
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    train_losses.append(total / len(train_loader))

    model.eval()
    vtotal = 0
    with torch.no_grad():
        for x, y in val_loader:
            vtotal += loss_fn(model(x), y).item()
    val_losses.append(vtotal / len(val_loader))

    print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}")

# %%
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.title(f'Loss Curve (lr={lr})')
plt.savefig(f'loss_curve_lr{lr}.png')
plt.close()

# %%
correct = total = 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Test accuracy (lr={lr}): {correct/total*100:.2f}%")

# %%
x, y = next(iter(test_loader))
with torch.no_grad():
    pred = model(x).argmax(dim=1)

plt.figure()
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(x[i][0], cmap='gray')
    plt.title(f"P:{pred[i].item()}\nA:{y[i].item()}")
    plt.axis('off')
plt.savefig(f'predictions_lr{lr}.png')
plt.close()

# %%
torch.save(model.state_dict(), f'digit_model_lr{lr}.pth')
print("Done. Graphs and model saved in the folder.")