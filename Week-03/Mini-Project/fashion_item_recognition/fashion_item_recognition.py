# %% [markdown]
# # Week 3 — Mini Project 3
# ## Fashion Item Recognition
#
# **Goal:** Build a classifier for clothing categories (shirts, trousers, bags, footwear and more) using Fashion-MNIST. Three models are compared: a fully connected baseline, a small CNN, and an improved CNN with batch normalisation, dropout and data augmentation. The best model is saved for later use.

# %% [markdown]
# ## 1. Import libraries and load the data
# 50,000 images for training, 10,000 for validation, and the 10,000 test images for the final result only. Training images get a random horizontal flip as augmentation (clothes look the same when mirrored). All three models are trained on the same data.

# %%
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42)


def find_data_dir():
    """Return <repo root>/data, no matter which folder the code is started from."""
    here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / '.git').exists():
            return folder / 'data'
    return Path.cwd() / 'data'


DATA_DIR = find_data_dir()
print('Data folder:', DATA_DIR)

MEAN, STD = 0.2860, 0.3530
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                               transforms.Normalize((MEAN,), (STD,))])
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

train_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=train_tf)
eval_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=eval_tf)
test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=eval_tf)

perm = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
train_set = Subset(train_full, perm[:50000])
val_set = Subset(eval_full, perm[50000:])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)
print('Train:', len(train_set), '| Validation:', len(val_set), '| Test:', len(test_set))

# %% [markdown]
# ## 2. The 10 clothing categories

# %%
fig, axes = plt.subplots(1, 10, figsize=(16, 2.2))
for c, ax in enumerate(axes):
    idx = int((test_set.targets == c).nonzero()[0])
    ax.imshow(test_set.data[idx], cmap='gray')
    ax.set_title(CLASSES[c], fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Build three models
# 1. **Fully connected baseline** works on the flattened 784 pixels.
# 2. **Small CNN** has two convolution blocks with max pooling.
# 3. **Improved CNN** has wider filters (32 and 64), batch normalisation after each convolution, and dropout before the final layers to reduce overfitting.

# %%
def make_fc():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


fc, small, improved = make_fc(), SmallCNN(), ImprovedCNN()
for name, m in [('Fully connected', fc), ('Small CNN', small), ('Improved CNN', improved)]:
    print(f'{name:<16} parameters: {count_params(m):,}')

# %% [markdown]
# ## 4. Train all three models
# The improved CNN is the slowest, so this cell can take 5 to 10 minutes on a CPU.

# %%
loss_fn = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_model(model, epochs=8, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        v_loss, v_acc = evaluate(model, val_loader)
        hist['train_loss'].append(run_loss / n)
        hist['train_acc'].append(correct / n)
        hist['val_loss'].append(v_loss)
        hist['val_acc'].append(v_acc)
        print(f'  Epoch {epoch + 1}: train acc={correct / n * 100:.2f}%, val loss={v_loss:.4f}, val acc={v_acc * 100:.2f}%')
    hist['seconds'] = time.time() - start
    return hist


print('Training the fully connected baseline')
fc_hist = train_model(fc)
print('\nTraining the small CNN')
small_hist = train_model(small)
print('\nTraining the improved CNN')
imp_hist = train_model(improved)

# %% [markdown]
# ## 5. Training and validation curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, hist, color in [('Fully connected', fc_hist, 'tab:orange'),
                          ('Small CNN', small_hist, 'tab:blue'),
                          ('Improved CNN', imp_hist, 'tab:green')]:
    axes[0].plot(hist['train_loss'], color=color, label=f'{name} train')
    axes[0].plot(hist['val_loss'], color=color, linestyle='--', label=f'{name} validation')
    axes[1].plot([a * 100 for a in hist['train_acc']], color=color, label=f'{name} train')
    axes[1].plot([a * 100 for a in hist['val_acc']], color=color, linestyle='--', label=f'{name} validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss (solid = train, dashed = validation)')
axes[0].legend(fontsize=7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy')
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Comparison table

# %%
results = []
for name, model, hist in [('Fully connected', fc, fc_hist), ('Small CNN', small, small_hist),
                          ('Improved CNN', improved, imp_hist)]:
    _, test_acc = evaluate(model, test_loader)
    results.append((name, count_params(model), hist['val_acc'][-1], test_acc, hist['seconds']))

print(f"{'Model':<18}{'Parameters':>12}{'Val acc':>10}{'Test acc':>10}{'Train time (s)':>16}")
for name, params, val_acc, test_acc, secs in results:
    print(f'{name:<18}{params:>12,}{val_acc * 100:>9.2f}%{test_acc * 100:>9.2f}%{secs:>16.0f}')

print('\nMarkdown table for the README:')
print('| Model | Parameters | Val acc | Test acc | Train time (s) |')
print('|---|---|---|---|---|')
for name, params, val_acc, test_acc, secs in results:
    print(f'| {name} | {params:,} | {val_acc * 100:.2f}% | {test_acc * 100:.2f}% | {secs:.0f} |')

# %% [markdown]
# ## 7. Per-class accuracy and confusion matrix (improved CNN)
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()


# %%
def predict_all(model, loader):
    model.eval()
    preds, trues, probs_all, imgs = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x), dim=1)
            probs_all.append(p)
            preds.append(p.argmax(1))
            trues.append(y)
            imgs.append(x)
    return torch.cat(preds), torch.cat(trues), torch.cat(probs_all), torch.cat(imgs)


pred, true, probs, imgs = predict_all(improved, test_loader)

cm = np.zeros((10, 10), dtype=int)
for t, p in zip(true.tolist(), pred.tolist()):
    cm[t, p] += 1

print(f"{'Class':<14}{'Accuracy':>9}")
for i, name in enumerate(CLASSES):
    print(f'{name:<14}{cm[i, i] / cm[i].sum() * 100:>8.1f}%')

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(10), CLASSES, rotation=45, ha='right', fontsize=8)
plt.yticks(range(10), CLASSES, fontsize=8)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix (test set, improved CNN)')
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.show()

pairs = sorted([(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j], reverse=True)
print('Most common mistakes:')
for count, t, p in pairs[:5]:
    print(f'  true {CLASSES[t]} predicted as {CLASSES[p]}: {count} times')

# %% [markdown]
# ## 8. Prediction gallery
# 32 test images with the improved CNN's prediction and confidence. Green titles are correct, red titles are mistakes.
plt.savefig('prediction_gallery.png', dpi=120)
plt.show()
# %%
fig, axes = plt.subplots(4, 8, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i][0], cmap='gray')
    ok = pred[i] == true[i]
    ax.set_title(f'{CLASSES[pred[i].item()]}\n{probs[i][pred[i]].item() * 100:.0f}%',
                 color='green' if ok else 'red', fontsize=8)
    ax.axis('off')
plt.suptitle('Prediction gallery (green = correct, red = wrong)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Some wrong predictions

# %%
wrong = (pred != true).nonzero().flatten()
print(f'The improved CNN made {len(wrong)} mistakes on {len(true)} test images')

fig, axes = plt.subplots(2, 5, figsize=(13, 6))
for ax, idx in zip(axes.flat, wrong[:10].tolist()):
    ax.imshow(imgs[idx][0], cmap='gray')
    ax.set_title(f'True: {CLASSES[true[idx].item()]}\nPred: {CLASSES[pred[idx].item()]} '
                 f'({probs[idx][pred[idx]].item() * 100:.0f}%)', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Save the model and load it again
# Only the learned weights (the state dict) are saved. To use the model later, create `ImprovedCNN()` again and load the weights into it.

# %%
torch.save(improved.state_dict(), 'fashion_cnn.pth')

reloaded = ImprovedCNN()
reloaded.load_state_dict(torch.load('fashion_cnn.pth'))
reloaded.eval()
_, reloaded_acc = evaluate(reloaded, test_loader)
print(f'Saved fashion_cnn.pth. Reloaded model test accuracy: {reloaded_acc * 100:.2f}%')

# %% [markdown]
# ## 11. Predict a single image with the saved model
# `predict_image` takes any PIL image, turns it into a 28x28 greyscale image, and returns the top 3 predictions. Fashion-MNIST items are light on a dark background, so use `invert=True` for a dark item on a light background.

# %%
def predict_image(img, invert=False):
    img = img.convert('L').resize((28, 28))
    if invert:
        img = ImageOps.invert(img)
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(reloaded(x), dim=1)[0]
    top = p.topk(3)
    return [(CLASSES[i], round(v * 100, 1)) for v, i in zip(top.values.tolist(), top.indices.tolist())]


sample = Image.fromarray(test_set.data[0].numpy())
print('True label:', CLASSES[test_set.targets[0].item()])
print('Top 3 predictions:', predict_image(sample))

# %% [markdown]
# ## 12. Summary and error analysis
# (Write 5 or 6 lines using your own results above.)
#
# - **Best model:** which of the three models had the highest test accuracy, and by how much it beat the fully connected baseline.
# - **Why CNNs work better here:** convolution filters look at small neighbourhoods and reuse the same weights across the whole image, so they detect shapes such as sleeves, straps and soles anywhere in the picture.
# - **What the improvements did:** batch normalisation, dropout and flip augmentation reduce overfitting (compare the train and validation curves).
# - **Hardest classes:** name the classes with the lowest accuracy and the most common mistakes from the confusion matrix, for example Shirt confused with T-shirt/top and Pullover confused with Coat, because they have similar outlines at 28x28 pixels.
# - **Next steps:** more epochs, a deeper CNN, a learning rate scheduler, or a pre-trained model.