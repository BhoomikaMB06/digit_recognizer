# %% [markdown]
# # Week 3 — Assignment 3
# ## Build and Evaluate a CNN on Fashion-MNIST
#
# **Goal:** Load and visualise Fashion-MNIST, build a small CNN, compare it with a fully connected baseline, plot training and validation curves, and explain five misclassified examples.

# %% [markdown]
# ## 1. Import libraries and load the data
# Fashion-MNIST has 70,000 greyscale images (28x28 pixels) of 10 clothing categories. The 60,000 training images are split into 50,000 for training and 10,000 for validation. The 10,000 test images are used only for the final result.
#
# **Normalisation:** each pixel is shifted by the dataset mean (0.2860) and divided by its standard deviation (0.3530), so the inputs are centred around 0, which helps training.

# %%
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib

IN_NOTEBOOK = 'ipykernel' in sys.modules      # True for notebook / Run Cell
if not IN_NOTEBOOK:
    matplotlib.use('Agg')                      # no pop-up windows when run as a plain file
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)


def find_data_dir():
    """Return <repo root>/data, no matter which folder the code is started from."""
    here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / '.git').exists():
            return folder / 'data'
    return Path.cwd() / 'data'


DATA_DIR = find_data_dir()
FIG_DIR = Path(__file__).resolve().parent / 'figures' if '__file__' in globals() else Path.cwd() / 'figures'
FIG_DIR.mkdir(exist_ok=True)
print('Data folder:', DATA_DIR)


def show_fig(name):
    """Save the current figure as a PNG; show it inline in a notebook."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{name}.png', dpi=120)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()
        print('Saved figure:', FIG_DIR / f'{name}.png')


CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])

full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
train_set, val_set = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)
print('Train:', len(train_set), '| Validation:', len(val_set), '| Test:', len(test_set))

# %% [markdown]
# ## 2. Visualise the images
# Each image is a tensor with shape (1, 28, 28): 1 colour channel, 28 rows and 28 columns.

# %%
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, lab in zip(axes.flat, images, labels):
    ax.imshow(img[0], cmap='gray')
    ax.set_title(CLASSES[lab.item()], fontsize=8)
    ax.axis('off')
show_fig('01_sample_images')

print('One image tensor has shape', tuple(images[0].shape), '(channels, height, width)')
print('A batch has shape', tuple(images.shape), '(batch, channels, height, width)')
print('Pixel range after normalisation:', round(images.min().item(), 2), 'to', round(images.max().item(), 2))

# %% [markdown]
# ## 3. Build the models
# **Fully connected baseline:** flattens the image into 784 numbers and uses two hidden layers. It ignores where pixels are in the image.
#
# **Small CNN:** two convolution blocks (3x3 filters with padding, ReLU, 2x2 max pooling) shrink the image from 28x28 to 14x14 to 7x7 while learning edge and shape filters. A small fully connected head then predicts the class.

# %%
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def make_fc():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


fc_model = make_fc()
cnn_model = SmallCNN()
print('Fully connected parameters:', f'{count_params(fc_model):,}')
print('CNN parameters:            ', f'{count_params(cnn_model):,}')

# %% [markdown]
# ## 4. Train both models

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
fc_hist = train_model(fc_model)
print('\nTraining the CNN')
cnn_hist = train_model(cnn_model)

# %% [markdown]
# ## 5. Training and validation curves
# Solid lines are training, dashed lines are validation.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, hist, color in [('Fully connected', fc_hist, 'tab:orange'), ('CNN', cnn_hist, 'tab:blue')]:
    axes[0].plot(hist['train_loss'], color=color, label=f'{name} train')
    axes[0].plot(hist['val_loss'], color=color, linestyle='--', label=f'{name} validation')
    axes[1].plot([a * 100 for a in hist['train_acc']], color=color, label=f'{name} train')
    axes[1].plot([a * 100 for a in hist['val_acc']], color=color, linestyle='--', label=f'{name} validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy')
axes[1].legend()
show_fig('02_training_curves')

# %% [markdown]
# ## 6. Comparison table (CNN vs fully connected baseline)

# %%
results = []
for name, model, hist in [('Fully connected', fc_model, fc_hist), ('Small CNN', cnn_model, cnn_hist)]:
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
# ## 7. Five misclassified examples

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


pred, true, probs, imgs = predict_all(cnn_model, test_loader)
wrong = (pred != true).nonzero().flatten()
print(f'The CNN made {len(wrong)} mistakes on {len(true)} test images')

chosen = wrong[:5].tolist()
fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
for ax, idx in zip(axes, chosen):
    ax.imshow(imgs[idx][0], cmap='gray')
    ax.set_title(f'True: {CLASSES[true[idx].item()]}\nPred: {CLASSES[pred[idx].item()]} '
                 f'({probs[idx][pred[idx]].item() * 100:.0f}%)', fontsize=8)
    ax.axis('off')
show_fig('03_misclassified')

for k, idx in enumerate(chosen, 1):
    top2 = probs[idx].topk(2)
    guesses = ', '.join(f'{CLASSES[i]} {p * 100:.1f}%' for p, i in zip(top2.values.tolist(), top2.indices.tolist()))
    print(f'Example {k}: true = {CLASSES[true[idx].item()]}, predicted = {CLASSES[pred[idx].item()]}')
    print(f'   top-2 probabilities: {guesses}')

# %% [markdown]
# ## 8. Explanation of the misclassified examples
# Replace the text below with your own explanation of the five examples printed above. Use the true label, the predicted label, and the top-2 probabilities for each one.
#
# **Example 1:** (true class, predicted class, and why they look alike)
#
# **Example 2:** ...
#
# **Example 3:** ...
#
# **Example 4:** ...
#
# **Example 5:** ...
#
# **Common reasons to mention:**
# - Shirt, T-shirt/top, Pullover and Coat have similar outlines, and at 28x28 pixels only small details such as buttons or collars tell them apart.
# - Sneaker, Sandal and Ankle boot share a similar shape, and low-contrast or unusual designs look like another class.
# - Some images are ambiguous even for a person, or the label itself is debatable.
# - The model was trained for only 8 epochs on small images with no augmentation.

# %% [markdown]
# ## 9. Conclusion
# The small CNN performed better than the fully connected baseline, reaching ___% test accuracy compared with ___% for the baseline, an improvement of about ___ percentage points. The CNN also did this with ___ parameters against ___ for the baseline. A CNN suits images because it reuses the same small filters across the whole image, so it detects edges and shapes wherever they appear and keeps the spatial layout of the pixels. A fully connected network flattens the image and loses that structure. The CNN's remaining mistakes were mostly between similar classes such as Shirt, T-shirt/top, Pullover and Coat, and more epochs, data augmentation or a deeper CNN could reduce them.
