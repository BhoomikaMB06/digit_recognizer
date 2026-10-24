"""Recreate the confusion matrix and wrong-prediction images from the saved model (no retraining)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
MEAN, STD = 0.2860, 0.3530


def find_data_dir():
    for folder in [HERE, *HERE.parents]:
        if (folder / '.git').exists():
            return folder / 'data'
    return Path.cwd() / 'data'


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


model_path = HERE / 'fashion_cnn.pth'
if not model_path.exists():
    raise SystemExit(f'Model file not found: {model_path}')

eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
test_set = datasets.FashionMNIST(find_data_dir(), train=False, download=True, transform=eval_tf)
loader = DataLoader(test_set, batch_size=256)

model = ImprovedCNN()
model.load_state_dict(torch.load(model_path))
model.eval()

preds, trues, confs, imgs = [], [], [], []
with torch.no_grad():
    for x, y in loader:
        p = torch.softmax(model(x), dim=1)
        c, k = p.max(1)
        preds.append(k)
        trues.append(y)
        confs.append(c)
        imgs.append(x)
pred, true, conf, imgs = torch.cat(preds), torch.cat(trues), torch.cat(confs), torch.cat(imgs)
print(f'Test accuracy: {(pred == true).float().mean().item() * 100:.2f}%')

# ---------- confusion matrix ----------
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(true.tolist(), pred.tolist()):
    cm[t, p] += 1

print(f"\n{'Class':<14}{'Accuracy':>9}")
for i, name in enumerate(CLASSES):
    print(f'{name:<14}{cm[i, i] / cm[i].sum() * 100:>8.1f}%')

pairs = sorted([(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j], reverse=True)
print('\nMost common mistakes:')
for count, t, p in pairs[:5]:
    print(f'  true {CLASSES[t]} predicted as {CLASSES[p]}: {count} times')

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
fig.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(10))
ax.set_yticklabels(CLASSES, fontsize=8)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (test set, improved CNN)')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
fig.tight_layout()
fig.savefig(HERE / 'confusion_matrix.png', dpi=120)
plt.close(fig)

# ---------- wrong predictions ----------
wrong = (pred != true).nonzero().flatten().tolist()[:10]
fig, axes = plt.subplots(2, 5, figsize=(13, 6))
for ax, idx in zip(axes.flat, wrong):
    ax.imshow(imgs[idx][0].numpy(), cmap='gray')
    ax.set_title(f'True: {CLASSES[true[idx].item()]}\nPred: {CLASSES[pred[idx].item()]} '
                 f'({conf[idx].item() * 100:.0f}%)', fontsize=8)
    ax.axis('off')
fig.tight_layout()
fig.savefig(HERE / 'wrong_predictions.png', dpi=120)
plt.close(fig)

print('\nSaved confusion_matrix.png and wrong_predictions.png in', HERE)