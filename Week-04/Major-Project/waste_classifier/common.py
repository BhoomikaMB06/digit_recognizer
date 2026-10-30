"""Shared settings, preprocessing and models.

train.py AND app.py both import this file, so an image uploaded in the app gets exactly
the same preprocessing as the images used for validation and testing.
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageOps
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / 'models'
MODEL_FILES = {'baseline': 'baseline_cnn.pth', 'improved': 'improved_mobilenet.pth'}

IMG_SIZE = 160
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]     # the values the pretrained network expects

# Used for validation, testing and for every image uploaded in the app
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Used only to train the improved model: random changes so it sees a different version of each photo
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch: three convolution blocks and a small classifier."""

    def __init__(self, num_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))     # 160 -> 80 -> 40 -> 20
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 20 * 20, 64), nn.ReLU(),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def make_improved(num_classes, pretrained=True):
    """MobileNetV2 (pretrained on ImageNet) with a new output layer for our classes."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, num_classes))
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_model(kind, num_classes):
    """Load a saved model ('baseline' or 'improved') ready for predictions."""
    model = BaselineCNN(num_classes) if kind == 'baseline' else make_improved(num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / MODEL_FILES[kind], map_location='cpu'))
    model.eval()
    return model


def predict_probs(model, pil_image):
    """Apply the training preprocessing to a PIL image and return the score of every class."""
    img = ImageOps.exif_transpose(pil_image).convert('RGB')      # fixes rotated phone photos
    x = eval_transform(img).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0]