# Mini Project 3: Fashion Item Recognition

A convolutional neural network that classifies clothing images into 10 categories (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot). Three models are compared, and the best one is saved.

Notebook with the code, graphs and analysis: [fashion_item_recognition.ipynb](fashion_item_recognition.ipynb)
## categories
![categories] (https://github.com/user-attachments/assets/dffd1555-c3a0-426d-be62-bffda4ea0be6
)

## Setup
- Data: Fashion-MNIST, 28x28 greyscale images, 50,000 train / 10,000 validation / 10,000 test
- Preprocessing: pixels normalised with mean 0.2860 and std 0.3530
- Augmentation (training only): random horizontal flip
- Loss: cross-entropy, optimizer: Adam (learning rate 0.001), 8 epochs, batch size 64

## Models compared
1. **Fully connected baseline:** 784 -> 128 -> 64 -> 10 with ReLU
2. **Small CNN:** two convolution blocks (16 and 32 filters, 3x3, padding 1, ReLU, 2x2 max pooling), then 64 -> 10
3. **Improved CNN:** two convolution blocks (32 and 64 filters) with batch normalisation, dropout 0.3, then 128 -> 10

## Results
| Model | Parameters | Val acc | Test acc | Train time (s) |
|---|---|---|---|---|
| Fully connected | 109,386 | __% | __% | __ |
| Small CNN | 105,866 | __% | __% | __ |
| Improved CNN | 421,834 | __% | __% | __ |

(Copy the markdown table printed by section 6 of the notebook.)

**Best model:** __ with a test accuracy of __%, which is __ points above the fully connected baseline.

## Training and validation curves
![Training curves](https://github.com/user-attachments/assets/1d3c75a9-85d3-425d-8e26-574ab37f1c02
)

## Confusion matrix
![Confusion matrix](https://github.com/user-attachments/assets/f1a9032f-9d24-4188-9864-18f8a424956b
)

- Lowest per-class accuracy: __ (__%)
- Most common mistakes: __ predicted as __ (__ times), __ predicted as __ (__ times)

## Prediction gallery
![Prediction gallery](https://github.com/user-attachments/assets/33ea2d40-9629-4f3a-a65b-23689eca7c9b
)

Green titles are correct predictions, red titles are mistakes.

## Error analysis
(Write 4 or 5 lines using your own results. For example: the model mixes up classes with similar outlines, such as Shirt, T-shirt/top, Pullover and Coat, because at 28x28 pixels only small details like collars and buttons tell them apart. Footwear classes (Sneaker, Sandal, Ankle boot) are confused when the shoe design is unusual. Some images are ambiguous even for a person.)

## Saved model
`fashion_cnn.pth` holds the weights of the improved CNN. To use it, create `ImprovedCNN()` from the notebook and call `load_state_dict(torch.load('fashion_cnn.pth'))`. The notebook reloads it and checks that the test accuracy matches.

## How to run
```
pip install torch torchvision matplotlib numpy pillow
```
Open `fashion_item_recognition.ipynb` and run all cells. Fashion-MNIST is downloaded automatically on the first run.
