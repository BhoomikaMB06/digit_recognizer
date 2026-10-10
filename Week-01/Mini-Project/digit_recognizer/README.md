# Digit Recognizer (MNIST)

A neural network built with PyTorch that recognizes handwritten digits (0-9), with a Flask web app where you draw a digit and get a live prediction.

## Model
784 -> 128 -> 64 -> 10, ReLU activations, Adam optimizer, CrossEntropyLoss, 10 epochs, batch size 64.

## Data split
50,000 train / 10,000 validation / 10,000 test (MNIST).

## Sample digits
![Sample digits](https://github.com/user-attachments/assets/00a13af5-214e-4b65-8590-88fa800adeff)
)

## Results
| Learning rate | Final train loss | Final val loss | Test accuracy |
|---------------|------------------|----------------|---------------|
| 0.001         | 0.0210           | 0.0918         | 97.70%        |
| 0.1           | 1.9955           | 1.9314         | 19.63%        |

### Loss curve (lr = 0.001)
![Loss curve lr 0.001](https://raw.githubusercontent.com/BhoomikaMB06/digit_recognizer/main/loss_curve_lr0.001.png)
)

### Loss curve (lr = 0.1)
![Loss curve lr 0.1](https://raw.githubusercontent.com/BhoomikaMB06/digit_recognizer/main/loss_curve_lr0.1.png)

)

### Predictions (lr = 0.001)
![Predictions lr 0.001](https://raw.githubusercontent.com/BhoomikaMB06/digit_recognizer/main/predictions_lr0.001.png)
)

### Predictions (lr = 0.1)
![Predictions lr 0.1](https://raw.githubusercontent.com/BhoomikaMB06/digit_recognizer/main/predictions_lr0.1.png)
)

## Observations
- **Overfitting (lr = 0.001):** Mild. Training loss kept falling to 0.0210, while validation loss stopped improving around epoch 8 (0.0825) and fluctuated afterward.
- **Effect of learning rate:** lr = 0.001 trained smoothly and reached 97.70% accuracy. lr = 0.1 was too large, so the updates overshot, the loss stayed near 2.0 (close to random guessing at about 2.30), and accuracy dropped to 19.63%.
- **Conclusion:** a learning rate that is too high stops the model from converging. About 0.001 works well for Adam on this problem.
- **Possible improvements:** fewer epochs (about 8), dropout, early stopping, a learning rate scheduler, or a CNN.

## Run the web app

```
pip install -r requirements.txt
python digit_recognizer1.py
python app.py
```

1. `pip install -r requirements.txt` installs the libraries.
2. `python digit_recognizer1.py` trains the model (lr = 0.001) and saves `digit_model_lr0.001.pth`.
3. `python app.py` starts the server. Open http://127.0.0.1:5000 and draw a digit.

## Project files
| File | Purpose |
|------|---------|
| `digit_recognizer1.py` | Trains the model and saves graphs |
| `app.py` | Flask server that loads the model and predicts |
| `index.html` | Drawing page for the web app |
| `requirements.txt` | Python libraries needed |
