# Mini Project 1: Numerical Pattern Predictor

A neural network that learns the relationship y = 3x² + 2x + 5 from noisy generated data and predicts outputs for inputs it has never seen.

## Setup
- Data: 1000 points, x between -3 and 3, noise standard deviation 0.5
- Split: 700 train, 150 validation, 150 test
- Model: 1 -> 64 -> 64 -> 1 with ReLU activations
- Loss: MSE, optimizer: Adam (learning rate 0.01), 500 epochs

## Results
**Test MSE: 0.2522**

Predictions for inputs the model never saw:

| x    | Predicted | True   | Error |
|------|-----------|--------|-------|
| -2.5 | 18.905    | 18.750 | 0.155 |
| 0.0  | 5.014     | 5.000  | 0.014 |
| 1.5  | 14.540    | 14.750 | 0.210 |
| 2.8  | 34.001    | 34.120 | 0.119 |

## Loss curve
![Loss curve](https://github.com/user-attachments/assets/cf594eee-3510-4d80-a37f-675f401c718a
)

## Actual vs predicted
![Actual vs predicted](https://github.com/user-attachments/assets/e5430396-9d2b-4c5b-9308-de036396b6fb
)

## Explanation
The model learned the curved relationship well. The test MSE of 0.2522 is almost equal to the noise variance in the data (0.5² = 0.25), which is the lowest error any model can reach, because random noise cannot be predicted. This means the network fits the true pattern y = 3x² + 2x + 5 almost perfectly. The red prediction curve follows the parabola through the test points, and the four unseen inputs were predicted within about 0.2 of their true values. The training and validation losses stayed close together in the loss curve, so there was no sign of overfitting. The ReLU hidden layers were enough to model the curve, even though the relationship is not a straight line.
