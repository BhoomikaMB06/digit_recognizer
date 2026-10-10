# Assignment 1: Understanding and Training a Neural Network

## 1. Key concepts
- **Neuron:** the basic unit of a network. It multiplies each input by a weight, adds a bias, and passes the result through an activation function to produce an output.
- **Weights:** numbers that decide how much each input matters. The network learns them during training.
- **Bias:** a constant added to a neuron's sum. It shifts the output so the neuron can fit data even when the inputs are zero.
- **Epoch:** one full pass through the whole training set. Training for 50 epochs means the model sees every training sample 50 times.
- **Batch:** a small group of samples processed together before one weight update. With 700 training samples and batch size 32, one epoch has 22 updates.
- **Learning rate:** the step size of each weight update, in w = w - lr x gradient. Too small and learning is very slow. Too large and the updates overshoot and may not converge.

## 2. Tensor operations
Done in `assignment1.py`: creation (`tensor`, `arange`, `zeros`, `rand`), reshaping (`reshape`), element-wise and matrix multiplication (`*`, `@`), and indexing and slicing (`b[1, 2]`, `b[:, 0]`, `b[0:2, 1:3]`).

## 3. Model and data
- Synthetic data: y = 3x + 2 + noise, 1000 samples.
- Split: 700 train, 150 validation, 150 test.
- Model: Linear(1, 16), ReLU, Linear(16, 1). Loss: MSE. Optimizer: SGD. 50 epochs, batch size 32.

## 4. Loss curves
![Loss curves](https://github.com/user-attachments/assets/6440e86b-a5a6-4fb4-b632-f041c9b4123d)

## 5. Learning rate comparison
| Learning rate | Final train loss | Final val loss | Test loss |
|---------------|------------------|----------------|-----------|
| 0.001         | 0.1025           | 0.1177         | 0.0830    |
| 0.1           | 0.0865           | 0.0999         | 0.0821    |

**Conclusion:** The data was generated as y = 3x + 2 with noise of standard deviation 0.3, so the lowest achievable MSE is about 0.09. Both learning rates reached close to this level after 50 epochs, and lr = 0.1 finished slightly lower (val 0.0999 vs 0.1177). In the loss curves, lr = 0.1 converged within the first few epochs, while lr = 0.001 improved more slowly. This shows that the learning rate mainly controls how fast training converges. On this simple problem the small rate caught up given enough epochs, but a larger rate saves training time. A rate that is too large can overshoot, as seen in the digit recognizer, where lr = 0.1 with Adam failed.

![Actual vs predicted](https://github.com/user-attachments/assets/faee60d1-ab42-4868-83fe-6a328995f804)
)
