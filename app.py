from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn

app = Flask(__name__)

# Same architecture as training
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
model.load_state_dict(torch.load('digit_model_lr0.001.pth', map_location='cpu'))
model.eval()


@app.route('/')
def home():
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    pixels = request.get_json().get('pixels', [])
    if len(pixels) != 784:
        return jsonify({'error': 'Expected 784 pixel values'}), 400

    x = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()

    return jsonify({'probs': probs, 'digit': int(max(range(10), key=lambda i: probs[i]))})


if __name__ == '__main__':
    app.run(debug=True)