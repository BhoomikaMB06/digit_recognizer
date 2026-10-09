import json
import torch

# Load the model you already trained (lr = 0.001 run)
state = torch.load('digit_model_lr0.001.pth', map_location='cpu')

def flat(name):
    return [round(v, 5) for v in state[name].flatten().tolist()]

weights = {
    'w1': flat('1.weight'), 'b1': flat('1.bias'),   # 784 -> 128
    'w2': flat('3.weight'), 'b2': flat('3.bias'),   # 128 -> 64
    'w3': flat('5.weight'), 'b3': flat('5.bias'),   # 64 -> 10
}

with open('weights.js', 'w') as f:
    f.write('const WEIGHTS = ' + json.dumps(weights) + ';')

print('Saved weights.js')