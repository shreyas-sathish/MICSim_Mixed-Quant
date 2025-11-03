#!/usr/bin/env python3
"""
Tiny test for layer-wise LSQ quantization + scaling on a 3x3 input.

Run: python script_example/accuracy/cnn-toy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

os.environ["CONFIG"] = "./Accuracy/config/resnet18/config_toy.ini"

#fixing the seed values
torch.manual_seed(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# make sure repo root is on PYTHONPATH if you run from repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import your modified layers / quantizer
from Accuracy.src.Layers.QLayer.CNN.QLinear_custom import QLinear
from Accuracy.src.Layers.QLayer.CNN.QConv2d_custom import QConv2d
from Accuracy.src.Modules.CNN.Quantizer.LSQuantizer import LSQuantizer  # modified LSQuantizer.py

# Tiny network using your quantized layers
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv: in=1, out=2, kernel 2x2 -> on 3x3 input gives 2x2 output
        self.conv1 = QConv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0, bias=False, name='conv1')
        # conv2: in=2, out=2, kernel 2x2 -> reduces to 1x1 per map
        self.conv2 = QConv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0, bias=False, name='conv2')
        # flatten then fc: input features = 2*1*1 = 2 (batch dim later)
        self.fc = QLinear(in_channels=2, out_channels=2, bias=True, name='fc')

        # attach quantizer overrides manually for the test (simulate apply_layerwise_config)
        # conv1: 8-bit weights, 8-bit activations, scale 0.25
        self.conv1.quantizer.layer_weight_precision = 8
        self.conv1.quantizer.layer_input_precision = 8
        self.conv1.quantizer.layer_scaling = 0.25
        setattr(self.conv1, 'scaling', 0.25)

        # conv2: 4-bit weights, 8-bit activations, scale 0.5
        self.conv2.quantizer.layer_weight_precision = 4
        self.conv2.quantizer.layer_input_precision = 8
        self.conv2.quantizer.layer_scaling = 0.5
        setattr(self.conv2, 'scaling', 0.5)

        # fc: 3-bit weights, 6-bit activations, scale 0.8
        self.fc.quantizer.layer_weight_precision = 3
        self.fc.quantizer.layer_input_precision = 6
        self.fc.quantizer.layer_scaling = 0.8
        setattr(self.fc, 'scaling', 0.8)

    def forward(self, x):
        print("Input:", x)
        x = self.conv1(x)
        print("After conv1:", x)
        x = F.relu(x)
        x = self.conv2(x)
        print("After conv2:", x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print("After fc:", x)
        return x

if __name__ == "__main__":
    # 1 sample, 1 channel, 3x3
    x = torch.tensor([[[[0.1, 0.5, -0.2],
                        [0.4, -0.3, 0.2],
                        [0.6, -0.1, -0.4]]]], dtype=torch.float32)

    net = TinyNet()
    net.eval()
    with torch.no_grad():
        out = net(x)
    print("Final:", out)