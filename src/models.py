#
# Define models here
#

import torch.nn as nn
import torch

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # -> (32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), # -> (64, 24, 24)
            nn.ReLU(),
        )
        # compute the output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            n_features = self.conv_layers(dummy).numel()
            print(f"Flatten size: {n_features}")

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
