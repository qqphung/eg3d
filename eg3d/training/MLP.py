from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(216, 300 ),
            nn.ReLU(),
            nn.Linear(300, 512)  
        )

    def forward(self, x):
        return self.layers(x)
