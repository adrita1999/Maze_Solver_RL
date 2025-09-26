import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=5, activation = nn.ReLU,num_layers = 2):
        super(DQN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        print(type(layers))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
