import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LinearNet(nn.Module):
    def __init__(self, input_shape, output_size, activation=F.log_softmax):
        super(LinearNet, self).__init__()
        self.activation = activation
        self.fc = nn.Linear(int(np.prod(input_shape)), output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.activation:
            x = self.activation(x, dim=1)
        return x