import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output