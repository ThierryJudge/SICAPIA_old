import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import numpy as np
from torchsummary import summary
from sicapia.utils.compute_output import compute_output_shape

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class CNNNet(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), output_size=10):
        super(CNNNet, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        conv_output_size = compute_output_shape(self.conv_net, input_shape)

        self.fc_net = nn.Sequential(
            Flatten(),
            nn.Linear(int(np.prod(conv_output_size)), 128),
            nn.Dropout(0.25),
            nn.Linear(128, output_size)
        )


    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        output = F.log_softmax(x, dim=1)
        return output

# def conv_output_shape(chw, filters=1, kernel_size=1, stride=1, pad=0, dilation=1):
#     if type(kernel_size) is not tuple:
#         kernel_size = (kernel_size, kernel_size)
#     h = floor(((chw[1] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
#     w = floor(((chw[2] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
#     chw = np.array([filters, h, w])
#     print(chw)
#     return chw

if __name__ == '__main__':
    from torchsummary import summary

    input_shape = (1, 24, 24)
    cnn = CNNNet(input_shape=input_shape)
    print(cnn)
    summary(cnn, input_shape)
