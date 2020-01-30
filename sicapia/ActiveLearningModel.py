from sicapia.ActiveLearningDataset import ActiveLearningDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class ActiveLearningModel:
    # model handels training, must be given network, loss, and metrics
    # aswell as other hyperparatms : learning rate, batch_size, optimizer...

    def __init__(self, net:nn.Module=Net()):
        self.net = net
        self.batch_size=32
        self.learning_rate=0.1

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        print(self.net.parameters())

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, train_dataset:ActiveLearningDataset, val_dataset=None, epochs=10, device='cpu', verbose=True):
        train_loader = DataLoader(train_dataset, batch_size=32)
        self.reset_parameters()
        self.net.train()
        for epoch in range(epochs):
            loss_mean = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = F.nll_loss(output, target)
                loss_mean += loss.item()
                loss.backward()
                self.optimizer.step()
            if verbose:
                loss_mean /= len(train_dataset.dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_mean))

    def eval(self, test_dataset:Dataset, device='cpu'):
        test_loader = DataLoader(test_dataset, batch_size=32)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        return {'loss': test_loss,
                'accuracy': accuracy}
