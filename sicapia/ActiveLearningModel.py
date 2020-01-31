from sicapia.ActiveLearningDataset import ActiveLearningDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math

class ActiveLearningModel:
    # model handels training, must be given network, loss, and metrics
    # aswell as other hyperparatms : learning rate, batch_size, optimizer...

    def __init__(self, net:nn.Module):
        self.net = net
        self.batch_size=32
        self.learning_rate=0.1

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        torch.save(self.net.state_dict(), 'initial_weights')
        print(self.net.parameters())

    def reset_parameters(self):
        # torch.manual_seed(10)
        # for m in self.net.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        #         if m.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #             bound = 1 / math.sqrt(fan_in)
        #             nn.init.uniform_(m.bias, -bound, bound)
        # self.net = Net()
        self.net.load_state_dict(torch.load('initial_weights'))

    def train(self, train_dataset:ActiveLearningDataset, val_dataset=None, epochs=10, device='cpu', verbose=True,
              reset=False):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        if val_dataset:
            val_dataloader = Dataset(val_dataset, batch_size=self.batch_size)
        if reset:
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
                print('Train Epoch: {}: \tLoss: {:.6f}'.format(epoch, loss_mean))

    def eval(self, test_dataset:Dataset, device='cpu'):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
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
