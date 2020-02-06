import os
from argparse import ArgumentParser, Namespace
from typing import List, Callable, Sequence

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset
import math
from sicapia.utils.metrics import *
import argparse


class ActiveLearningModel(pl.LightningModule):

    def __init__(self, network: nn.Module, train_dataset: Dataset, test_dataset: Dataset, hparams: Namespace,
                 loss_fn: Callable, val_dataset: Dataset = None, metrics: Sequence[Callable] = None):
        """

        Args:
            network: nn.Module, network to train
            train_dataset: torch.utils.data.Dataset, training dataset
            val_dataset: torch.utils.data.Dataset, validation dataset
            test_dataset: torch.utils.data.Dataset, test dataset
            hparams:
            loss_fn: callable, torch.nn.functional loss function
            metrics: list(callable), list of metrics to evaluate model
        """
        super(ActiveLearningModel, self).__init__()

        self.network = network
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.hparams = hparams

    def reset_parameters(self):
        # torch.manual_seed(10)
        for m in self.network.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        # self.net = Net()
        # self.net.load_state_dict(torch.load('initial_weights'))

    def update_datasets(self, train_dataset, val_dataset=None, test_dataset=None):
        self.train_dataset = train_dataset
        if val_dataset:
            self.val_dataset = val_dataset
        if test_dataset:
            self.test_dataset = test_dataset

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        metrics = self.compute_metrics(y_hat, y)

        tensorboard_logs = {'train_loss': loss}
        progress_bar = metrics
        return_dict =  {'loss': loss,
                'log': tensorboard_logs,
                'progress_bar': progress_bar}
        return_dict.update(metrics)
        return return_dict

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        return_dict = {'val_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y)
            return_dict.update(metrics)
        return return_dict

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss, 'val_acc': accuracy}
        return {'avg_val_loss': avg_loss,
                'progress_bar': logs,
                'log': logs}

    def testing_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        return_dict = {'test_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y, prefix='test_')
            return_dict.update(metrics)

        return return_dict

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()

        return {'avg_test_loss': avg_loss, 'avg_test_acc': accuracy}

    def compute_metrics(self, y_hat, y, prefix='', to_tensor=True):
        metrics = {}
        for metric in self.metrics:
            m = metric(y_hat, y)
            if to_tensor:
                m = torch.tensor(m)
            metrics[prefix+metric.__name__] = m

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Intentionally omit @pl.data_loader to reload train_dataloader before every training run
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    def train_model(self, trainer, reset=False):
        if reset:
            self.reset_parameters()
        trainer.fit(self)

    def evaluate_model(self, set='test'):
        if set == 'train':
            loader = self.train_dataloader()
        elif set == 'test':
            loader = self.test_dataloader()[0]
        elif set == 'val':
            if not self.val_dataset:
                raise ValueError("No validation dataset")
            loader = self.val_dataloader()[0]
        else:
            raise ValueError('Set must be one of \'train\', \'val\', \'test\'. Got \'{}\''.format(set))

        metrics = defaultdict(float)

        self.network.eval()
        loss = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x, y = batch
                y_hat = self.network(x)
                loss += self.loss_fn(y_hat, y).item()
                m = self.compute_metrics(y_hat, y, to_tensor=False)
                for k in m.keys():
                    metrics[k] += m[k]

        metrics.update({'loss': loss})
        for k in metrics.keys():
            metrics[k] /= len(loader)

        return dict(metrics)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-b', '--batch-size', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        return parser


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from sicapia.networks.LinearNet import LinearNet
    from sicapia.networks.CNNNet import CNNNet

    args = ArgumentParser()
    args = ActiveLearningModel.add_model_specific_args(args)
    params = args.parse_args()

    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_val_1 = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    network = LinearNet(input_shape=(1, 28, 28), output_size=10, activation=None)
    metrics = [accuracy]
    model = ActiveLearningModel(network=network, train_dataset=mnist_train, val_dataset=mnist_val,
                                test_dataset=mnist_test, metrics=metrics, hparams=params, loss_fn=F.nll_loss)

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=1)

    # model.train_model(trainer)
    #
    # print(len(model.train_dataset))
    # print(len(model.train_dataloader()))
    #
    #
    # model.update_datasets(mnist_train)
    #
    # print(len(model.train_dataset))
    # print(len(model.train_dataloader()))


    model.train_model(trainer)
    print(model.evaluate_model())
    model.reset_parameters()
    print(model.evaluate_model())

