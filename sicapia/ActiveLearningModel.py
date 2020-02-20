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
import glob
from sicapia.utils.path_utils import create_model_directories

class ActiveLearningModel(pl.LightningModule):

    def __init__(self, network: nn.Module, train_dataset: Dataset, test_dataset: Dataset, hparams: Namespace,
                 loss_fn: Callable, val_dataset: Dataset = None, metrics: Sequence[Callable] = None):
        """

        Args:
            network: nn.Module, network to train
            train_dataset: torch.utils.data.Dataset, training dataset
            val_dataset: torch.utils.data.Dataset, validation dataset
            test_dataset: torch.utils.data.Dataset, test dataset
            hparams: Namespace, hyperparams from argparser
            loss_fn: Callable, torch.nn.functional loss function
            metrics: Sequence[Callable], list of metrics to evaluate model
        """
        super(ActiveLearningModel, self).__init__()

        self.network = network
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.hparams = hparams

        self.name = self.hparams.name
        # If name is given, create directories, else use default Trainer path
        if self.name:
            self.name  = create_model_directories(self.name)

        self.metric_names = [m.__name__ for m in self.metrics]

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

    def update_datasets(self, train_dataset, val_dataset=None, test_dataset=None):
        """
            Update training dataset to allow active learning process. New dataset is passed once more labeled data is
            available.
            Optionally update validation and test dataset.
        Args:
            train_dataset: torch.utils.data.Dataset, training dataset
            val_dataset: torch.utils.data.Dataset, validation dataset
            test_dataset: torch.utils.data.Dataset, test dataset
        """
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

        loss = self.loss_fn(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        if self.metrics:
            metrics = self.compute_metrics(y_hat, y)
            progress_bar = metrics
            tensorboard_logs.update(metrics)

        return_dict =  {'loss': loss,
                        'log': tensorboard_logs}

        if self.metrics:
            return_dict.update(metrics)
            return_dict.update({'progress_bar': progress_bar})

        return return_dict

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)

        return_dict = {'val_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y, prefix='val_')
            return_dict.update(metrics)
        return return_dict

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss}

        if self.metrics:
            average_metrics = {}
            for name in self.metric_names:
                average_metrics['val_' + name] = torch.stack([x['val_' + name] for x in outputs]).mean()
            logs.update(average_metrics)

        return_dict =  {'avg_val_loss': avg_loss,
                        'progress_bar': logs,
                        'log': logs}
        if self.metrics:
            return_dict.update(average_metrics)

        return return_dict

    def testing_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)

        return_dict = {'test_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y, prefix='test_')
            return_dict.update(metrics)

        return return_dict

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss}

        if self.metrics:
            average_metrics = {}
            for name in self.metric_names:
                average_metrics['test_' + name] = torch.stack([x['test_' + name] for x in outputs]).mean()
            logs.update(average_metrics)

        return_dict = {'avg_test_loss': avg_loss,
                       'log': logs}
        if self.metrics:
            return_dict.update(average_metrics)

        return return_dict

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

    def save(self, path: str, checkpoint_path: str = None):
        if checkpoint_path:
            try:
                checkpoint_path = glob.glob(checkpoint_path)[0]
                print(checkpoint_path)
                self.load_weights(checkpoint_path)
            except:
                print("Failed to load checkpoint file, saving file with current weights")

        save_model_path = os.path.join(path, 'model.pkl')
        print("Saving model to {}".format(save_model_path))
        torch.save(self.network, save_model_path)

    def load_model(self, path):
        self.network = torch.load(path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['state_dict'])

    def train_model(self, trainer, reset=False):
        if reset:
            self.reset_parameters()
        try:
            trainer.fit(self)
        except KeyboardInterrupt:
            print("Training canceled")
        self.save(trainer.weights_save_path, checkpoint_path=trainer.weights_save_path)


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
        parser.add_argument('--epochs', default=5, type=int, help='Number of total epochs to run')
        parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='Initial learning rate', dest='lr')
        parser.add_argument('--name', default=None, type=str, help='Path to save model')
        return parser


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from sicapia.networks.LinearNet import LinearNet

    args = ArgumentParser(add_help=False)
    args = ActiveLearningModel.add_model_specific_args(args)
    params = args.parse_args()

    mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

    network = LinearNet(input_shape=(1, 28, 28), output_size=10, activation=None)
    metrics = [accuracy]
    model = ActiveLearningModel(network=network, train_dataset=mnist_train, val_dataset=mnist_val,
                                test_dataset=mnist_test, metrics=metrics, hparams=params, loss_fn=F.nll_loss)

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=2, default_save_path=model.name)
    print(trainer.default_save_path)

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

