import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict

from sicapia.utils.metrics import *


class ActiveLearningModel(pl.LightningModule):

    def __init__(self, network, train_dataset, val_dataset, test_dataset, hparams=None, loss_fn=F.nll_loss, metrics=None):
        super(ActiveLearningModel, self).__init__()
        self.network = network
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.hparams = hparams

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

        metrics = self.compute_metrics(y_hat, y)

        return_dict = {'val_loss': loss}
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

        metrics = self.compute_metrics(y_hat, y, prefix='test_')

        return_dict = {'test_loss': loss}
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
        return DataLoader(self.train_dataset, batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

    def train_model(self, trainer):
        trainer.fit(self)

    def eval_model(self, test=False):
        if test:
            loader = self.test_dataloader()[0]
        else:
            loader = self.val_dataloader()[0]

        metrics = defaultdict(float)

        self.network.eval()
        loss = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x, y = batch
                y_hat = self.network(x)
                loss += self.loss_fn(y_hat, y).item()  # sum up batch loss

                m = self.compute_metrics(y_hat, y, to_tensor=False)
                for k in m.keys():
                    metrics[k] += m[k]

        metrics.update({'loss': loss})
        for k in metrics.keys():
            metrics[k] /= len(loader)

        return dict(metrics)


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from sicapia.networks.LinearNet import LinearNet
    from sicapia.networks.CNNNet import CNNNet

    args = ArgumentParser()
    args.add_argument('--final_dim', type=int, default=128)
    args.add_argument('--lr', type=float, default=0.02)
    params = args.parse_args()

    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_val_1 = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    network = LinearNet(input_shape=(1, 28, 28), output_size=10, activation=None)
    metrics = [accuracy]
    model = ActiveLearningModel(network=network, train_dataset=mnist_train, val_dataset=mnist_val,
                                test_dataset=mnist_test, metrics=metrics, hparams=params)

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
    print(model.eval_model())

