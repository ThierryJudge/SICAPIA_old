import argparse
import os
from argparse import Namespace
from typing import Callable, Sequence

from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sicapia.ActiveLearningDataset import ActiveLearningDataset
from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.utils.metrics import *


class SemiSupervisedLearningModel(ActiveLearningModel):

    def __init__(self, network: nn.Module, train_dataset: ActiveLearningDataset, test_dataset: Dataset,
                 hparams: Namespace, loss_fn: Callable, val_dataset: Dataset = None,
                 metrics: Sequence[Callable] = None):
        """

        Args:
            network: nn.Module, network to train
            train_dataset: ActiveLearningDataset, training dataset, must allow get_semi_supervised_dataset
            val_dataset: torch.utils.data.Dataset, validation dataset
            test_dataset: torch.utils.data.Dataset, test dataset
            hparams: Namespace, hyperparams from argparser
            loss_fn: Callable, torch.nn.functional loss function
            metrics: Sequence[Callable], list of metrics to evaluate model
        """

        super().__init__(network=network, train_dataset=train_dataset, test_dataset=test_dataset, hparams=hparams,
                         loss_fn=loss_fn, val_dataset=val_dataset, metrics=metrics)

    def training_step(self, batch, batch_nb):
        batch_type, batch = batch
        if batch_type == 0:
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

        elif batch_type == 1:
            print("unsupervised")
            return {'loss':torch.tensor(0.0, requires_grad=True)}

    # Intentionally omit @pl.data_loader to reload train_dataloader before every training run
    def train_dataloader(self):
        return DataLoader(self.train_dataset.get_semi_supervised_dataset(batch_size=self.hparams.batch_size), batch_size=None)




if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from sicapia.networks.LinearNet import LinearNet

    parent_parser = argparse.ArgumentParser(add_help=False)
    args = SemiSupervisedLearningModel.add_model_specific_args(parent_parser)
    params = args.parse_args()

    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    al_dataset = ActiveLearningDataset(mnist_train, initial_label_rate=0.1)

    network = LinearNet(input_shape=(1, 28, 28), output_size=10, activation=None)
    metrics = [accuracy]
    model = SemiSupervisedLearningModel(network=network, train_dataset=al_dataset, val_dataset=mnist_val,
                                test_dataset=mnist_test, metrics=metrics, hparams=params, loss_fn=F.nll_loss)

    trainer = Trainer(max_nb_epochs=10)

    model.train_model(trainer)
    print(model.evaluate_model())


