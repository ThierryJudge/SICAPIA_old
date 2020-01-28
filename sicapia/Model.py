
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer

from argparse import ArgumentParser

import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

    def forward(self, x):
      pass

    def training_step(self, batch, batch_nb):
       pass

    def validation_step(self, batch, batch_nb):
       pass

    def validation_end(self, outputs):
        pass

    def testing_step(self, batch, batch_nb):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        pass

    @pl.data_loader
    def train_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass

