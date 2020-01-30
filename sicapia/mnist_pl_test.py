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
from sicapia.ActiveLearningDataset import ActiveLearningDataset

class MNISTExample(pl.LightningModule):

    def __init__(self, hparams):
        super(MNISTExample, self).__init__()
        self.hparams = hparams
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # out_channels x feat_map_w x feat_map_h
        num_features = 64*12*12
        self.dense_1 = torch.nn.Linear(in_features=num_features, out_features=128)
        self.dense_2 = torch.nn.Linear(in_features=self.hparams.final_dim, out_features=10)

        self.dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        self.al_dataset = ActiveLearningDataset(self.dataset)

    def forward(self, x):
        # feature extractor
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, 0.25)

        # flatten
        x = x.view(x.size(0), -1)

        # classifier
        x = F.relu(self.dense_1(x))
        x = F.dropout(x, 0.50)
        x = self.dense_2(x)
        return x

    def training_step(self, batch, batch_nb):
        # print(batch)
        # print(batch_nb)
        # print(batch[0])
        # print(batch[1])
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.data.max(1, keepdim=True)[1]
        accuracy = pred.eq(y.data.view_as(pred)).sum().double() / len(x)
        tensorboard_logs = {'train_loss': loss}
        progress_bar = {'acc': accuracy}
        return {'loss': loss,
                'log': tensorboard_logs,
                'progress_bar': progress_bar}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        logits = F.log_softmax(y_hat, dim=1)
        preds = torch.topk(logits, dim=1, k=1)[1].view(-1)
        accuracy = accuracy_score(y,  preds)

        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss,
                'accuracy': torch.tensor(accuracy)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss, 'val_acc': accuracy}
        return {'avg_val_loss': avg_loss,
                'progress_bar': logs,
                'log': logs}

    def testing_step(self, batch, batch_nb):
        # in this case do the same thing as val step
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': accuracy}
        return {'avg_val_loss': avg_loss,
                'progress_bar': logs,
                'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.al_dataset, batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
                          batch_size=32)

    def train(self):
        trainer = Trainer()
        trainer.fit(model)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--final_dim', type=int, default=128)
    args.add_argument('--lr', type=float, default=0.02)
    params = args.parse_args()

    model = MNISTExample(params)

    # most basic trainer, uses good defaults
    trainer = Trainer()
    trainer.fit(model)