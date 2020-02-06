import os
from argparse import ArgumentParser

from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.networks.CNNNet import CNNNet
from sicapia.networks.LinearNet import LinearNet
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
from sicapia.utils.metrics import accuracy
from torch.nn import functional as F

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--lr', type=float, default=0.02)
    args.add_argument('--epochs', type=float, default=10)
    params = args.parse_args()

    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    net = LinearNet((1, 28, 28), 10)
    metrics = [accuracy]
    loss = F.nll_loss
    model = ActiveLearningModel(network=net, train_dataset=mnist, test_dataset=mnist_test, val_dataset=mnist_test,
                                loss_fn=loss, metrics=metrics, hparams=params)

    trainer = Trainer(min_epochs=params.epochs, max_epochs=params.epochs)
    model.train_model(trainer)
    train_results = model.evaluate_model(set='train')
    print(train_results)
    test_results = model.evaluate_model(set='test')
    print(test_results)