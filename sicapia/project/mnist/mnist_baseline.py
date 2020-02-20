import os
import argparse

from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.networks.CNNNet import CNNNet
from sicapia.networks.LinearNet import LinearNet
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
from sicapia.utils.metrics import accuracy
from torch.nn import functional as F

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    args = ActiveLearningModel.add_model_specific_args(parent_parser)
    params = args.parse_args()

    mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

    net = LinearNet((1, 28, 28), 10)
    metrics = [accuracy]
    loss = F.nll_loss
    model = ActiveLearningModel(network=net, train_dataset=mnist_train, test_dataset=mnist_test, val_dataset=mnist_test,
                                loss_fn=loss, metrics=metrics, hparams=params)

    trainer = Trainer(min_nb_epochs=params.epochs, max_nb_epochs=params.epochs, default_save_path=model.name)
    model.train_model(trainer)
    train_results = model.evaluate_model(set='train')
    print("Train evaluation results: {}".format((train_results)))
    test_results = model.evaluate_model(set='test')
    print("Test evaluation results: {}".format((test_results)))