import json
from collections import defaultdict

from matplotlib import pyplot as plt
from sicapia.ActiveLearningDataset import ActiveLearningDataset
from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.ActiveLearningStrategy import ActiveLearningStrategy
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from sicapia.utils.path_utils import create_model_directories
from os.path import join as pjoin
import pandas


class ActiveLearningSystem:
    def __init__(self, al_dataset: ActiveLearningDataset, test_dataset: Dataset,
                 model: ActiveLearningModel, strategy: ActiveLearningStrategy, val_dataset: Dataset = None,
                 path=None):

        """

        Args:
            al_dataset: ActiveLearningDataset, dataset for training
            test_dataset: torch.utils.data.Dataset, dataset for testing
            model:
            strategy:
            val_dataset: torch.utils.data.Dataset, dataset for validation
        """
        self.al_dataset = al_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.strategy = strategy

        self.history = defaultdict(list)
        self.name = self.__class__.__name__ if path is None else path
        self.name,  = create_model_directories(self.name)

    def active_learning_loop(self, iterations=5):
        self.al_dataset.reset()
        self.model.reset_parameters()
        for i in range(iterations):
            self.trainer = Trainer(max_epochs=self.model.hparams.epochs, default_save_path=self.name)
            print("Active Learning iteration: {} of {}".format(i, iterations))
            print("Training on {} samples: ".format(len(self.al_dataset)))
            self.model.train_model(self.trainer)

            test_results = self.model.evaluate_model()

            print("Test results on {} samples:".format(len(self.al_dataset)), end='')
            for k in test_results.keys():
                print("{}: {}, ".format(k, test_results[k]), end='')
                self.history[k].append(test_results[k])
            print()
            self.history['num_samples'] = len(self.al_dataset)

            ind = self.strategy.get_samples_indicies(self.al_dataset.get_pool(), self.model, 100)
            self.al_dataset.label(ind)

        for k in self.history.keys():
            plt.figure()
            plt.title(k)
            plt.plot(self.history[k])


        history = pandas.DataFrame(self.history)
        history.to_csv(pjoin(self.name, 'history.csv'), sep='\t')
        plt.show()
        history_path = pjoin(self.name, '{}.json'.format(self.strategy.__class__.__name__))
        plot_path = pjoin(self.name, '{}.jpg'.format(self.strategy.__class__.__name__))
        plt.savefig(plot_path)
        json.dump(self.history, open(history_path, 'w'))


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import os
    from torchvision import transforms
    from sicapia.ActiveLearningStrategy import *
    from sicapia.networks.CNNNet import CNNNet
    import argparse
    from torch.nn import functional as F
    from sicapia.utils.metrics import accuracy

    parent_parser = argparse.ArgumentParser(add_help=False)
    args = ActiveLearningModel.add_model_specific_args(parent_parser)
    params = args.parse_args()

    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    al_dataset = ActiveLearningDataset(mnist, initial_label_rate=0.001)

    net = CNNNet((1, 28, 28), 10)
    metrics = [accuracy]
    model = ActiveLearningModel(network=net, train_dataset=al_dataset, val_dataset=mnist_test, test_dataset=mnist_test,
                                hparams=params, loss_fn=F.nll_loss, metrics=metrics)

    al_strategy = ConfidenceSamplingStrategy()

    al_system = ActiveLearningSystem(al_dataset=al_dataset, test_dataset=mnist_test, model=model,
                                     strategy=al_strategy)

    al_system.active_learning_loop()
