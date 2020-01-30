from sicapia.ActiveLearningDataset import ActiveLearningDataset
from sicapia.ActiveLearningStrategy import ActiveLearningStrategy, RandomStrategy, ConfidenceSamplingStrategy, MarginSamplingStrategy
from torch.utils.data import Dataset
from sicapia.ActiveLearningModel import ActiveLearningModel
from collections import defaultdict
from matplotlib import pyplot as plt
import json


class ActiveLearningSystem:
    def __init__(self, al_dataset:ActiveLearningDataset, test_dataset:Dataset,
                 model: ActiveLearningModel, strategy:ActiveLearningStrategy, val_dataset:Dataset=None,
                 active_learning_loops=10):

        self.al_dataset = al_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.strategy = strategy

        self.history = defaultdict(list)

    def active_learning_loop(self, itterations=10):
        for i in range(itterations):
            print("Active Learning itteration: {} of {}".format(i, itterations))
            print("Training on {} samples: ".format(len(self.al_dataset)))
            self.model.train(self.al_dataset, self.val_dataset, epochs=10)

            test_results = self.model.eval(self.test_dataset)

            print("Test results on {} samples:".format(len(self.al_dataset)), end='')
            for k in test_results.keys():
                print("{}: {}, ".format(k, test_results[k]), end='')
                self.history[k].append(test_results[k])
            print()

            ind = self.strategy.get_samples_indicies(self.al_dataset.get_pool(), self.model, 100)
            self.al_dataset.label(ind)

        for k in self.history.keys():
            plt.figure()
            plt.title(k)
            plt.plot(self.history[k])

        plt.show()
        json.dump(self.history, open('{}.json'.format(self.strategy.__class__.__name__), 'w'))


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import os
    from torchvision import transforms
    from sicapia.ActiveLearningStrategy import *

    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    model = ActiveLearningModel()


    al_dataset = ActiveLearningDataset(mnist, initial_label_rate=0.001)

    al_strategy = EntropySamplingStrategy()

    al_system = ActiveLearningSystem(al_dataset=al_dataset, test_dataset=mnist_test, model=model, strategy=al_strategy)

    al_system.active_learning_loop()