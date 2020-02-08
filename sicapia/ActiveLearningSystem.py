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
        ActiveLearning system manages the active learning process of a ActiveLearningModel trained on an
        ActiveLearningDataset using an ActiveLearningStrategy.

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
        self.name = "AL_" + self.strategy.__class__.__name__ if path is None else path
        self.name,  = create_model_directories(self.name)

    def active_learning_loop(self, iterations=10):
        self.al_dataset.reset()
        self.model.reset_parameters()
        for i in range(iterations):
            self.trainer = Trainer(max_nb_epochs=self.model.hparams.epochs, default_save_path=self.name)
            print("Active Learning iteration: {} of {}".format(i, iterations))
            print("Training on {} samples: ".format(len(self.al_dataset)))
            self.model.train_model(self.trainer)

            test_results = self.model.evaluate_model()

            print("Test results on {} samples:".format(len(self.al_dataset)), end='')
            for k in test_results.keys():
                print("{}: {}, ".format(k, test_results[k]), end='')
                self.history[k].append(test_results[k])
            print()
            self.history['num_samples'].append(len(self.al_dataset))

            ind = self.strategy.get_samples_indicies(self.al_dataset.get_pool(), self.model, 100)
            self.al_dataset.label(ind)

        for k in self.history.keys():
            if k != 'num_samples':
                plt.figure()
                plt.title(k)
                plt.plot(self.history['num_samples'], self.history[k])
                plot_path = pjoin(self.name, '{}.jpg'.format(k))
                plt.savefig(plot_path)
        plt.show()


        history = pandas.DataFrame(self.history)
        history.to_csv(pjoin(self.name,'{}.csv'.format(self.strategy.__class__.__name__)))