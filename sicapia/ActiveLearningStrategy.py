import random
import numpy as np
import torch
import scipy.stats


class ActiveLearningStrategy:
    def __init__(self, max):
        self.max = max

    def get_uncertainty(self, sample, model):
        raise NotImplementedError

    def get_samples_indicies(self, pool_dataset, model, n_samples=100):
        uncertainties = []
        for sample in pool_dataset:
            uncertainties.append(self.get_uncertainty(sample, model))

        uncertainties = np.array(uncertainties)
        pool_indicies = np.arange(len(pool_dataset))
        pool_indicies = pool_indicies[uncertainties.argsort()]

        if self.max:
            return pool_indicies[len(pool_dataset) - n_samples:]  # get highest uncertainties
        else:
            return pool_indicies[:n_samples]  # get lowest uncertainties


class RandomStrategy(ActiveLearningStrategy):
    def __init__(self):
        super().__init__(max=True)

    def get_uncertainty(self, sample, model):
        return random.uniform(0, 1)


class ConfidenceSamplingStrategy(ActiveLearningStrategy):
    """
    Get lowest probable prediction
    """

    def __init__(self):
        super().__init__(max=False)

    def get_uncertainty(self, sample, model):
        model_pred = model.net(sample)
        return torch.max(model_pred).detach().numpy()


class MarginSamplingStrategy(ActiveLearningStrategy):
    """
    Get smallest difference between two most probable predictions
    """

    def __init__(self):
        super().__init__(max=False)

    def get_uncertainty(self, sample, model):
        model_pred = model.net(sample).detach().numpy()
        sort_predictions = np.sort(model_pred, axis=1)
        return float(sort_predictions[:, -1] - sort_predictions[:, -2])


class EntropySamplingStrategy(ActiveLearningStrategy):
    """
    Get smallest difference between two most probable predictions
    """

    def __init__(self):
        super().__init__(max=False)

    def get_uncertainty(self, sample, model):
        model_pred = model.net(sample).detach().numpy().flatten()
        entropy = scipy.stats.entropy(model_pred)
        return entropy
