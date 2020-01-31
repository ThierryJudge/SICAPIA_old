from torch.utils.data import Dataset, Subset
import numpy as np
import random
import copy
from typing import Union

class ActiveLearningDataset(Dataset):
    """
    Active learning dataset
    """
    def __init__(self, dataset: Dataset, initial_label_rate: float=0.2):
        """
        Args:
            dataset: torch.utils.data.Dataset, original dataset
            initial_label_rate: float, amount of the original dataset to be initially labeled
        """
        self.dataset = dataset
        self.initial_label_rate = initial_label_rate

        self.labeled = np.zeros(len(self.dataset))
        self.labeled_indicies = random.sample(range(len(dataset)), int(initial_label_rate * len(dataset)))
        self.labeled[self.labeled_indicies] = 1
        self.initial_labeled = copy.deepcopy(self.labeled)

        self.labeled_data_subset = Subset(self.dataset, np.where(self.labeled == 1)[0])

        self.create_pool()

    def __getitem__(self, index):
        return self.labeled_data_subset[index]

    def __len__(self) -> int:
        return len(self.labeled_data_subset)

    def create_pool(self):
        pool_subset = Subset(self.dataset, np.where(self.labeled == 0)[0])
        self.active_learning_pool = ActiveLearningPool(pool_subset)

    def get_pool(self):
        return self.active_learning_pool

    def reset(self):
        """
            Reset to initial labeled data
        """
        self.labeled = copy.deepcopy(self.initial_labeled)
        self.labeled_data_subset = Subset(self.dataset, np.where(self.labeled == 1)[0])
        self.create_pool()

    def label(self, indices: Union[list, int]):
        """
            Label samples indicated by indices
        Args:
            indices: list[int], sample indices to label relative to pool
        """
        unlabeled_indices = np.where(self.labeled == 0)[0]
        samples_to_label = unlabeled_indices[indices]
        self.labeled[samples_to_label] = 1
        self.create_pool()
        self.labeled_data_subset = Subset(self.dataset, np.where(self.labeled == 1)[0])


def remove_label(x):
    return x[0]


class ActiveLearningPool(Dataset):
    def __init__(self, dataset: Dataset, remove_label_fn=remove_label):
        self.dataset = dataset
        self.remove_label = remove_label_fn

    def __getitem__(self, index):
        return self.remove_label(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import os
    from torchvision import transforms
    from matplotlib import pyplot as plt

    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    al_dataset = ActiveLearningDataset(mnist)
    al_pool = al_dataset.get_pool()

    print(len(al_dataset))
    print(len(al_pool))

    al_dataset.label([10, 11])
    al_pool = al_dataset.get_pool()

    print(len(al_dataset))
    print(len(al_pool))

    img, gt = al_dataset[0]

    plt.figure()
    plt.imshow(img.squeeze())
    plt.show()

    img = al_pool[0]
    plt.figure()
    plt.imshow(img.squeeze())
    plt.show()
