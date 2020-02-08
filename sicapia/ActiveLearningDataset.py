from torch.utils.data import Dataset, Subset
import numpy as np
import random
import copy
from typing import Union
from torch.utils.data._utils.collate import default_collate
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

    def get_semi_supervised_dataset(self, batch_size=32):
        return SemiSupervisedLearningDataset(self.labeled_data_subset, self.active_learning_pool, batch_size=batch_size)


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


class SemiSupervisedLearningDataset(Dataset):
    def __init__(self, supervised_dataset: Dataset, unsupervised_dataset: Dataset, unsupervised_freq: int=3,
                 batch_size=32):
        """
        This dataset handles creating batches!
        !!!Must be used with Dataloader with batch_size and batch_sampler set to None!!!

        TODO
        Epoch must include all labeled data and un-labeled data every x batches according to supervised_rate
        or
        Epcoh is all the data
        ADD counter for non supervised data to repeating data

        Args:
            supervised_dataset: torch.utils.data.Dataset, labeled dataset
            unsupervised_dataset: torch.utils.data.Dataset, unlabeled dataset
            supervised_rate: float, rate of samples for labeled data with respect to unlabeled data
                                if None, use rate between data
        """

        self.supervised_dataset = supervised_dataset
        self.unsupervised_dataset = unsupervised_dataset
        self.unsupervised_freq = unsupervised_freq
        self.batch_size = batch_size

        self.unsupervised_dataset_len = int(np.ceil(len(self.unsupervised_dataset) / self.batch_size))
        self.supervised_dataset_len = int(np.ceil(len(self.supervised_dataset) / self.batch_size))

    def __getitem__(self, index):
        if index%self.unsupervised_freq==0:
            index = index % self.unsupervised_dataset_len

            batch_samples = range(index * self.batch_size, (
            (index + 1) * self.batch_size if (index + 1) * self.batch_size < len(self.unsupervised_dataset) else len(
                self.unsupervised_dataset)))

            return 1, default_collate([self.unsupervised_dataset[i] for i in batch_samples])
        else:
            index = index % self.supervised_dataset_len

            batch_samples = range(index * self.batch_size, (
            (index + 1) * self.batch_size if (index + 1) * self.batch_size < len(self.supervised_dataset) else len(
                self.supervised_dataset)))

            return 0, default_collate([self.supervised_dataset[i] for i in batch_samples])

    def __len__(self):
        return self.supervised_dataset_len + self.supervised_dataset_len//self.unsupervised_freq


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import os
    from torchvision import transforms
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader

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

    semi_dataset = al_dataset.get_semi_supervised_dataset()

    print(len(semi_dataset))

    semi_dataloader = DataLoader(semi_dataset, batch_size=None)
    sup_cout = 0
    unsup_cout = 0

    for batch in semi_dataloader:
        if batch[0] == 0:
            x, y = batch[1]
            print("Supervised")
            print(x.shape)
            print(y.shape)
            sup_cout += 1

        else:
            x = batch[1]
            print("Unsupervised")
            print(x.shape)
            unsup_cout += 1

    print(sup_cout)
    print(unsup_cout)