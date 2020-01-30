from torchvision.datasets import MNIST
from torchvision import transforms
import os
from PIL import Image
import torch

class Subset():
    """
    Args:
        train: id of the training subset inside the HDF5 file
        valid: id of the validation subset inside the HDF5 file
        test: id of the testing subset inside the HDF5 file
        pool:
    """
    train = 'train'
    valid = 'valid'
    test = 'test'
    pool = 'pool'


class MnistActiveLearning(MNIST):
    def __init__(self, root, label_rate=1.0, train=True, download=True, transform=None, target_transform=None):
        super().__init__(root=root, train=train,transform=transform, target_transform=target_transform,
                 download=download)

        self.train_data, self.train_targets = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.test_data, self.test_targets = torch.load(os.path.join(self.processed_folder, self.training_file))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get(self):
        return self



if __name__ == '__main__':
    mnist = MnistSet(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()).get()
    print(mnist[0])
