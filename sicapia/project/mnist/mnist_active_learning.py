from torchvision.datasets import MNIST
import os
from torchvision import transforms
from sicapia.ActiveLearningStrategy import *
from sicapia.networks.CNNNet import CNNNet
from sicapia.ActiveLearningModel_vanila import ActiveLearningModel
from sicapia.ActiveLearningDataset import ActiveLearningDataset
from sicapia.ActiveLearningSystem import ActiveLearningSystem

if __name__ == '__main__':
    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    net = CNNNet((1, 28, 28), 10)
    model = ActiveLearningModel(net=net)

    al_dataset = ActiveLearningDataset(mnist, initial_label_rate=0.001)

    al_strategy = ConfidenceSamplingStrategy()

    al_system = ActiveLearningSystem(al_dataset=al_dataset, test_dataset=mnist_test, model=model,
                                     strategy=al_strategy)

    al_system.active_learning_loop()