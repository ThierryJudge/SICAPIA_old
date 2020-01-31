import os

from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.networks.CNNNet import CNNNet
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == '__main__':
    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    net = CNNNet((1, 28, 28), 10)
    model = ActiveLearningModel(net=net)

    model.train(train_dataset=mnist, val_dataset=mnist_test, epochs=10)
    model.eval(test_dataset=mnist_test)