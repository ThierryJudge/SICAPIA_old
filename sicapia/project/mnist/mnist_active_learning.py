import argparse
import os

from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from sicapia.ActiveLearningDataset import ActiveLearningDataset
from sicapia.ActiveLearningModel import ActiveLearningModel
from sicapia.ActiveLearningStrategy import AL_STRATEGIES, get_strategy
from sicapia.ActiveLearningSystem import ActiveLearningSystem
from sicapia.networks.LinearNet import LinearNet
from sicapia.utils.metrics import accuracy

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--strategy', default='random', choices=AL_STRATEGIES,
                               help='Active learning sampling strategy')
    parent_parser.add_argument('--al_iters', default=10, type=int,
                               help='Number of active learning itterations')
    args = ActiveLearningModel.add_model_specific_args(parent_parser)
    params = args.parse_args()
    print(params)

    mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

    al_dataset = ActiveLearningDataset(mnist_train, initial_label_rate=0.001)

    net = LinearNet((1, 28, 28), 10, activation=F.log_softmax)
    metrics = [accuracy]
    model = ActiveLearningModel(network=net, train_dataset=al_dataset, val_dataset=mnist_test, test_dataset=mnist_test,
                                hparams=params, loss_fn=F.nll_loss, metrics=metrics)

    al_strategy = get_strategy(params.strategy)()

    al_system = ActiveLearningSystem(al_dataset=al_dataset, test_dataset=mnist_test, model=model,
                                     strategy=al_strategy)

    al_system.active_learning_loop()