import torch
import torch.nn.functional as F
import numpy as np

from torch import nn

class CNN_CIFAR10(nn.Module):
    """
    CNN model.
    The model has 2 convolutional layers followed by 2 fully connected layers.
    """
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()

        self.relu = nn.ReLU()

        # batch_size x 32 x 32 x 3
        # 32x32x3 input image size
        # (24+4-5) / 1 + 1 = 24x24 , 64 channels
        self.convLayer1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        # (28 - 3) / 2  + 1 = 13x13
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1.0)  # TODO- check

        # (13- 5) / 1  + 1 = 9x9, 64 channels
        self.convLayer2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        # (9 - 3) / 2  + 1 = 4x4 , 64 channels
        self.norm2 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1.0)  # TODO- check
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.flatten = nn.Flatten()
        self.linearLayer1 = nn.Linear(6 * 6 * 64, 384)
        self.linearLayer2 = nn.Linear(384, 192)
        self.linearLayer3 = nn.Linear(192, 10)

        # initialize the conv and linear layer's biases (following the setting in IFCA)
        nn.init.constant_(self.convLayer1.bias, 0)
        nn.init.constant_(self.convLayer2.bias, 0.1)

        nn.init.constant_(self.linearLayer1.bias, 0.1)
        nn.init.constant_(self.linearLayer2.bias, 0.1)
        nn.init.constant_(self.linearLayer3.bias, 0.0)

        self.layers = []
        self.layers.append(self.convLayer1)
        self.layers.append(self.convLayer2)
        self.layers.append(self.linearLayer1)
        self.layers.append(self.linearLayer2)
        self.layers.append(self.linearLayer3)

    def forward(self, x):
        x = self.relu(self.convLayer1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.relu(self.convLayer2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linearLayer1(x)
        x = self.linearLayer2(x)
        x = self.linearLayer3(x)
        return x

    @property
    def num_layers(self):
        return len(self.layers)

    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the weights from the linear layer of this model
        # print(self.layers[layer_num-1].weight)
        return self.layers[layer_num - 1].weight

    def get_layer_bias(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the bias from the linear layer of this model
        return self.layers[layer_num - 1].bias


class CNN_EMNIST(nn.Module):
    """
        CNN model for EMNIST dataset (the same as using in IFCA)
        The model has 2 convolutional layers followed by 2 fully connected layers.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CNN_EMNIST, self).__init__()

        self.relu = nn.ReLU()

        # batch_size x 28 x 28 x 1
        # 28 x 28 x 1 input image size

        self.convLayer1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        # padding='same' and stride=1 => 28X28, 32 channels

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (28 - 2) / 2 + 1 = 14x14, 32 channels

        self.convLayer2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        # padding='same' and stride=1 => 14x14, 64 channels

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (14 - 2) / 2 + 1 = 7x7, 64 channels

        self.flatten = nn.Flatten()
        self.linearLayer1 = nn.Linear(7 * 7 * 64, 2048)
        self.linearLayer2 = nn.Linear(2048, self.num_classes)

        self.layers = []
        self.layers.append(self.convLayer1)
        self.layers.append(self.convLayer2)
        self.layers.append(self.linearLayer1)
        self.layers.append(self.linearLayer2)

    def forward(self, x):
        x = self.relu(self.convLayer1(x))
        x = self.pool1(x)
        x = self.relu(self.convLayer2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linearLayer1(x)
        x = self.linearLayer2(x)
        return x

    @property
    def num_layers(self):
        return len(self.layers)

    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the weights from the linear layer of this model
        return self.layers[layer_num - 1].weight

    def get_layer_bias(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the bias from the linear layer of this model
        return self.layers[layer_num - 1].bias









