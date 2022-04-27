from torch.nn import Module, Linear, ReLU, Dropout, Softmax, Conv2d, Sequential, BatchNorm2d, MaxPool2d
import torch.nn.functional as tf
import torch.nn as nn


class SimpleRHClassifier(Module):
    """
    Simple genre classifier with two hidden layers of 100 neurons each based on rhythm histograms (in-feature=60).
    """
    def __init__(self):
        super(SimpleRHClassifier, self).__init__()
        self.fc1 = Linear(60, 100)
        self.fc2 = Linear(100, 100)
        self.fc3 = Linear(100, 13)

    def forward(self, x):
        x = tf.relu(self.fc1(x))
        x = tf.relu(self.fc2(x))
        x = tf.softmax(self.fc3(x))
        return x


class RPClassifier(Module):

    def __init__(self, dropout=0.2):
        super(RPClassifier, self).__init__()

        # CONV1 (?, 60, 24, 1)
        # -> (?, 60, 24, 32)
        # -> (?, 30, 12, 32)
        self.conv1 = Sequential(
            Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(dropout)
        )

        # Conv2 (?, 30, 12, 32)
        # -> (?, 30, 12, 64)
        # -> (?, 15, 6, 64)
        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(dropout)
        )

        # CONV3 (?, 15, 6, 64)
        # -> (?, 15, 6, 128)
        # -> (?, 8, 4, 128)
        self.conv3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=1),
            Dropout(dropout)
        )

        # FC1 4096 -> 512
        self.fc1 = Sequential(
            Linear(4096, 512, bias=True),
            ReLU(),
            Dropout(dropout)
        )

        # FC2 1024 -> 13
        self.fc2 = Sequential(
            Linear(512, 13, bias=True),
        )

    def forward(self, x):
        out = self.conv1(x.view(-1, 1, 60, 24))
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
