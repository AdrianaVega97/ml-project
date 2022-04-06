from torch.nn import Module, Linear
import torch.nn.functional as tf


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
