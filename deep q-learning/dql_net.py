"""
A deep Q-learning network used for dql algorithm.
"""
import torch
import numpy as np

from torch import nn, optim
from general_net import CnnExtractor


class DqlNet(CnnExtractor):
    def __init__(self, img_size: tuple, out_channels: int):
        """
        Include a 2D convolution network, and the hidden layer use linear function with 512 neurons.
        The lose function is Mse, and the optimiser choose Adam algorithm with learn rate 1e-4.

        Args:
            img_size: A tuple in (in_channels, height, width) form.
            out_channels: Num of the output action.
        """
        super().__init__(img_size)
        self.fully_net = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels))

        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        """
        Predict action through input images.

        Args:
            x: A tensor includes the information of images.

        Returns:
            prediction: A tensor includes the prediction value of all actions.
        """
        x = self.conv(x)
        prediction = self.fully_net(x)
        return prediction
