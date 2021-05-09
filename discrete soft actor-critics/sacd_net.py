"""
Two networks used for dql algorithm.
"""
import torch
from torch import nn, optim


class DenseNet(nn.Module):
    def __init__(self, input_num=512, output_num=1):
        """
        A fully connected network, including two layers, each layer has 512 neurons.

        Args:
            input_num: The number of input elements.
            output_num: The number of results elements.
        """
        super().__init__()
        self.fully_net = nn.Sequential(
            nn.Linear(input_num, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_num))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        Predict results according input tensor.
        """
        x = self.fully_net(x)
        return x


class PolicyNetwork(DenseNet):
    def __init__(self, input_num=512, output_num=1):
        """
        The policy network for Actor-Critics algorithm.

        Args:
            input_num: The number of input elements.
            output_num: The number of results elements.
        """
        super(PolicyNetwork, self).__init__(input_num, output_num)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_best_action(self, s):
        """
        Get the best action about certain state.

        Args:
            s: The state of the environment.

        Returns:
            action: The best action for this state.
        """
        action_value = self.fully_net(s)
        action = torch.argmax(action_value).item()
        return action

    def sample_action(self, s):
        """
        Sample action with their probabilities.

        Args:
            s: The state of the environment.

        Returns:
            action: The action sample from certain probabilities.
            action_prob: The probabilities for different actions.
            log_action_prob: Rescale actions' probabilities in log scale.
        """
        action_prob = torch.nn.functional.softmax(self.fully_net(s), dim=1)
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        z = (action_prob == 0.0).float() * 1e-8
        log_action_prob = torch.log(action_prob + z)
        return action, action_prob, log_action_prob
