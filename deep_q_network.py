"""
Reference and credit:
https://www.youtube.com/watch?v=UlJzzLYgYoE

The ConvNet the parameterize the Action-value function Q(s,a)
Input: Observation (Pixel)
Output: estimated Q(s,a)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork_CNN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, cheakpoint_dir, name):
        """
        Init the network
        :param lr: Learning Rate
        :param n_actions: number of action
        :param input_dims: Channel following Pytorch convention (batch, channel, height, width)
        :param cheakpoint_dir: Path to store the weight in .pth file
        :param name: Name of the checkpoint file
        """
        super(DeepQNetwork_CNN, self).__init__()
        self.checkpoint_dir = cheakpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        """
        Network structure:
        3 Layer ConvNet -> 2 Layer FC layer -> output
        """
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        # TODO: poor me no GPU, build a colab training script
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        """
        A dummy forward pass to get the output dimension after the ConvNet,
        The output is flatten and feed into fully connected layers
        :param input_dims: input dimension
        :return: flatten dimension
        """
        # *expression to unroll a tuple
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        conv_state = conv3.flatten(start_dim=1)  # don't flatten the batch dimension

        flat1 = F.relu(self.fc1(conv_state))
        actions = F.relu(self.fc2(flat1))

        return actions

    def save_checkpoint(self):
        print('Save model weight at dir: {}'.format(self.checkpoint_dir))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading weight')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DeepQNetwork_FC(nn.Module):
    def __init__(self, lr, n_actions, input_dims, cheakpoint_dir, name):
        """
        Init the network
        :param lr: Learning Rate
        :param n_actions: number of action
        :param input_dims: Channel following Pytorch convention (batch, channel, height, width)
        :param cheakpoint_dir: Path to store the weight in .pt file
        :param name: Name of the checkpoint file
        """
        super(DeepQNetwork_FC, self).__init__()
        self.checkpoint_dir = cheakpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".pt")

        """
        Network structure:
        3 Layer ConvNet -> 2 Layer FC layer -> output
        """
        self.fc1 = nn.Linear(input_dims[0], 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, n_actions)

        self.head = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        # TODO: poor me no GPU, build a colab training script
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        actions = F.relu(self.fc3(fc2))

        return actions

    def save_checkpoint(self):
        print('Save model weight at dir: {}'.format(self.checkpoint_dir))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading weight')
        self.load_state_dict(torch.load(self.checkpoint_file))