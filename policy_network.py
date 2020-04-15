"""
The policy that output the probability of the actions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PolicyNetwork_FC(nn.Module):
    def __init__(self, lr, n_actions, input_dims, cheakpoint_dir, name):
        """
        Init the network
        :param lr: Learning Rate
        :param n_actions: number of action
        :param input_dims: Channel following Pytorch convention (batch, channel, height, width)
        :param cheakpoint_dir: Path to store the weight in .pt file
        :param name: Name of the checkpoint file
        """
        super(PolicyNetwork_FC, self).__init__()
        self.checkpoint_dir = cheakpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".pt")

        self.fc1 = nn.Linear(input_dims[0], 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, n_actions)

        self.head = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # TODO: poor me no GPU, build a colab training script
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))

        actions = self.head(fc3)  # softmax to normalise the action probability

        return actions

    def save_checkpoint(self):
        print('Save model weight at dir: {}'.format(self.checkpoint_dir))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading weight')
        self.load_state_dict(torch.load(self.checkpoint_file))