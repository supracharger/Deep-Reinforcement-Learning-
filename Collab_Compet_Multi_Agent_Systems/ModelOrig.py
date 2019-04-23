import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
""" CITED: Udacity's ddpg-pendulum """

def hidden_init(layer):
    """ Udacity's Weight Initialization Technique. """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)     # Fully Connected 1
        self.fc2 = nn.Linear(fc1_units, fc2_units)      # Fully Connected 2
        self.fc3 = nn.Linear(fc2_units, action_size)    # Fully Connected 3
        self.reset_parameters()

    def reset_parameters(self):
        """ Udacity's Weight Initialization Technique. """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))     # Relu Fully Connected 1
        x = F.relu(self.fc2(x))         # Relu Fully Connected 2
        return F.tanh(self.fc3(x))      # Tanh Fully Connected 3


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)               # Fully Connected 1
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)     # Fully Connected 2
        self.fc3 = nn.Linear(fc2_units, 1)                          # Fully Connected 3
        self.reset_parameters()

    def reset_parameters(self):
        """ Udacity's Weight Initialization Technique. """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))           # FC: Encode state Input
        x = torch.cat((xs, action), dim=1)      # Concat Encoded State & Input Action
        x = F.relu(self.fc2(x))                 # Relu Fully Connected 2
        return self.fc3(x)                      # Linear Fully Connected 3
