import torch.nn as nn
import torch.nn.functional as F
import torch as T 

class model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(model, self).__init__()
        # Fully Connected Linear Layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """ Forward Pass """
        state = F.elu(self.fc1(state))      # Layer 1 FC
        return F.elu(self.fc2(state))       # Layer 2 FC