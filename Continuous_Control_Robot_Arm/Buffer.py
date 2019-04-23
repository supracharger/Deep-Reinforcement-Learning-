import numpy as np
import random
from collections import namedtuple, deque
import torch

class ReplayBuffer2:
    """Fixed-size buffer to store experience tuples.
        CITED: Udacity's ddpg_agent.py"""

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def addMulti(self, state, action, reward, next_state, done, limit=-1):
        """ Add to Buffer from Multiple different Agents. """
        if limit<0: limit = len(state)      # Limit number of Agents to add to Buffer
        for i in range(limit):
            self.add(state[i], action[i], reward[i], next_state[i], done[i])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # Get Type from experiances
        statesL, actionsL, rewardsL, next_statesL, donesL = [], [], [], [], []
        for e in experiences:
            if e is None: continue
            statesL.append(e.state), actionsL.append(e.action), rewardsL.append(e.reward)
            next_statesL.append(e.next_state), donesL.append(e.done)
        # To Torch Tensor
        states = torch.from_numpy(np.vstack(statesL)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actionsL)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewardsL)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_statesL)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(donesL).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)