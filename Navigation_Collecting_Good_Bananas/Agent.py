import torch as T 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import random 
from model import model
from ReplayBuffer import ReplayBuffer

BUFFER_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.95
LR = 1e-4
HIDDEN_SIZE = 128
UPDATE_EVERY = 4
TAU = 'N/A' #1e-3
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size=37, action_size=4):
        """ Agent to act in world & learn from experiances """
        self.action_size = action_size 
        # Network & Optimizer
        self.local = model(state_size, action_size, HIDDEN_SIZE).to(device)
        # self.target = model(state_size, action_size, HIDDEN_SIZE).to(device)
        self.opt = T.optim.Adam(self.local.parameters(), lr=LR)
        # Experiance Replay Memory
        self.Memory = ReplayBuffer(BUFFER_SIZE, state_size)
        self.step = 0
        # Print Parameters
        print('\nBUFFER_SIZE:', BUFFER_SIZE,
        '\nBATCH_SIZE:', BATCH_SIZE,
        '\nGAMMA:', GAMMA,
        '\nLR:', LR,
        '\nHIDDEN_SIZE:', HIDDEN_SIZE,
        '\nUPDATE_EVERY:', UPDATE_EVERY,
        '\nTAU:', TAU, '\n')
        print(self.local)

    def Action(self, state, isRandom=False):
        """ Chooses Action by either Exploration(Random) or Expolation, according to isRandom:bool """
        # Random Exploration
        if isRandom: return random.choice(np.arange(self.action_size))
        # Model Explotation 
        state = T.from_numpy(state).float().unsqueeze(0).to(device)     # State to Tensor, to Device
        self.local.eval()                                               # Eval Mode
        with T.no_grad():
            actions = self.local(state)                                 # Get Action from Network
        # self.local.train()
        return int(np.argmax(actions.cpu().numpy()).squeeze())          # Choose highest Action QValue

    def Step(self, state, action, reward, next_state, done):
        """ Adds Experiance to Memory & Periodically trains Network acording to UPDATE_EVERY """
        # Add to Memory
        self.Memory.Add(state, action, reward, next_state, done)
        # Periodic Learn
        self.step += 1
        if self.step == UPDATE_EVERY: 
            self.step = 0
            # Train Model
            self._Learn()
        
    def _Learn(self):
        """ Learns from a Random Sample from the Replay Memory Buffer """
        # Exit if under BatchSize
        if len(self.Memory)<BATCH_SIZE: return
        self.local.train()          # Set in Train() Mode
        # Sample Batch to Train, and Get Outputs & Targets to Train
        states, actions, rewards, nextStates, dones = [v.to(device) for v in self.Memory.Sample(BATCH_SIZE)]
        currents, targets = self._SarsaQUpdate(self.local, states, actions, rewards, nextStates, dones)
        # MSE Loss & Update Step Network
        loss = F.smooth_l1_loss(currents, targets.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # # Soft Update Target Network
        # self._DoubleSoftUpdate()

    def _DoubleDQNUpdate(self, states, actions, rewards, nextStates, dones):
        """ Double DQN (Target & Local) Network Update Rule for Local Network """
        QExpected = self.local(states)[T.arange(len(states),device=device), actions]
        QTargets = self.target(nextStates).detach().max(1)[0]
        QTargets = rewards + (GAMMA * QTargets * (1-dones))
        return QExpected, QTargets

    def _DoubleSoftUpdate(self):
        """ Updates the Target Network 
            Udacity's Soft Update of Target Network. """
        for target_param, local_param in zip(self.target.parameters(), self.local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0-TAU) * target_param.data)

    def _SarsaQUpdate(self, model, states, actions, rewards, nextStates, dones):
        """ SarsaQ Max Update Rule """
        current = model(states)[T.arange(len(states),device=device), actions]
        QNext = model(nextStates).max(1)[0]
        target = rewards + (GAMMA * QNext * (1-dones))
        return current, target 

    def _SarsaExpUpdate(self, eps, model, states, actions, rewards, nextStates):
        """ Expected Sarsa Update Rule 
            DOES NOT WORK with this Calc. """
        current = model(states)[actions]
        QNext = model(nextStates)
        policy_s = T.ones(len(current),self.action_size,device=device) * eps / self.action_size
        policy_s[QNext.argmax(1)] = 1 - eps + (eps / self.action_size)
        target = current.clone()
        tar = T.sum(QNext * policy_s, dim=1)  # Like np.dot()
        target[T.arange(len(states),device=device), actions] = rewards + GAMMA * tar
        return current, target