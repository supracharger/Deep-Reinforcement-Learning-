import torch as T 
import torch.nn.functional as F
import torch.optim as optim
import random
from ModelOrig import *
from Buffer import ReplayBuffer2

BUFFER_SIZE = int(5e5)  # Replay
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
UPDATE_MULTI = 100
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class Agent():
    """CITED: Udacity's ddpg_agent.py"""
    def __init__(self, limit, noise, random_seed, n_state=24, n_action=2):
        """ Initializes Actor & Critic Models plus the Replay Buffer. """
        # Actor Model: Local, Target, & Optimizer
        self.Actor = Actor(n_state, n_action, random_seed).to(device)            
        self.actorOpt = optim.Adam(self.Actor.parameters(), lr=LR_ACTOR)
        self.ActorTarget = Actor(n_state, n_action, random_seed).to(device)                
        # Critic Model: Local, Target, & Optimizer
        self.Critic = Critic(n_state, n_action, random_seed).to(device)
        self.CriticTarget = Critic(n_state, n_action, random_seed).to(device)
        self.criticOpt = optim.Adam(self.Critic.parameters(), lr=LR_CRITIC)
        # Replay Buffer
        self.Memory = ReplayBuffer2(n_action, BUFFER_SIZE, BATCH_SIZE, device, random_seed)
        self.step = 0
        # OUNoise Process
        self.noise = noise
        # Display
        print('\nBUFFER_SIZE', BUFFER_SIZE,
        '\nBATCH_SIZE', BATCH_SIZE,
        '\nGAMMA', GAMMA,
        '\nTAU', TAU,
        '\nLR_ACTOR', LR_ACTOR,
        '\nLR_CRITIC', LR_CRITIC,
        '\nUPDATE_MULTI', UPDATE_MULTI)
        # Display Actor & Critic
        print('\nACTOR[i]:\n', self.Actor)
        print('CRITIC:\n', self.Critic)

    def Action(self, states, eps=0, isRandom=False):
        """ Returns actions for given state as per current policy. """
        action = []
        # No gradient needed
        with T.no_grad():
            states = T.Tensor(states).float().to(device)                    # Send states to GPU
            self.Actor.eval()                                  # Eval Mode
            # Loop Each Actor Model to get Action as np.array()
            action = self.Actor(states).cpu().data.numpy()
            self.Actor.train()                                 # Back to Train Mode
        # Add Epsilon Greedy Noise
        action += self.Noise(1 if isRandom else eps, action.shape)
        return action

    def Step(self, state, action, reward, next_state, done, limit=-1):
        """ Add Experiance to Replay Buffer. """
        self.Memory.addMulti(state, action, reward, next_state, done, limit)

    def _Train(self):
        """ Train the model of 'limit' with n number of Rounds """
        if len(self.Memory)>BATCH_SIZE: 
            # 'n' number of rounds to train      
            for _ in range(UPDATE_MULTI):
                # Get Batch Data
                experiances = self.Memory.sample()
                # Train Models
                self._Learn(self.Actor, self.ActorTarget, self.actorOpt, experiances)

    def _Learn(self, Actor, ActorTarget, actorOpt, experiances):
        """ Train/ Update Actor & Critic Models """
        Actor.train()      # Set in Train Mode
        # Get split experiances into: states, actions ...
        states, actions, rewards, nextStates, dones = experiances
        # ....................... Update Critic .......................
        QTargetsNext = self.CriticTarget(nextStates, ActorTarget(nextStates))
        QTargets = rewards + (GAMMA * QTargetsNext * (1 - dones))
        QExpected = self.Critic(states, actions)
        # Minimize Loss & Update Weights
        critic_loss = F.smooth_l1_loss(QExpected, QTargets.detach())
        self.criticOpt.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm(self.Critic.parameters(), 1)
        self.criticOpt.step()
        # ....................... Update Actor .......................
        actor_loss = -self.Critic(states, Actor(states)).mean()
        # Update Weights
        actorOpt.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm(Actor.parameters(), 1)
        actorOpt.step()
        # ............. Update Actor & Critic Target Nets .............
        self.SoftUpdate(self.Critic, self.CriticTarget, TAU)
        self.SoftUpdate(Actor, ActorTarget, TAU)

    def SoftUpdate(self, local, target, tau):
        """ Udacity's Soft Update Method. """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def Noise(self, eps, size):
        """ Random [-1,1] Noise scaled by Epsilon. """
        return eps * (np.random.uniform(size=size) * 2 - 1)

    def _TransAction(self, actionI, actSz):
        """ Change action from Conituous Action space to Deterministic. 
            OLD & Not Used"""
        action = np.zeros(actionI.shape, dtype=float)       # Continuous Action
        # Action Index to Continuous Action Size
        for i in range(len(actionI)):
            action[i] = self.actDist[actionI[i]]            # To Deterministic Action
        return action