[//]: # (Image References)

[image1]: imgReacher.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This project involves the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

### State & Action Spaces

The observation State space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### Solving the Environment

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- (Used for this Project) The second version contains 20 identical agents, each with its own copy of the environment.  

**This project uses the Asynchronous DDPG Algorithm, and gathers experience with a single replay buffer.**

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: (NOT USED).**

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.


2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!   

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

### Dependencies
* from unityagents import UnityEnvironment
* import numpy as np
* import random
* import torch as T
* from collections import deque
* import matplotlib.pyplot as plt
* Internal Classes Agent 

**Use "pip install [Your Package]" to install any needed dependencies.**

### Instructions Running Code
* Create Unity Environment Version 2 by “env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')”
* Step Environment: “env.step(actions)[brain_name]”
* Environment Info: “env_info = env.step(actions)[brain_name]”; nextStates: “env_info.vector_observations”; Rewards: “np.array(env_info.rewards)”; Dones: “env_info.local_done”
* Create Agent Object “agent = Agent(limit, noise:OUNoise, random_seed=2)”
* Get Action from agent: “actions = agent.Action(states, isRandom=eps>random.random())”
* Save to Replay buffer per time step: “agent.Step(states, actions, rewards, next_states, dones, limit)”
* Train Agent: “agent._Train(limit)”
* ‘limit’: is the number to limit Environment Agents. Max value is 20
* Reference “Continuous_Control.ipynb” Jupiter Notebook for specifics.


### CITED
Referenced from Udacity's Continuous Control Project:
https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control


