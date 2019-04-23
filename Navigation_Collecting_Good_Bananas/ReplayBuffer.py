import numpy as np 
import torch as T 

class ReplayBuffer:
    def __init__(self, size, state_size):
        """ Holds Buffer for experiance Replay """
        assert size>1
        self.size = size
        # Action, Reward, & done, all in one np.array
        self.rewdAct = np.empty((3,0), dtype=int)
        # states & nextStates np.arrays
        self.states = np.empty((0,state_size), dtype=float)
        self.nextStates = np.empty((0,state_size), dtype=float)
        # Lists to hold data until it can be appended to np.array
        self.rewdActL, self.statesL, self.nextStatesL = [], [], []

    def Add(self, state, action, reward, next_state, done):
        """ Add current experiance info to Lists """
        self.rewdActL.append([action, reward, done])
        self.statesL.append(state)
        self.nextStatesL.append(next_state)

    def Sample(self, batchSize, modMax=7, modMin=3):
        """ Random Sample data from Buffer """
        # Append Existing experiance list data to numpy arrays & clear those lists
        if len(self.statesL)>0:
            self._AppendClear()
        # Find Max Modulus value that can be used to fit the buffer size
        mod = min(int(len(self.states)/batchSize), modMax)
        mod = np.random.randint(modMin, mod) if mod>modMin else mod
        # Replay indexs to use, Random Mod sample
        idx = np.where(np.arange(len(self.states)) % mod == np.random.randint(0, mod))[0]
        idx = idx[:batchSize]        # Indexs to use
        # Return Experiance Data
        actions, rewards, dones = self.rewdAct[:, idx]
        states, nextStates = self.states[idx], self.nextStates[idx]
        # Convert Returned Experiance Data into Torch Tensor
        states, rewards, nextStates, dones = [T.from_numpy(v).float() for v in [states, rewards, nextStates, dones]]
        return states, T.from_numpy(actions).long(), rewards, nextStates, dones
        
    def _AppendClear(self):
        """ Append Existing experiance list data to numpy arrays & clear those lists """
        assert len(self.statesL) > 0
        # Append Lists to np.arrays
        self.rewdAct = np.append(np.transpose(self.rewdActL), self.rewdAct, axis=1)
        self.states = np.append(self.statesL, self.states, axis = 0)
        self.nextStates = np.append(self.nextStatesL, self.nextStates, axis = 0)
        # Clear Lists
        self.rewdActL, self.statesL, self.nextStatesL = [], [], []
        if len(self.states) <= self.size: return
        # Crop by Buffer Size
        self.rewdAct, self.states, self.nextStates = self.rewdAct[:self.size], self.states[:self.size], self.nextStates[:self.size]
    
    def __len__(self):
        """ Length of Buffer """
        return len(self.states) + len(self.statesL)
        

if __name__ == '__main__':
    import random 
    def RandAdd():
        Memory.Add(np.random.uniform(size=4), np.random.randint(0,4), np.random.randint(-1, 2), np.random.uniform(size=4), True)
    Memory = ReplayBuffer(10, 4)
    for _ in range(5): RandAdd()
    Memory.Sample(128)
    for _ in range(3): RandAdd()
    Memory.Sample(128)