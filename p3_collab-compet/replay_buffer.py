import numpy as np
import random
import torch
from collections import namedtuple, deque
from SumTree import SumTree

# Replay buffer (for PER)
class ReplayBuffer:
    def __init__(self, action_size, seed, buffer_size = int(1e5), batch_size = 64, epsilon = 0.01, alpha = 0.6, beta = 0.4, beta_increment_per_sampling = 0.001):
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTree(buffer_size)

    def get_priority(self, error):
        return ( np.abs(error) + self.epsilon ) ** self.alpha

    def add(self, error, state, action, reward, next_state, done):
        p = self.get_priority(error)
        e = self.experience(state, action, reward, next_state, done)
        self.tree.add(p, e)
    
    def sample(self, device):
        batch = []
        indices = []
        segment = self.tree.total() / self.batch_size
        priorities = []
        
        # increment beta every time we sample
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range( self.batch_size ):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform( a, b )
            (index, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            indices.append(index)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones, indices, is_weight)

    def update(self, index, error):
        p = self.get_priority(error)
        self.tree.update(index, p)

    def __len__(self):
        return self.tree.n_entries