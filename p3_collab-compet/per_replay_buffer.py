"""
Code adapted from Jonathan Pearce's implementation of a PrioritizedReplayBuffer:
https://github.com/Jonathan-Pearce/DDPG_PER/blob/master/PER_buffer.py
"""

import numpy as np
import random

import utils


class ReplayBuffer(object):
    def __init__(self, size, batch_size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self):
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        return self._encode_sample(indices)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, batch_size, alpha, beta, beta_increment_per_sampling):
        super(PrioritizedReplayBuffer, self).__init__(size, batch_size)
        assert alpha >= 0
        self._alpha = alpha
        
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self):
        # increment beta every time we sample
        # The original implementation by Pearce used a linear schedule to calculate beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        indices = self._sample_proportional(self.batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self.beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(indices)
        return tuple(list(encoded_sample) + [weights, indices])

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)