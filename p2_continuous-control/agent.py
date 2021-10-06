from replay_buffer import ReplayBuffer
from model import QNetwork

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():

    def __init__(self, state_size, action_size, seed, device, buffer_size = int(1e5), batch_size = 64, gamma = 0.99, tau = 1e-3, learning_rate=5e-4, update_every = 4):
        """Initializes the agent
        
        Params
        ======
            state_size (int): number of parameters in the state
            action_size (int): number of available actions
            seed (int): random seed
            buffer_size (int): size of the guy's memory buffer
            batch_size (int): size of the batch when we get a random sample from the memory
            gamma (float): discount factor
            tau (float): used for soft update
            learning_rate (float): learning rate for the optimizer
            update_every (int): how many steps before we get a new batch from the memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.update_every = update_every
        self.device = device

        #-- Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        # learning_rate doesn't need to be stored, as it's not being used outside this function

        #-- We're solving this problem by using Deep Q Learning. That means...

        # 1) We have two networks: local and target
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # 2) Initialize replay memory
        self.memory = ReplayBuffer(action_size, seed, buffer_size, batch_size)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        """ Updates the agent after an action is taken """
        
        # Remember this step
        self.memory.add(state, action, reward, next_state, done)

        # Decide whether we need to get another batch of samples from the memory
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.device)
                self.learn(experiences)
        
        self.t_step += 1
        if self.t_step >= self.update_every:
            self.t_step = 0

    def learn(self, experiences):
        """ Updates value parameters using the batch of experiences"""
        states, actions, rewards, next_states, dones = experiences

        # 1) Get predicted Q targets
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # 2) Calculate Q targets
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # 3) Get expected Q from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 

        # 4) Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # 5) And minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) Finally, update target network with the weights calculated so far for the local network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def act(self, state, epsilon=0.0):
        """ Selects an action to take from the current state using the current policy """
        # Transform state
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get current action values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection (i.e. random choice instead of following policy with some probabilily)
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

