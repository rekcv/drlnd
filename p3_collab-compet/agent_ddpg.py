from model_ddpg import Actor
from model_ddpg import Critic
from ounoise import OUNoise
from replay_buffer import ReplayBuffer

import numpy as np
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# DDPG for multiple agents
class AgentOverlord():

    def __init__(   self,
                    device,
                    state_size, action_size, num_agents, 
                    actor_hidden, actor_activation,
                    critic_hidden, 
                    gamma, tau, actor_learning_rate, critic_learning_rate, 
                    buffer_size, batch_size, 
                    random_seed, 
                    epsilon, alpha, beta, beta_increment_per_sampling):
        
        self.device = device
        self.action_size = action_size
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.random_seed = random.seed(random_seed)
      
        # Initialize Actor
        self.actor_local = Actor(state_size, action_size, actor_hidden, actor_activation, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, actor_hidden, actor_activation, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = actor_learning_rate)

        # Initialize Critic
        self.critic_local = Critic(state_size, action_size, critic_hidden, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, critic_hidden, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = critic_learning_rate)

        # Initialize noise generator - noise will be applied to the selected actions in "act"
        self.noise = OUNoise((num_agents, action_size), random_seed)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(action_size, random_seed, buffer_size, batch_size, epsilon, alpha, beta, beta_increment_per_sampling)
        
    def append_sample(self, state, action, reward, next_state, done):
        target = self.critic_local(Variable(T.FloatTensor(state))).data
        old_val = target[0][action]
        target_val = self.critic_target(Variable(T.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * T.max(target_val)
            
        error = abs(old_val - target[0][action])
            
        self.memory.add(error, state, action, reward, next_state, done)
        
    def step( self, states, actions, rewards, next_states, dones ):

        # We're receiving multiple samples (one per agent). Let's add them all to our replay buffer
        for state, action, reward, next_state, done in zip( states, actions, rewards, next_states, dones ):
            self.append_sample(state, action, reward, next_state, done)

        # If we have enough samples in our Replay Buffer...
        if len( self.memory ) > self.batch_size:
            
            # ... sample a batch of experiences...
            experiences = self.memory.sample(self.device)

            # ... and use it to train the model
            self.learn( experiences )
            
            
    def learn( self, experiences ):
        
        states, actions, rewards, next_states, dones, indices, is_weight = experiences
        
        # Select the actions we should take next, using the target actor 
        actions_next = self.actor_target( next_states )
        
        # Get the critic to value the actions the actor has just taken
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * ( 1 - dones ) )
        Q_expected = self.critic_local( states, actions )
        
        # Calculate the critic loss...
        critic_loss = (T.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_targets)).mean()
        
        # ... and minimize that loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Calculate the actions our current local actor would select for the received states...
        actions_pred = self.actor_local(states)
        
        # ... and calculate the loss 
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # update priority
        for i in range(self.batch_size):
            index = indices[i]
            self.memory.update(index, errors[i])
        
        # Now, minimize the loss for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Perform a soft update of the weights of the local critic and actor based on the targets
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)      


    def act(self, states):
        states = T.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.num_agents, self.action_size))
        
        # using the local actor, calculate what actions we should take
        self.actor_local.eval()
        with T.no_grad():
            for num, state in enumerate(states):
                actions[num,:] = self.actor_local(state).cpu().data.numpy()

        # Train the local actor
        self.actor_local.train()
        
        # Add some noise to the calculated actions...
        actions += self.noise.sample()
           
        # ... and make sure their values stay in the range [-1, 1], as expected by the environment
        return np.clip(actions, -1, 1)
        
    
    def soft_update(self, local_model, target_model):
        for (target_param, local_param) in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    
    def reset(self):
        self.noise.reset()

        
    def save(self):
        T.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        T.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')

        
    def load(self):
        self.actor_local.load_state_dict( T.load('checkpoint_actor.pth') )
        self.critic_local.load_state_dict( T.load('checkpoint_critic.pth') )
        