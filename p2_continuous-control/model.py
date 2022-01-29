import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

#-- MEMORY USED BY THE MODEL
class ModelMemory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def remember(self, state, action, reward):
        self.states.append( state )
        self.actions.append( action )
        self.rewards.append( reward ) 
        
    def get_states( self ):
        return T.tensor( self.states, dtype=T.float )
    
    def get_actions( self ):
        return T.tensor( self.actions, dtype=T.float )
        
    def calculate_batch_R( self, R, gamma ):
        # from "Multicore Deep Reinforcement Learning - https://www.youtube.com/watch?v=OcIx_TBu90Q"
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + gamma * R
            batch_return.append( R )
            
        # we need to reverse the order of the array to get them in the same order as they were seen
        batch_return.reverse()
        
        return T.tensor( batch_return, dtype=T.float )

#-- MODEL
class ActorCritic(nn.Module):

    def __init__(self, input_size, action_size, actor_layer_size=128, critic_layer_size = 128, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.base = nn.Sequential( nn.Linear( input_size, actor_layer_size ), 
                                    nn.ReLU())

        # Initialize actor network
        # -- This network will calculate the policy
        #    Our actions are continuous, so we will use a Normal distribution to sample them
        #    To do that, we want to calculate mean and standard variation for each element of the action
        self.actorMu = nn.Sequential( nn.Linear( actor_layer_size, action_size ), nn.Tanh() )  #use tanh since the values can be -1..1
        self.actorSigma = nn.Sequential( nn.Linear( actor_layer_size, action_size ), nn.Softplus() ) #softplus will allow some negative values to be transformed to positive
        
        # Initialize critic network
        # -- This network will calculate the value of the actions chosen by the actor
        self.critic = nn.Linear( critic_layer_size, 1 )

        # Memory
        self.memory = ModelMemory()
        
        # Hyperparameters
        self.gamma = gamma
        
        self.action_size = action_size

    def forward( self, state ):
        # first, map inputs to hidden layer
        base_out = self.base( state )
    
        # update actor
        mu = self.actorMu( base_out )
        sigma = self.actorSigma( base_out )

        # update critic
        v = self.critic( base_out )
    
        return mu, sigma, v
        
    def calculate_R( self, done ):
        states = self.memory.get_states()
        _, _, v = self.forward( states )

        # value of the last step, or zero if it's a terminal state
        R = 0 if done else v[ -1 ]

        batch_return = self.memory.calculate_batch_R( R, self.gamma )
        return batch_return
        
    def calculate_loss( self, done ):
        states = self.memory.get_states()
        actions = self.memory.get_actions()
        returns = self.calculate_R( done )
        
        mu, sigma, values = self.forward( states )
        
        values = values.squeeze()
        critic_loss = ( returns - values ) ** 2
 
        distribution = Normal( mu, sigma )
        log_probabilities = distribution.log_prob( actions )
        actor_loss = -log_probabilities * ( returns - values )

        total_loss = ( critic_loss + actor_loss ).mean()
        return total_loss
            
    def choose_action( self, state ):
        state = T.tensor( [state], dtype= T.float )
        mu, sigma, _ = self.forward( state )

        distribution = Normal( mu, sigma )
        action = distribution.sample()[0].numpy()
        action = np.clip(action, -1, 1)

        return action

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