import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal


#-- MEMORY USED BY THE MODEL ----------------------------------------------------------------------------------------------
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


#-- MODEL -----------------------------------------------------------------------------------------------------------------
class ActorCritic(nn.Module):

    def __init__(   self, input_size, action_size, actor_hidden, actor_activation, actor_mu_activation, actor_sigma_activation,
                    critic_hidden, critic_activation, 
                    gamma):
        super(ActorCritic, self).__init__()


        self.gamma = gamma
        self.actor_activation = actor_activation
        self.actor_mu_activation = actor_mu_activation
        self.actor_sigma_activation = actor_sigma_activation
        self.critic_activation = critic_activation

        # Actor
        self.actor = []
        previous_size = input_size
        for layer_size in actor_hidden:
            self.actor.append( nn.Linear( previous_size, layer_size ) )
            previous_size = layer_size
        
        self.actor_mu = nn.Linear( previous_size, action_size )
        self.actor_sigma = nn.Linear( previous_size, action_size )


        # Critic
        self.critic = []
        previous_size = input_size
        for layer_size in critic_hidden:
            self.critic.append( nn.Linear( previous_size, layer_size ) )
            previous_size = layer_size
        
        self.critic_v = nn.Linear( previous_size, 1 )

        # Memory
        self.memory = ModelMemory()
        

    def forward( self, states ):
        
        # Update actor
        x = states
        for layer, activation in zip( self.actor, self.actor_activation):
            x = activation( layer( x ) )
                
        mu = self.actor_mu_activation( self.actor_mu( x ) )
        sigma = self.actor_sigma_activation( self.actor_sigma( x ) ) + 0.001       # avoid zeros
        
        # Update critic
        x = states
        for layer, activation in zip( self.critic, self.critic_activation):
            x = activation( layer( x ) )
            
        v = self.critic_v( x )
    
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
        log_prob = distribution.log_prob( actions )
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + T.log(distribution.scale)
        
        critic_loss = critic_loss.reshape([returns.size(0), 1])
        exp_v = log_prob * critic_loss.detach() + 0.005 * entropy
        
        actor_loss = -exp_v
        
        total_loss = T.mean( critic_loss + actor_loss )        
        return total_loss
            
    def choose_action( self, state ):
        state = T.tensor( [state], dtype= T.float )
        mu, sigma, _ = self.forward( state )
        
        distribution = Normal( mu, sigma )
        action = distribution.sample()[0].numpy()
        action = np.clip(action, -1, 1)
        return action
