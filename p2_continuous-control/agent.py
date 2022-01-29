from model import ActorCritic

import numpy as np
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim

T_MAX = 5

class Agent:
    def __init__( self, state_size, action_size, actor_layer_size, critic_layer_size, gamma ):
        self.local_actor_critic = ActorCritic( state_size, action_size, actor_layer_size, critic_layer_size, gamma )
        
class AgentOverlord:

    def __init__( self, state_size, action_size, agent_num, actor_layer_size=128, critic_layer_size = 128, gamma = 0.99, learning_rate=5e-4 ):
        self.state_size = state_size
        self.action_size = action_size
        
        #-- We're solving this problem using A3C, which means we need a global actor critic (that will be shared
        #   among every agent), and we'll have a local actor critic per agent
        self.global_actor_critic = ActorCritic( state_size, action_size, actor_layer_size, critic_layer_size, gamma )
        self.agents = [ Agent( state_size, action_size, actor_layer_size, critic_layer_size, gamma ) for _ in range( agent_num ) ]
        
        #-- Store gamma and initialize optimizer. The optimizer is shared between all of the agents
        self.gamma = gamma
        self.optimizer = optim.Adam( self.global_actor_critic.parameters(), lr = learning_rate, betas=(0.92, 0.999) )
        
        self.t_step = 1
        
    def step( self, states, actions, rewards, dones ):				
        for state, action, reward, done, agent in zip( states, actions, rewards, dones, self.agents ):
            agent.local_actor_critic.memory.remember( state, action, reward )

            if self.t_step % T_MAX == 0 or done:
                loss = agent.local_actor_critic.calculate_loss( done )
                self.optimizer.zero_grad()
                loss.backward()
                for local_param, global_param in zip( agent.local_actor_critic.parameters(), self.global_actor_critic.parameters() ):
                    if global_param.grad is None:
                        global_param._grad = local_param.grad
                
                self.optimizer.step()
                
                # synchronize with global
                agent.local_actor_critic.load_state_dict( self.global_actor_critic.state_dict() )
                
                # clear local memory after every step
                agent.local_actor_critic.memory.clear()
                
            agent.done = done

        self.t_step += 1
            
    def clear( self ):
        for agent in self.agents:
            agent.local_actor_critic.memory.clear()
        
        self.t_step = 1
                         
    def choose_actions( self, states ):
        actions = []
        for agent, state in zip( self.agents, states ):
            action = agent.local_actor_critic.choose_action( state )
            actions.append(action.tolist() )
        
        return actions
 

