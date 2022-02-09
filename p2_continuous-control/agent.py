from model import ActorCritic

import numpy as np
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim

# A3C
class AgentOverlord:

    def __init__(   self, state_size, action_size, agent_num, 
                    actor_hidden, actor_activation, actor_mu_activation, actor_sigma_activation,
                    critic_hidden, critic_activation, 
                    gamma, learning_rate, update_global_step ):
        self.state_size = state_size
        self.action_size = action_size
        self.update_global_step = update_global_step
        
        #-- We're solving this problem using A3C, which means we need a global actor critic (that will be shared
        #   among every agent), and we'll have a local actor critic per agent
        self.global_actor_critic = ActorCritic( state_size, action_size, actor_hidden, actor_activation, actor_mu_activation, actor_sigma_activation, critic_hidden, critic_activation, gamma )
        self.agents = [ ActorCritic( state_size, action_size, actor_hidden, actor_activation, actor_mu_activation, actor_sigma_activation, critic_hidden, critic_activation, gamma ) for _ in range( agent_num ) ]
        
        #-- Store gamma and initialize optimizer. The optimizer is shared between all of the agents
        self.gamma = gamma
        self.optimizer = optim.Adam( self.global_actor_critic.parameters(), lr = learning_rate )
        
        self.t_step = 1
        
    def step( self, states, actions, rewards, dones ):

        # get the agents to remember this new data
        for state, action, reward, done, agent in zip( states, actions, rewards, dones, self.agents ):
            agent.memory.remember( state, action, reward )
              
            if self.t_step % self.update_global_step == 0 or done:
            
                # use samples in the memory to calculate losses for the agent
                loss = agent.calculate_loss( done )
                
                # Set gradient parameters back to zero
                self.optimizer.zero_grad()
                
                # back propagate
                loss.backward()
                
                # copy parameters to the global model
                for local_param, global_param in zip( agent.parameters(), self.global_actor_critic.parameters() ):
                    global_param._grad = local_param.grad

                # step optimizer
                self.optimizer.step()
                
                # copy parameters from global back to the local model
                agent.load_state_dict( self.global_actor_critic.state_dict() )
                
                # clear local memory after every step
                agent.memory.clear()

        self.t_step += 1
            
    def reset( self ):
        for agent in self.agents:
            agent.memory.clear()
        
        self.t_step = 1
                         
    def choose_actions( self, states ):
        actions = []
        for agent, state in zip( self.agents, states ):
            action = agent.choose_action( state )
            actions.append(action.tolist() )
        
        return actions
 

