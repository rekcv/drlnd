import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# Util function to intialize a hidden layer
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(fan_in)
    return (-lim, lim)


#-- MODEL -----------------------------------------------------------------------------------------------------------------
class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden, activation_functions, seed):
        super(Actor, self).__init__()

        self.seed = T.manual_seed(seed)
        self.state_size = state_size
        self.activation = activation_functions
        
        # Initialize our layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Starting from an input layer that takes the state...
        previous_size = state_size
        
        # ... add as many hidden layers as requested. We'll also apply batch normalization
        # at each layer
        for layer_size in hidden:
            self.layers.append( nn.Linear( previous_size, layer_size ) )
            self.batch_norms.append( nn.BatchNorm1d( previous_size ) )
            previous_size = layer_size
 
        # The final layer outputs the action
        self.batch_norms.append( nn.BatchNorm1d( previous_size ) )
        self.layers.append( nn.Linear( previous_size, action_size ) )

        # Initialize the model's parameters
        self.reset_parameters()
    
    
    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        
        last_layer = self.layers[-1]
        last_layer.weight.data.uniform_(-3e-3, 3e-3)


    def forward( self, state ):
        if len(state) == self.state_size:
            state = state.unsqueeze(0)
            
        # pass state through the model
        x = state
        for layer, batch_norm, activation in zip( self.layers, self.batch_norms, self.activation ):
            x = activation( layer( batch_norm( x ) ) )

        return x
        
        
class Critic(nn.Module):

    def __init__( self, state_size, action_size, hidden, seed ):
        super(Critic, self).__init__()

        self.seed = T.manual_seed(seed)
        
        # Our input layer receives the state
        self.input_layer = nn.Linear( state_size, hidden[ 0 ] )
        self.input_batch_norm = nn.BatchNorm1d( hidden[ 0 ] )
        
        # Now, we'll create as many hidden layers as requested
        self.layers = nn.ModuleList()        
        
        # Note the first hidden layer will have some extra "input neurons" that will receive the action
        previous_size = hidden[0] + action_size
        
        for layer_size in hidden[1:]:
            self.layers.append( nn.Linear( previous_size, layer_size ) )
            previous_size = layer_size

        # And finally, we output a single value
        self.layers.append( nn.Linear( previous_size, 1 ) )

        # Initialize the model's parameters
        self.reset_parameters()

        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
    
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        
        last_layer = self.layers[-1]
        last_layer.weight.data.uniform_(-3e-3, 3e-3)


    def forward( self, state, action ):
        # pass state through the input layer, and batch normalize
        xs = F.relu( self.input_layer( state ) )
        xs = self.input_batch_norm( xs )
        
        # now, the input for the rest of the network is state + action
        x = T.cat((xs, action), dim=1)
        
        for layer in self.layers:
            x = F.relu( layer(x) )
        
        return x    
