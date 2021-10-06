import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_dims=(64, 64), activation_func=F.relu):
        super(QNetwork, self).__init__()

        self.activation_func = activation_func
        self.seed = torch.manual_seed(seed)

        # Initialize input layer
        self.input_layer = nn.Linear(state_size, hidden_dims[0])

        # Create as many hidden layers as requested in the constructor
        self.hidden_layers = nn.ModuleList();
        for i in range(len(hidden_dims) - 1):
            new_hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(new_hidden_layer)

        # Create output layer
        self.output_layer = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state):

        # Send our state through the net...
        # 1) input layer
        x = self.activation_func(self.input_layer(state))

        # 2) hidden layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_func(hidden_layer(x))

        # 3) output layer. Don't apply the activation function here
        x = self.output_layer(x)

        # We're taking a state and converting it to action values
        # by making the original input go through every layer of the network
        return x

