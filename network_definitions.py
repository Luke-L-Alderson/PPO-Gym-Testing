import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Layer initialisation function - set to PPO paper inits
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Combined value/policy network
class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),  # Input to hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std = 1.0),   # Hidden to output layer
            )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),  # Input to hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std = 0.01),
            nn.Softmax(-1)
            )
        
    def get_val(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action = None):
        probs = Categorical(self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        

# Seperate value and policy networks - LEGACY
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Input to hidden layer
        self.fch1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, output_size)  # Hidden to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation on hidden layer
        x = F.relu(self.fch1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)  # Apply softmax to output layer
        return action_probs 

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Input to hidden layer
        self.fch1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 1)  # Hidden to output layer
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fch1(x))
        x = F.relu(self.fc2(x))
        return x

# Convolutional agent for pixel inputs - WIP
class ConvAgent(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()
        
        
        input_channels, h, w = input_shape
        self.conv1 = nn.Conv2d(input_channels, 16, 3) # Input to hidden layer
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3  = nn.Conv2d(32, 64, 3)
        fc_shape = self._get_conv_size(input_shape).prod()
        #print(f"conv_size_func: {fc_shape}")
        self.fc1 = nn.Linear(fc_shape, action_space)  # Hidden to output layer
    
    def _get_conv_size(self, shape):
        if type(shape) is not tuple:
            shape = tuple(shape)
        x = torch.zeros(shape)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        return torch.tensor(x.shape)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        #print(f"Before Flattening: {x.shape}")
        x = x.flatten(1)
        #print(f"After Flattening: {x.shape}")
        action_probs = F.softmax(self.fc1(x), dim=-1)  # Apply softmax to output layer
        return action_probs   
