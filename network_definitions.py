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
    def __init__(self, env, kernel_size = 3):
        super(ConvAgent, self).__init__()
        print(len(env.observation_space.shape))
        self.env = env
        assert len(env.observation_space.shape) == 3 or len(env.observation_space.shape) == 4, "Input must be an image of shape (c, h, w) or (N, c, h, w)"
        self.batch, self.h, self.w, self.c = env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2], 1
        
        action_space_size = env.single_action_space.n if len(env.action_space) > 1 else env.action_space.n
        
        assert self.c == 1, "Grayscale only"
        
        print(env.observation_space.shape)
        print(self.c)
        self.conv_layers = nn.Sequential(nn.Conv2d(self.c, 64, kernel_size),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 64, kernel_size),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 64, kernel_size),
                                         nn.Flatten()
            )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self._get_conv_size(), 512)),  # Input to hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std = 1.0),   # Hidden to output layer
            )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self._get_conv_size(), 512)),  # Input to hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space_size), std = 0.01),
            nn.Softmax(-1)
            )
    
    def _get_conv_size(self):
        x = torch.zeros((self.batch, self.c, self.h, self.w))
        print(x.shape)
        x = self.conv_layers(x)
        print(x.shape)
        return x.shape[-1]
    
    def get_val(self, x):
        x = self.conv_layers(x)
        return self.critic(x)
    
    def get_action_and_value(self, x, action = None):
        x = self.conv_layers(x)
        probs = Categorical(self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
