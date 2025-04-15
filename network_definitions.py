import torch.nn as nn
import torch.nn.functional as F

'''
The Policy Network is the Actor, so named because it dictates the action to be
taken when given a state as an input.
'''
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Input to hidden layer
        self.fch1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, output_size)  # Hidden to output layer
        
    def forward(self, state):
        x = F.relu(self.fc1(state))  # Apply ReLU activation on hidden layer
        x = F.relu(self.fch1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)  # Apply softmax to output layer
        return action_probs

'''
The Value Network is the Critic, so named because it evaluates the performance
of the Actor. In practice this means that the output of the Critic features in
the objective function used to assess and improve Actor performance.
'''
class ValueNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Input to hidden layer
        self.fch1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, output_size)  # Hidden to output layer
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fch1(x))
        x = F.relu(self.fc2(x))
        return x