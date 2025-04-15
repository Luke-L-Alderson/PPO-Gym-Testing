import torch.nn as nn
import torch.nn.functional as F

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