import numpy as np
import torch
from network_definitions import PolicyNetwork
from torch.distributions import Categorical
import torch.optim as optim
from matplotlib import pyplot as plt

class DiceGame():
    def __init__(self, size, number, traps, max_episode_steps=100):
        self.size = size
        self.number = number
        self.traps = traps
        self.state = torch.tensor([0, 10, 1])
        self.max_episode_steps = max_episode_steps
        self.action_space = 2
        self.state_space = 4
        self.terminated = False
        self.truncated = False
        
    def reset(self):
        self.state = torch.tensor([0, 10, 1])
        self.terminated = False
        self.truncated = False
        info = False
        return self.state, info
        
    
    def step(self, action):
        reward = 0
        
        if not self.terminated:
            if action == 1:
                reward = 0
                self.terminated = True
                self.state[0] = self.state[0]
                self.state[1], self.state[2] = self.get_traps(self.state[0], self.traps, self.number, self.size)
            else:
                rolls = 0
                for _ in range(self.number):
                    rolls += np.random.randint(1, self.size+1)
                
                next_state = self.state
                next_state[0] = self.state[0] + rolls
                
                if next_state[0] % self.traps == 0:
                    self.terminated = True
                    reward = -1
                    self.state[0] = next_state[0]
                    self.state[1], self.state[2] = self.get_traps(next_state[0], self.traps, self.number, self.size)
                else:
                    reward = 1
                    self.state[0] = next_state[0] # cumulative total
                    self.state[1], self.state[2] = self.get_traps(next_state[0], self.traps, self.number, self.size)
            
        info = None
        
        return self.state, reward, self.terminated, self.truncated, info
    
    def get_traps(self, state, traps, number, size):
        max_roll = number*size
        min_roll = number
        traps_list = [t for t in range(state + min_roll, state + max_roll + 1) if t % traps == 0]
        distance_to_first_trap = (traps_list[0] - state) if state % traps != 0 else 0
        num_traps = len(traps_list)
        return distance_to_first_trap, num_traps
        
    
    
def run_episode(env, policy, device):
        
    state, done = env.reset()
    gamma = 0
    G = 0
    rewards, states, actions, discounted_returns, log_probs = [], [state], [], [], []
    while not done:
        action_probs = policy(state.unsqueeze(0).to(device))
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_p = action_dist.log_prob(action)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        log_probs.append(log_p)
        rewards.append(reward)
        actions.append(action.item())
        
        state = next_state
    
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_returns.insert(0, G)
        
    discounted_returns = torch.tensor(discounted_returns, requires_grad=True, device=device)
    log_probs = torch.tensor(log_probs, requires_grad=True, device=device)
    
    return states, actions, rewards, discounted_returns, log_probs

def update_network(optimizer, log_probabilities, discounted_returns, device):
       
    # Reset the gradients
    optimizer.zero_grad()
    
    # Combine log probabilities and scale by returns
    loss = -torch.dot(discounted_returns, log_probabilities)  # Negative for gradient ascent
    #print(loss)
    
    # Backpropagate and update the network
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
        
    policy = PolicyNetwork(1, 2).to(device)
    die = DiceGame(6, 2, 10)
    state, info = die.reset()
    print(state)
    s, r, t, te, i = die.step(0)
    print(s)
    s, r, t, te, i = die.step(0)
    print(s)

