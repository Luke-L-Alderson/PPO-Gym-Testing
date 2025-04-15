from network_definitions import PolicyNetwork, ValueNetwork
import torch
import torch.optim as optim
import torch.nn as nn
from env import DiceGame, run_episode, update_network
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import gym
from scipy.stats import entropy

def reinforce_learner(env, params, device):
    
    ''' 
    Accepts an environment and a parameter dictionary and applies REINFORCE to the provided environment
    Returns: A list of returns for plotting
    '''
    
    episodes = params.get('episodes')
    lr = params.get('lr')
    gamma = params.get('gamma')
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    
    policy_net = PolicyNetwork(state_space, action_space).to(device) #test
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    reward_totals = []
    
    for n in range(episodes):
        rewards, states, actions, discounted_returns, log_probs = [], [], [], [], []
        done = False
        trunc = False
        
        G = 0
        steps = 0
        
        state, info = env.reset()
        state = torch.from_numpy(state).to(device)
        
        while not done and not trunc:
            action_probs = policy_net(state)
                
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done, trunc, info = env.step(action.item())
            
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
            
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            steps+=1
            state = torch.from_numpy(next_state).to(device)
        
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
            
        # Reset the gradients
        optimizer.zero_grad()
            
        # Combine log probabilities and scale by returns
        log_probs = torch.stack(log_probs)
        discounted_returns = torch.tensor(discounted_returns, device=device)
        
        loss = -torch.dot(log_probs, discounted_returns)  # Negative for gradient ascent
        
        # Backpropagate and update the network
        loss.backward()
        optimizer.step()
        
        reward_totals.append(sum(rewards))
        
        if (n+1)%100==0:
            print(f"Episode {n+1}, t = {steps}")
        
    return reward_totals

def a2c_learner(env, params, device):
    
    ''' 
    Description: Accepts an environment and a parameter dictionary and applies
    Advantage Actor Critic (A2C) to the provided environment.
    
    Returns: A list of returns for plotting
    '''
    
    # Get hyperparameter values from the parameter dictionary
    episodes = params.get('episodes')
    lr = params.get('lr')
    gamma = params.get('gamma')
    
    # Define action and state space sizes.
    # State Space --> [Cart Pos, Cart Vel, Pole Pos, Pole Vel]
    # Action Space --> [0 (left), 1 (right)]
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    
    # Create instances of actor (policy) and critic (value) networks, and
    # corresponding optimizers.
    policy_net = PolicyNetwork(state_space, action_space).to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    value_net = ValueNetwork(state_space, 1).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    # Set loss function to be used and init other variables
    mse_loss = nn.MSELoss()
    done, trunc = False, False
    reward_totals = []
    print("\nStarting Training...")
    
    for n in range(episodes):
        
        # Start a new episode
        
        state, info = env.reset()
        steps, rewards = 0, 0
        state = torch.from_numpy(state).to(device)
        finished = False
        
        while not finished:
            # Select an action using the Actor
            action_probs = policy_net(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            
            # Take the chosen action and observe the next state and reward
            next_state, reward, done, trunc, info = env.step(action.item())
            
            finished = done or trunc
            next_state = torch.from_numpy(next_state).to(device)
            
            # Compute the advantage
            state_value = value_net(state)
            next_state_value = value_net(next_state)
            
            # Use Temporal Difference (TD) target as an estimate for the state-action value
            TD_target = reward + gamma*next_state_value.detach()*(1-finished)
            
            # Calculate advantage
            advantage = reward + gamma*next_state_value.detach()*(1-finished) - state_value

            # Calculate the policy loss (from policy gradient proof). Negate
            # the product for gradient ascent. 
            log_prob = action_dist.log_prob(action)
            policy_loss = -log_prob*advantage.detach()
            
            # Policy --> Zero gradients, backprop, step.
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            # Calculate the MSE between TD Target and State Value.
            value_loss = mse_loss(TD_target, state_value)
            
            # Value --> Zero gradients, backprop, step.
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        
            # Increment counters and set new state
            rewards += reward
            steps += 1
            state = next_state            
            
        reward_totals.append(rewards)
        
        if (n+1)%100==0:
            print(f"Ep: {n+1}, R: {rewards}, policy: {action_probs.cpu().detach().numpy()}, vloss: {value_loss.cpu().detach().numpy()}, ploss: {policy_loss.cpu().detach().numpy()}")
    
    print("Finished!\n")    
    return reward_totals