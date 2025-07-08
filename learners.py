from network_definitions import PolicyNetwork, ValueNetwork
import torch
import torch.optim as optim
import torch.nn as nn
from env import DiceGame, run_episode, update_network
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym
from scipy.stats import entropy
import copy
import optuna

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

def ppo_learner(env, params, device, trial = False):
    
    ''' 
    Description: Accepts an environment and a parameter dictionary and applies
    Advantage Actor Critic (A2C) to the provided environment.
    
    Returns: A list of returns for plotting
    '''
    
    # Get hyperparameter values from the parameter dictionary
    iterations = params.get('iterations')
    lr = params.get('lr')
    gamma = params.get('gamma')
    T = params.get('rollout_len')
    eps = params.get('eps')
    
    # Define action and state space sizes.
    # State Space --> [Cart Pos, Cart Vel, Pole Pos, Pole Vel]
    # Action Space --> [0 (left), 1 (right)]
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    
    # Create instances of actor (policy) and critic (value) networks, and
    # corresponding optimizers.
    policy_net = PolicyNetwork(state_space, action_space).to(device)
    old_policy_net = copy.deepcopy(policy_net)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    value_net = ValueNetwork(state_space, 1).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    # Set loss function to be used and init other variables
    mse_loss = nn.MSELoss()
    reward_totals = []
    global_steps = 0
    print("\nStarting Training...")
    state, info = env.reset()
    state = torch.from_numpy(state).to(device)
    
    for n in range(iterations):
        
        buffer = perform_rollout(env, old_policy_net, value_net, T, device) # (50, )
        
        #return_to_go = get_returns(buffer["reward"], gamma) # (50, ) OLD
        return_to_go = get_returns(buffer["reward"], buffer["dones"], gamma) # NEW
        
        rollout_states = torch.stack(buffer["state"]).to(device)
        rollout_returns = torch.tensor(return_to_go).to(device)
        rollout_actions = torch.tensor(buffer["action"]).to(device)
        rollout_values = torch.tensor(buffer["values"]).to(device)
        old_log_probs = torch.tensor(buffer["old_log_probs"]).to(device)
        
        advantage = rollout_returns - rollout_values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        action_probs = policy_net(rollout_states)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(rollout_actions)
             
        ratio = torch.exp(new_log_probs - old_log_probs)
        noclip_term = advantage*ratio
        clip_term = torch.clamp(ratio, min=1-eps, max=1+eps)*advantage
        ppo_loss = -torch.min(noclip_term, clip_term).mean()
        
        policy_optimizer.zero_grad()
        ppo_loss.backward()
        policy_optimizer.step()
        
        
        training_values = value_net(rollout_states).squeeze()
        value_loss = mse_loss(training_values, rollout_returns)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        old_policy_net = copy.deepcopy(policy_net)
            
        divider = round(0.01*iterations)
        
        if n==0 or n==iterations-1 or (n+1)%divider==0:
            eval_return = evaluate_policy(env, policy_net, device, num_episodes=5)
            reward_totals.append(eval_return)
            if trial:
                trial.report(eval_return, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            print(f"Iter: {n+1}, R: {eval_return}")
            
    
    print("Finished!\n")    
    return reward_totals

# def get_returns(rewards, gamma=0.99):
#     returns = []
#     for i in range(len(rewards)):
#         R = 0
#         for reward in reversed(rewards[i:]):
#             R = reward + gamma * R
#         returns.append(R)
#     return returns

def get_returns(rewards, dones, gamma=0.99):
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)
    return returns

def select_action(policy, state):
    action_probs = policy(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action, log_prob

def perform_rollout(env, actor, critic, T, device):
    buffer = {
        "state": [],
        "action": [],
        "reward": [],
        "dones": [],
        "old_log_probs": [],
        "values": []
        }
    
    state, info = env.reset()
    state = torch.from_numpy(state).to(device)
    
    for _ in range(T):
        with torch.no_grad():
            action, log_prob = select_action(actor, state)
            value = critic(state)
            
        next_state, reward, done, trunc, info = env.step(action.item())
        next_state = torch.from_numpy(next_state).to(device)
        
        buffer["state"].append(state)
        buffer["action"].append(action)
        buffer["reward"].append(reward)
        buffer["dones"].append(done or trunc)
        buffer["old_log_probs"].append(log_prob)
        buffer["values"].append(value)
        
        state = next_state
        
        if done or trunc:
            state, info = env.reset()
            state = torch.from_numpy(state).to(device)
            
    return buffer
    
def evaluate_policy(env, policy_net, device, num_episodes=5):
    returns = []
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        trunc = False
        ep_return = 0
        while not (done or trunc):
            with torch.no_grad():
                state_tensor = torch.tensor(state).to(device)
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
            state, reward, done, trunc, info = env.step(action)
            ep_return += reward
        returns.append(ep_return)
    return sum(returns) / len(returns)

class PPOAgent(nn.Module):
    def __init__(self, env, critic, actor):
        '''

        Parameters
        ----------
        env : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        pass
    
    def perform_policy_rollout():
        pass
    
    def evaluate_performance():
        pass
        