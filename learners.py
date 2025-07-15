from network_definitions import PolicyNetwork, ValueNetwork, PolicyNetworkConv, ValueNetworkConv
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
from helpers import save_gif
import numpy as np
from gymnasium.wrappers import RecordVideo
from datetime import datetime

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
    action_space = env.action_space[0].n
    state_space = env.observation_space.shape[-1]
    
    # Create instances of actor (policy) and critic (value) networks, and corresponding optimizers.
    policy_net = PolicyNetwork(state_space, action_space).to(device)
    old_policy_net = copy.deepcopy(policy_net)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)
    
    value_net = ValueNetwork(state_space).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr, eps=1e-5)
    
    # Set loss function to be used and init other variables
    mse_loss = nn.MSELoss()
    reward_totals = []
    full_pixels = []
    global_steps = 0
    num_episodes = 0
    
    print("\nStarting Training...")
    state, info = env.reset()
    state = torch.from_numpy(state).to(device)
    
    for n in range(iterations):
        
        buffer = perform_rollout(env, old_policy_net, value_net, T, device) # (50, )
        
        rollout_states = buffer["state"].to(device)
        rollout_actions = buffer["action"].to(device)
        rollout_values = buffer["values"].to(device)
        rollout_dones = buffer["dones"].to(device)
        old_log_probs = buffer["old_log_probs"].to(device)
        rollout_rewards = buffer["reward"].to(device)
        
        with torch.no_grad():
            next_value = value_net(rollout_states[-1, :].unsqueeze(0))
        
        #advantage = rollout_returns - rollout_values
        
        rollout_values = torch.cat([rollout_values, next_value]).squeeze()
        
        advantage, rollout_returns = compute_gae(rollout_rewards, rollout_dones, rollout_values, next_value, gamma)
        rollout_returns = rollout_returns.to(device)
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
            
        divider = round(0.05*iterations)
        
        num_episodes += rollout_dones.sum().item()
        
        if n==0 or n==iterations-1 or (n+1)%divider==0:
            eval_return, eval_num = evaluate_policy(env, policy_net, device, num_episodes=5)
            reward_totals.append(eval_return)
            if trial:
                trial.report(eval_return, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            print(f"Iter: {n+1}, R: {eval_return}")
            
    env.close()
    print(f"Finished! - Training Epsiodes: {num_episodes}\n")    
    return reward_totals

def ppo_learner_image(env, params, device, trial = False):
    
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
    entropy_coef = params.get("entropy")
    
    # Define action and state space sizes.
    # State Space --> [Cart Pos, Cart Vel, Pole Pos, Pole Vel]
    # Action Space --> [0 (left), 1 (right)]
    action_space = env.action_space.n
    state_space = np.roll(np.asarray(env.observation_space.shape), 1)
    # Create instances of actor (policy) and critic (value) networks, and
    # corresponding optimizers.
    policy_net = PolicyNetworkConv(state_space, action_space).to(device)
    old_policy_net = copy.deepcopy(policy_net)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    value_net = ValueNetworkConv(state_space, 1).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    # Set loss function to be used and init other variables
    mse_loss = nn.MSELoss()
    reward_totals = []
    global_steps = 0
    max_eval = -1
    max_index = 0
    print("\nStarting Training...")
    state, info = env.reset()
    
    state = torch.from_numpy(state).float().movedim(-1, 0).unsqueeze(0).to(device)
    #print(f"Initial State dtype: {state.dtype}, Shape: {state.shape}")
    for n in range(iterations):
        
        buffer = perform_rollout(env, old_policy_net, value_net, T, device) # (50, )
        #print(buffer["reward"])
        return_to_go = get_returns(buffer["reward"], buffer["dones"], gamma) # NEW)
        
        rollout_states = torch.stack(buffer["state"]).float().to(device)
        rollout_returns = torch.tensor(return_to_go).to(device)
        rollout_actions = torch.tensor(buffer["action"]).to(device)
        rollout_values = torch.tensor(buffer["values"]).to(device)
        old_log_probs = torch.tensor(buffer["old_log_probs"]).to(device)
        
        advantage = rollout_returns - rollout_values
        
        
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        #print(f"Passing rollout Data into Network - Rollout Shape: {rollout_states.shape}")
        action_probs = policy_net(rollout_states.movedim(-1, 1))
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(rollout_actions)
             
        ratio = torch.exp(new_log_probs - old_log_probs)
        noclip_term = advantage*ratio
        clip_term = torch.clamp(ratio, min=1-eps, max=1+eps)*advantage
        entropyt = action_dist.entropy()
        ppo_loss = -torch.min(noclip_term, clip_term).mean() - entropy_coef*entropyt.mean()
        
        policy_optimizer.zero_grad()
        ppo_loss.backward()
        policy_optimizer.step()
        
        
        training_values = value_net(rollout_states.movedim(-1, 1)).squeeze()

        value_loss = mse_loss(training_values, rollout_returns.float())
        print(f"Training Shape {training_values.shape}, Training Sum {training_values.sum()}, Rollout Shape {rollout_returns.float().shape}, Rollout Sum {rollout_returns.float().sum()}")
        print()
        print("Rollouts...)")
        print()
        #print(f"Loss Datatype: {value_loss.dtype}")
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        old_policy_net = copy.deepcopy(policy_net)
            
        divider = round(0.01*iterations)
        
        if n==0 or n==iterations-1 or (n+1)%divider==0:
            eval_return, ep_length = evaluate_policy(env, policy_net, device, num_episodes=5)
            if eval_return > max_eval:
                max_eval = eval_return
                max_index = n
            reward_totals.append(eval_return)
            if trial:
                trial.report(eval_return, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            print(f"Iter: {n+1}, R: {eval_return}, V: {value_loss}, P: {ppo_loss}, EpLen: {ep_length}, Last Action: {rollout_actions[-1]}")
    #print(rollout_states.shape)     
    #save_gif(rollout_states[max_index, :].char(), "demo.gif")
    print("Finished!\n")    
    return reward_totals

def get_returns(rewards, dones, gamma=0.99):
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        else:
            R = reward + gamma * R
        returns.insert(0, R)
    return returns

def select_action(policy, state):
    
    action_probs = policy(state)
    try:
        action_dist = Categorical(action_probs)
    except:
        print(f"State: {state.dtype} - {state.shape} - {state}")
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action, log_prob

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def perform_rollout(env, actor, critic, T, device):
    buffer = {
        "state": [],
        "action": [],
        "reward": [],
        "dones": [],
        "old_log_probs": [],
        "values": [],
        "pixels": []
        }
    
    state, info = env.reset()
    state = torch.from_numpy(state).float().to(device)
    for _ in range(T):
        with torch.no_grad():
            action, log_prob = select_action(actor, state)
            value = critic(state)
        next_state, reward, done, trunc, info = env.step(action.cpu().numpy())
        next_state = torch.from_numpy(next_state).float().to(device)
        
        buffer["state"].append(state)
        buffer["action"].append(action)
        buffer["reward"].append(reward)
        buffer["dones"].append(done)
        buffer["old_log_probs"].append(log_prob)
        buffer["values"].append(value) #env.render is numpy array
        #print((env.render().shape))
        #buffer["pixels"].append(rgb2gray(env.render()))
        state = next_state.float()
        
        if len(state.shape) == 1 and (done or trunc):
            state, info = env.reset()
            state = torch.from_numpy(state).float().to(device)
    
    buffer["state"] = torch.cat(buffer["state"])
    buffer["action"] = torch.cat(buffer["action"])
    buffer["values"] = torch.cat(buffer["values"])
    buffer["dones"] = torch.tensor(np.concat(buffer["dones"]))
    buffer["old_log_probs"] = torch.cat(buffer["old_log_probs"])
    buffer["reward"] = torch.tensor(np.concat(buffer["reward"]))
        
    return buffer
    
def evaluate_policy(env, policy_net, device, num_episodes=5):
    returns = []
    #print("Evaluating Current Policy")
    for num in range(num_episodes):
        state, info = env.reset()
        states = [state]
        done = False
        trunc = False
        ep_return = 0
        while not (done or trunc):
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().to(device)
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs, dim=-1)
            state, reward, done, trunc, info = env.step(action.cpu().numpy())
            #print(f"{num}: {reward}")
            states.append(state)
            done = done[-1]
            trunc = trunc[-1]
            
            ep_return += reward[-1]
        returns.append(ep_return)
    return sum(returns) / len(returns), len(states)

def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> (torch.Tensor, torch.Tensor):

    T = rewards.size(0)
    # buffer for advantages
    
    advantages = torch.zeros_like(rewards)
    
    # start GAE accumulator at zero (shape = batch‐shape)
    gae = torch.tensor(0)

    # append V(s_{T}) so we can always look “one step ahead”
    #next_value = next_value.unsqueeze(0)          # shape (1, N) or (1,)
    #values = torch.cat([values, next_value], dim=0)

    for t in reversed(range(T)):
        # mask = 0 if done, 1 otherwise
        mymask = (~dones[t]).float()
        # TD error δ_t = r_t + γ·V_{t+1}·mask − V_t
        delta = rewards[t] + gamma * values[t + 1] * mymask - values[t]
        # GAE recursion
        gae = delta + gamma * lam * mymask * gae
        advantages[t] = gae

    # compute discounted returns R_t = A_t + V_t
    returns = advantages + values[:-1]
    return advantages, returns.float()

'''
Generator function for vectorised environments
'''
def make_env(environment = 'CartPole-v1', seed = 42, idx = 0, max_epsiode_steps = 200, capture_video = False):
    def thunk():
        trigger = lambda t: t % 100 == 0
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        env = gym.make(environment, max_episode_steps = max_epsiode_steps, render_mode='rgb_array')
        #env.seed(seed)
        #env.action_space.seed(seed)
        #env.observation_space.seed(seed)
        env = RecordVideo(env, video_folder=f"./training_videos", episode_trigger=trigger, disable_logger=True)
        return env
    
    return thunk

class PPOagent():
    def __init__(self,
                 environment = 'CartPole-v1',
                 lr = 1e-6,
                 eps = 1e-5,
                 max_episode_steps = 200, 
                 rollout_len = 128,
                 training_steps = 10000,
                 vectorised_envs = False,
                 record_video = False,
                 gamma = 0.99,
                 lam = 0.95,
                 device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                 ):
        
        # init hyperparams
        
        # init environment
        if vectorised_envs:
            assert not vectorised_envs, "Not yet implemented, set vectorised_envs to False"
        else:
            env = gym.make(environment, max_episode_steps = max_episode_steps, render_mode='rgb_array')
        
        if record_video:
            assert not vectorised_envs, "No recording support for vectorised environments in gym"
            trigger = lambda t: t % 100 == 0
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            env = RecordVideo(env, video_folder=f"./training_videos", episode_trigger=trigger, disable_logger=True)
        
        # init networks and optimizers
        action_space = env.action_space.n
        state_space = np.asarray(env.observation_space.shape)[0]
        
        policy_net = PolicyNetwork(state_space, action_space).to(device)
        old_policy_net = copy.deepcopy(policy_net)
        value_net = ValueNetwork(state_space).to(device)
        
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=eps)
        value_optimizer = optim.Adam(value_net.parameters(), lr=lr, eps=eps)
        
        
        # Define action and state space sizes.
        # State Space --> [Cart Pos, Cart Vel, Pole Pos, Pole Vel]
        # Action Space --> [0 (left), 1 (right)]
        


        
        # Set loss function to be used and init other variables
        mse_loss = nn.MSELoss()
        reward_totals = []
        global_steps = 0
        max_eval = -1
        max_index = 0
        
    def _init_hyperparams(self, ):
        
        pass
        
    def rollout():
        pass
    
    def calculate_gae(rewards: torch.Tensor,
                      dones: torch.Tensor,
                      values: torch.Tensor,
                      next_value: torch.Tensor,
                      gamma: float = 0.99,
                      lam: float = 0.95
                      ) -> (torch.Tensor, torch.Tensor):

        T = rewards.size(0)
        
        advantages = torch.zeros_like(rewards)
        
        # start GAE accumulator at zero (shape = batch‐shape)
        gae = torch.zeros_like(next_value)
    
        # append V(s_{T}) so we can always look “one step ahead”
        #next_value = next_value.unsqueeze(0)          # shape (1, N) or (1,)
        values = torch.cat([values, next_value], dim=0)
    
        for t in reversed(range(T)):
            # mask = 0 if done, 1 otherwise
            mymask = (~dones[t]).float()
            # TD error δ_t = r_t + γ·V_{t+1}·mask − V_t
            delta = rewards[t] + gamma * values[t + 1] * mymask - values[t]
            # GAE recursion
            gae = delta + gamma * lam * mymask * gae
            advantages[t] = gae
    
        # compute discounted returns R_t = A_t + V_t
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update_params():
        pass
    
    
    
    
    
        