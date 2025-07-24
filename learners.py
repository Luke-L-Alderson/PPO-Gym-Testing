from network_definitions import Agent, ConvAgent, ValueNetwork, PolicyNetwork
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import copy
import optuna
import numpy as np
from gymnasium.wrappers import RecordVideo
from helpers import make_env, make_env_t, stack_frames
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

class PPOagent(): # does not do value clipping, debug variables, or KL early stopping

    '''
    Description: OOP-approach to applying Proximal Policy Optimisation (PPO) to the provided environment.
    Returns: A list of returns for plotting
    '''

    def __init__(self,
                 trial = False,
                 env_name = 'CartPole-v1',
                 lr = 1e-6,
                 eps = 1e-5,
                 entropy_coef = 0.01,
                 value_coef = 0.5,
                 trunc = 200,
                 max_episode_steps = 200, 
                 rollout_len = 128,
                 training_steps = 30000,
                 num_minibatches = 1,
                 update_epochs = 4,
                 vectorised_envs = True,
                 num_envs = 2,
                 record_video = False,
                 gamma = 0.99,
                 lam = 0.95,
                 annealing = False,
                 global_gradient_norm = 0.5,
                 state_space_not_pixels = True,
                 device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 num_frames = 4
                 ):
        
        # init hyperparams
        self.trial = trial
        self.env_name = env_name
        self.lr = lr
        self.eps = eps
        self.max_episode_steps = max_episode_steps 
        self.rollout_len = rollout_len
        self.training_steps = int(training_steps)
        self.vectorised_envs = vectorised_envs
        self.record_video = record_video
        self.gamma = gamma
        self.lam = lam
        self.num_envs = num_envs
        self.device = device
        self.annealing = annealing
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.trunc = trunc
        self.batch_size = int(self.num_envs*self.rollout_len)
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = int(self.training_steps // self.batch_size)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.global_gradient_norm = global_gradient_norm
        self.num_frames = num_frames
        
        new_shape = (84, 84)
        t1 = v2.Compose([   #[envs, stack, H, W, C]
                            v2.Lambda(lambda a: torch.movedim(a, -1, -3)),              # [envs, stack, C, H, W]
                            v2.Lambda(lambda x: v2.functional.rgb_to_grayscale(x)),     # grayscale: [envs, stack, 1, H, W]
                            v2.Lambda(lambda x: x.squeeze(2) if x.shape[-3]==1 else stack_frames(x)),                          # remove channel dim: [envs, stack, H, W]
                            v2.ToDtype(torch.float32),
                            v2.Resize(new_shape)
                            ]) #[envs, C*stack, H, W]
        
        # init environment
        if vectorised_envs:
            if self.env_name == "tetris_gymnasium/Tetris":
                
                self.env = gym.vector.SyncVectorEnv([make_env_t(environment=self.env_name, max_epsiode_steps=trunc, capture_video=record_video, frames=self.num_frames)]*self.num_envs)
                self.env = gym.wrappers.vector.RecordEpisodeStatistics(self.env)
                self.env = gym.wrappers.vector.NumpyToTorch(self.env, self.device)
                self.env = gym.wrappers.vector.TransformObservation(self.env, lambda x: t1(x))
                self.env.metadata['render_fps'] = 30
            else:
                self.env = gym.vector.SyncVectorEnv([make_env(environment=self.env_name, max_epsiode_steps=trunc, capture_video=record_video)]*self.num_envs)
                self.env = gym.wrappers.vector.RecordEpisodeStatistics(self.env)
                self.env = gym.wrappers.vector.NumpyToTorch(self.env, self.device)
            
            
        else:
            self.env = gym.make(env_name, max_episode_steps = max_episode_steps, render_mode='rgb_array')
        
        assert isinstance(self.env.single_action_space, gym.spaces.Discrete), "Discrete Spaces Only"
        
        self.action_space = self.env.single_action_space.n
        self.state_space = self.env.single_observation_space if env_name != "tetris_gymnasium/Tetris" else torch.zeros((self.num_frames,)+new_shape)
        # init networks and optimizers
        self.agent = Agent(self.env).to(device) if state_space_not_pixels else ConvAgent(self.env).to(device)
        #self.old_agent = copy.deepcopy(self.agent)
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=eps)
        #self.mse_loss = nn.MSELoss()
        
    def train_agent(self):
        writer = SummaryWriter()
        self.global_steps = 0
        start_time = time.time()
        next_obs, info = self.env.reset()
        obs = torch.zeros((self.rollout_len, self.num_envs) + self.state_space.shape).to(self.device)
        actions = torch.zeros((self.rollout_len, self.num_envs)).to(self.device)
        log_probs = torch.zeros((self.rollout_len, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.rollout_len, self.num_envs)).to(self.device)
        dones = torch.zeros((self.rollout_len, self.num_envs)).to(self.device)
        values = torch.zeros((self.rollout_len, self.num_envs)).to(self.device)
        

        next_done = torch.zeros(self.num_envs).to(self.device)
        next_trunc = torch.zeros(self.num_envs).to(self.device)
        reporting_rewards = []
        for update in range(1, self.num_updates+1):
            if self.annealing:
                frac = 1 - (update - 1) / self.num_updates
                lrnow = frac*self.lr
                self.optimizer.param_groups[0]["lr"] = lrnow
        
            # record a rollout
            for step in range(self.rollout_len):
                self.global_steps+=1*self.num_envs
                obs[step] = next_obs
                
                dones[step] = next_done
                
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                
                actions[step] = action
                log_probs[step] = log_prob
                next_obs, reward, next_done, next_trunc, info = self.env.step(action)
                '''
                if self.env_name == "tetris_gymnasium/Tetris":
                    reward[~(next_done|next_trunc)]+=0.1 # external small reward per timestep if not done or truncated
                '''
                rewards[step] = reward
                
                last_episode_returns = self.print_info(info, self.global_steps, printing = True)
                if last_episode_returns is not None:
                    reporting_rewards.append(last_episode_returns)
                    if type(last_episode_returns[0]) is float:
                        writer.add_scalar("Reward", last_episode_returns[0], global_step=self.global_steps)
                    
                    if self.trial:
                        if type(last_episode_returns[0]) is float:
                            self.trial.report(last_episode_returns[0], step=self.global_steps)
                            if self.trial.should_prune():
                                raise optuna.TrialPruned()
            
            
            # calculate advantages
            advantages, returns = self.calculate_gae(next_obs, next_done, rewards, dones, values)
            
            # calculate losses
            batch_obs = obs.reshape((-1,)+self.state_space.shape)
            batch_actions = actions.reshape(-1)
            batch_logprobs = log_probs.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values.reshape(-1)
            
            # Minibatch Training
            batch_indices = np.arange(self.batch_size)
            
            for epoch in range(self.update_epochs):
                #print(f"learning {epoch+1}/{self.update_epochs+1}")
                np.random.shuffle(batch_indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = batch_indices[start:end]
                    
                    _, newlogprob, entropy, new_values = self.agent.get_action_and_value(batch_obs[minibatch_indices], batch_actions[minibatch_indices])
                    logratio = newlogprob - batch_logprobs[minibatch_indices]
                    ratio = logratio.exp()
                    
                    minibatch_advantages = batch_advantages[minibatch_indices]
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                    noclip_term = minibatch_advantages*ratio
                    clip_term = torch.clamp(ratio, min=1-self.eps, max=1+self.eps)*minibatch_advantages
                    policy_loss = -torch.min(noclip_term, clip_term).mean()
                    value_loss = 0.5 * ((new_values.view(-1) - batch_returns[minibatch_indices])**2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss - self.entropy_coef * entropy_loss + self.value_coef * value_loss
                    
                    # update network
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.global_gradient_norm)
                    self.optimizer.step()
            
        
        
        writer.close()
        stop_time = time.time() - start_time
        print(f"{stop_time} seconds. ({stop_time/60} minutes).")
        self.env.close()
        return [x for x in reporting_rewards if x is not None]
    
    
    def calculate_gae(self, next_obs, next_done, rewards, dones, values):
        with torch.no_grad():
            next_value = self.agent.get_val(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.rollout_len)):
                if t == self.rollout_len - 1:
                    nextnonterminal = ~next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns
    
    
    def print_info(self, info, update=0, printing = False):
        if info.get("_episode") is not None:
            finished_episodes = [(idx) for (idx, val) in enumerate(info.get("_episode").tolist()) if val is True]
            if printing:
                print(f"Update {update}/{self.training_steps} - Finished Episode(s) {finished_episodes} - Episode Return(s): {info.get("episode").get("r").tolist()}")
            result_dict = {key: [] for key in range(self.num_envs)}   
            
            for index in finished_episodes:
                result_dict[index] = info.get("episode").get("r")[index].item()
            
            return result_dict

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
    Description: Accepts an environment and a parameter dictionary and applies Advantage Actor Critic (A2C) to the provided environment.
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
    LEGACY
    
    Description: Accepts an environment and a parameter dictionary and applies Proximal Policy Optimisation (PPO) to the provided environment.
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
        
        buffer = perform_rollout(env, state, old_policy_net, value_net, T, device) # (50, )
        
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
        state = rollout_states[-1, :]
        
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
    
if __name__ == "__main__":
    agent = PPOagent()
    agent.train_agent()
    
    
        