import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import defaultdict

def moving_avg(data, kernel):
    return np.convolve(data, np.ones(kernel), 'valid') / kernel

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
    
    state = state if state.dim >= 2 else state.unsqueeze(0)
    
    action_probs = policy(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action, log_prob

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def perform_rollout(env, state, actor, critic, T, device):
    buffer = {
        "state": [],
        "action": [],
        "reward": [],
        "dones": [],
        "old_log_probs": [],
        "values": []
        }
    
    #state, info = env.reset()
    #state = torch.from_numpy(state).float().to(device) 
    print(state.shape)
    for _ in range(T):
        with torch.no_grad():
            action, log_prob = select_action(actor, state.unsqueeze(0))
            value = critic(state)
        print(action.shape)
        next_state, reward, done, trunc, info = env.step(action.squeeze().cpu().numpy())
        next_state = torch.from_numpy(next_state).float().to(device)
        
        buffer["state"].append(state)
        buffer["action"].append(action)
        buffer["reward"].append(reward)
        buffer["dones"].append(done)
        buffer["old_log_probs"].append(log_prob)
        buffer["values"].append(value)
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
    for num in range(num_episodes):
        state, info = env.reset()
        states = [state]
        done = [False, False]
        trunc = [False, False]
        ep_return = 0
        while not (any(done) or any(trunc)):
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().to(device)
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs, dim=-1)
                
            state, reward, done, trunc, info = env.step(action.cpu().numpy())
            states.append(state)
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

    for t in reversed(range(T)):
        # mask = 0 if done, 1 otherwise
        mymask = (~dones[t]).float()
        
        # TD error δ_t = r_t + γ·V_{t+1}·mask − V_t
        delta = rewards[t] + gamma * values[t + 1] * mymask - values[t]
        
        gae = delta + gamma * lam * mymask * gae
        advantages[t] = gae

    # compute discounted returns R_t = A_t + V_t
    returns = advantages + values[:-1]
    return advantages, returns.float()


# Generator function for vectorised environments
def make_env(environment = 'CartPole-v1', seed = 42, idx = 0, max_epsiode_steps = 200, capture_video = False):
    def thunk():
        env = gym.make(environment, max_episode_steps = max_epsiode_steps, render_mode='rgb_array')
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, video_folder="./training_videos", episode_trigger=lambda t: t % 100 == 0, disable_logger=True)
        
        return env
    
    return thunk

def save_gif(frames: np.ndarray, filename: str = "tetris.gif", fps: int = 10, resize_to: tuple[int, int] = None):
    t, c, h, w = frames.shape
    frames.reshape((t, h, w, c))
    assert frames.ndim == 4 and frames.shape[-1] == 3, "Expected shape [t, h, w, 3]"
    assert frames.dtype == np.uint8, "Frames must be of dtype uint8"
    if resize_to is not None:
        resized_frames = [cv2.resize(f, resize_to, interpolation=cv2.INTER_NEAREST) for f in frames]
    else:
        resized_frames = frames
    imageio.mimsave(filename, resized_frames, fps=fps)
    print(f"Saved {len(frames)} frames to {filename}")
    
def convert_to_one_dict(list_of_dicts):
    result = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            if value:
                result[key].append(value)
    return dict(result)