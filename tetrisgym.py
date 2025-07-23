import cv2
import gymnasium as gym
from helpers import save_gif, rgb2gray, make_env
from tetris_gymnasium.wrappers.observation import RgbObservation
from gymnasium.wrappers import TransformObservation, RecordVideo
import matplotlib.pyplot as plt
import numpy as np
from tetris_gymnasium.mappings.rewards import RewardsMapping
from torchvision.transforms import v2
import torch
import os
from network_definitions import ConvAgent
from collections import defaultdict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def make_env_t(environment = 'tetris_gymnasium/Tetris', seed = 42, idx = 0, max_epsiode_steps = 200, capture_video = False):
    def thunk():
        rewardmap = RewardsMapping(game_over=-10, alife=0, clear_line=1)
        env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array", rewards_mapping = rewardmap)
        env = RgbObservation(env)
        if env.render_mode == "rgb_array": env.render_scaling_factor = 10

        
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, video_folder="./training_videos", episode_trigger=lambda t: t % 10000 == 0, disable_logger=True)
        
        return env
    
    return thunk

def run_test_episode(env):
    print("Running a random episode.")
    obs, info = env.reset(seed=42)
    valdict = defaultdict(list)
    terminated = [False]*obs.shape[0]
    while not any(terminated):
        action = torch.from_numpy(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        valdict["obs"].append(obs)
        valdict["r"].append(reward)
    
    print("Done!")
    return valdict, obs
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose([
                        v2.Lambda(lambda b: torch.unsqueeze(b, 0) if b.dim()==3 else b),
                        v2.Lambda(lambda a: torch.movedim(a, -1, 1)), # (1, 24, 34, 3) or (N, 24, 34, 3)
                        v2.Grayscale(),
                        v2.ToDtype(torch.float32),
                        v2.Normalize([0], [1]),
                        ])
    
    env = gym.vector.SyncVectorEnv([make_env_t(environment="tetris_gymnasium/Tetris", max_epsiode_steps=200, capture_video=False)]*2)
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    env = gym.wrappers.vector.NumpyToTorch(env, device)
    env = gym.wrappers.vector.TransformObservation(env, lambda x: transform(x))
    
    '''
    # Works to establish a non-vector environment with video and logging
    
    
    rewardmap = RewardsMapping(game_over=-10, alife=0, clear_line=1)
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array", rewards_mapping = rewardmap)
    env = RgbObservation(env)
    if env.render_mode == "rgb_array": env.render_scaling_factor = 10
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    env = gym.wrappers.NumpyToTorch(env)
    env = TransformObservation(env, lambda x: transform(x), observation_space=None)
    env = RecordVideo(env, video_folder="./training_videos", episode_trigger=lambda t: t % 1 == 0, disable_logger=True)
    '''
    _, final_obs = run_test_episode(env)
    
    
    agent = ConvAgent(env).to(device)
    agent.get_val(final_obs)
    agent.get_action_and_value(final_obs)
    
    plt.imshow(torch.movedim(final_obs[0], 0, -1).cpu())
    
    #obs_list = np.stack(obs_list)
    #save_gif(obs_list, filename="demo.gif", fps=10, resize_to=(600, 400))

    
    
    print("Game Over!")
    env.close()